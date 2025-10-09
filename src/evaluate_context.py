import torch
from TrackMamba import TrackMamba
from config import TrackMambaConfig
import pandas as pd
import argparse
from datasets import Dataset
from typing import Tuple, List
from torch.utils.data import DataLoader
from tqdm import tqdm
from rich.table import Table
from rich.console import Console
import matplotlib.pyplot as plt
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

def save_plot(out: torch.Tensor, true: torch.Tensor, short_context_pred: torch.Tensor, out_name: str, image_path: str):

    # Convert to numpy for plt
    out = out.numpy()
    true = true.numpy()
    short_context_pred = short_context_pred.numpy()

    # Plot
    plt.figure(figsize=(12, 4))

    # True signal
    plt.plot(true, label="True", color="#1f77b4", linewidth=2)

    # Model output
    plt.plot(out.squeeze(), label="Mamba-DNA", color="#FF0C0C", linewidth=2, alpha=0.85)

    # Plot short context prediction
    short_context_pred = short_context_pred.squeeze() # because of the unsqueeze for inference
    training_len = len(short_context_pred)
    x_extra = range(len(true) - training_len, len(true))
    plt.plot(x_extra, short_context_pred, color="orange", linewidth=2.5, label=f"Short Context prediction")

    # Vertical line for model context (4k)
    plt.axvline(training_len, color="black", linestyle="--", linewidth=1.5, label="4k context limit")

    # Styling
    # Remove top and right spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xlabel("Genomic position")
    plt.ylabel("DNase-seq intensity")
    plt.title(out_name)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(image_path, f"{out_name}.png"), dpi=300)
    plt.show()


def format_df(df_path: str) -> Tuple[Dataset, List[int]]:

    print("Loading dataframe...")

    # Load dataset using pandas and convert it to HF
    # Convert to HF dataset
    df = pd.read_json(df_path)

    # Remove possible duplicates from concurrency
    df = df.drop_duplicates("idx")

    # Add seq_len information
    df['seq_len'] = df['end'] - df['start']
    seq_lenghts = df['seq_len'].unique().tolist()

    # Add peak information
    df['peak'] = df['chrom'] + ":" + df['peak_start'].astype(str) + ":" + df['peak_end'].astype(str)

    print("Dataframe loaded!")

    return df, seq_lenghts


def collate_fn(batch, tokenizer):

    signals = [torch.tensor(ex["signal"], dtype=torch.float32) for ex in batch]
    signals = torch.stack(signals)  # if all same length

    idxs = [ex["idx"] for ex in batch]
    peak_starts = torch.tensor([ex["peak_start"] for ex in batch], dtype=torch.int32)
    peak_ends = torch.tensor([ex["peak_end"] for ex in batch], dtype=torch.int32)

    starts = torch.tensor([ex["start"] for ex in batch], dtype=torch.int32)
    ends = torch.tensor([ex["end"] for ex in batch], dtype=torch.int32)
    chroms = [ex["chrom"] for ex in batch]

    sequences = [ex["sequence"] for ex in batch]
    encoded = tokenizer(
        sequences,
        add_special_tokens=False,
        return_tensors='pt'
    )['input_ids']

    return {
        "signal": signals,
        "idxs": idxs,
        "peak_starts": peak_starts,
        "peak_ends": peak_ends,
        "input_ids": encoded,
        "starts": starts,
        "ends": ends,
        "chroms": chroms
    }

def print_summary_table(seq_lengths, mse_results):
    console = Console()
    table = Table(title="Mean MSE per Sequence Length")
    table.add_column("Seq Length", justify="center")
    table.add_column("Mean MSE", justify="center")

    for i, seq_len in enumerate(seq_lengths):
        mean_mse = mse_results[:, i].mean().item()
        table.add_row(str(seq_len), f"{mean_mse:.6f}")

    console.print(table)

def main():
    
    p = argparse.ArgumentParser()
    p.add_argument("--df_path", type=str, required=True, help="Path to the JSON dataset.")
    p.add_argument("--num_layers", type=int, required=True, help="Number of TrackMamba layers.")
    p.add_argument("--hidden_dim", type=int, required=True, help="Residual stream dimensionality.")
    p.add_argument("--use_MLP", action='store_true', default=False)
    p.add_argument("--use_pos_embs", action='store_true', default=False)
    p.add_argument("--model_path", type=str, required=True, help="Path to the trained model.")
    p.add_argument("--batch_size", type=int, required=False, default=32, help="Batch size for inference")
    p.add_argument("--save_path", type=str, required=True, help="Where to save MSE values")
    p.add_argument("--image_path", type=str, required=True, help="Where to save the images")

    args = p.parse_args()

    # Create out directory
    os.makedirs(args.image_path, exist_ok=True)

    # Load dataframe
    df, seq_lengths = format_df(args.df_path)

    # Initialize model and tokenizer
    cfg = TrackMambaConfig(
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        use_MLP=args.use_MLP,
        use_pos_embs=args.use_pos_embs
    )
    model = TrackMamba.from_pretrained(
        args.model_path,
        config=cfg
    ).to(device)
    tokenizer = model.cfg.tokenizer

    # Initialize tensor for final results
    all_idxs = df['peak'].unique()
    mse_results = torch.zeros((len(all_idxs), len(seq_lengths)))

    # Cycle over seq_lenghts
    for i, seq_len in enumerate(seq_lengths):

        # Filter dataset 
        subset_df = df[df['seq_len'] == seq_len]

        # Convert to HF Dataset
        ds = Dataset.from_pandas(subset_df)

        # Define Dataloader
        dataloader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=lambda batch: collate_fn(batch, tokenizer))
        offset = 0 # consider batches < args.batch_size

        for batch in tqdm(
            dataloader,
            total=len(dataloader),
            colour="GREEN",
            desc=f"Seq len: {seq_len}"
        ):

            # Calculate model's output
            tokens = batch['input_ids'].to(device)
            out = model.scale_context(tokens, use_forward=True) # (B, seq_len)

            # Extract true signals
            true = batch['signal'].to(device)

            # Extract peak starts and ends
            peak_starts = (batch['peak_starts'] - batch["starts"]).to(device)
            peak_ends = (batch['peak_ends'] - batch["starts"]).to(device)

            # Mask only the peak region
            B, L = out.shape

            mask = torch.zeros((B, L), dtype=torch.bool, device=out.device)
            for k in range(B):
                mask[k, peak_starts[k]:peak_ends[k]] = 1

            diff = (out - true) ** 2
            masked_diff = diff * mask.float()
            mse_per_seq = masked_diff.sum(dim=1) / mask.sum(dim=1) # (B)

            # Store results        
            batch_size = mse_per_seq.shape[0]
            mse_results[offset:offset+batch_size, i] = mse_per_seq
            offset += batch_size

            # Get minimum MSE in batch
            min_MSE_idx = mse_per_seq.argmin().item()

            # Rerun the model on short context prediction
            short_context_prediction = model.scale_context(tokens[min_MSE_idx, -model.cfg.context_len:].unsqueeze(0))

            # Get all the relevant infos
            start, end = batch['starts'][min_MSE_idx], batch['ends'][min_MSE_idx]
            chrom = batch['chroms'][min_MSE_idx]
            out_name = f"{chrom}:{start}-{end}"

            # Save plot
            save_plot(
                out[min_MSE_idx, :].detach().cpu(),
                true[min_MSE_idx, :].detach().cpu(),
                short_context_prediction.detach().cpu(),
                out_name,
                args.image_path
            )

    # Save dataframe and print summary results
    df_mse = pd.DataFrame(
        mse_results.cpu().numpy(),
        index=all_idxs,
        columns = [str(sl) for sl in seq_lengths]
    )
    df_mse.to_csv(args.save_path, sep="\t", index=True, header=True)
    print(f"MSE dataframe saved to {args.save_path}")

    # Print summary table
    print_summary_table(seq_lengths, mse_results.cpu().numpy())


if __name__ == "__main__":
    main()
