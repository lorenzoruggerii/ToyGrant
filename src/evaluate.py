import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import json
import torch
import argparse
import matplotlib.pyplot as plt
from config import TrackMambaConfig
from TrackMamba import TrackMamba
from pyfaidx import Fasta
import pyBigWig
import pandas as pd
from typing import List, Tuple
from tokenizer import CharacterTokenizer
from torch.nn.functional import sigmoid
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from alphagenome.models import dna_client
from alphagenome.data import genome
from scipy.stats import pearsonr, spearmanr

device = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 64

def extract_sig_and_seq(
        fasta_path: str,
        bigwig_path: str,
        bigbed_file: str, 
        chrom: str, 
        max_length: int,
        max_sequences: int = None,
        is_custom: bool = False
    ):
    """Extract signal and sequences from input cooordinates"""
    
    # Open all the inputs
    ref_genome = Fasta(fasta_path)
    bw = pyBigWig.open(bigwig_path)

    # coords = pd.read_csv(input_coords, sep="\t", header=None) # chr | start | stop
    if not is_custom:
        bb = pyBigWig.open(bigbed_file)
        outer_coords = bb.chroms()[chrom]
        original_coords = bb.entries(chrom, 0, outer_coords, withString=False)
    else:
        with open(bigbed_file, "r") as fopen:  
            bb = json.load(fopen)
            original_coords = bb[chrom]

    # Filter if max sequences is provided
    original_coords = original_coords[:max_sequences] if max_sequences is not None else original_coords

    # Transform to pandas dataframe
    coords = pd.DataFrame(original_coords, columns=["start", "end"])

    # Assert everything is inside context length
    assert ((coords["end"] - coords["start"]) <= max_length).all(), f"Every sequence must be < {max_length} nucleotides."

    # Apply padding and shuffle the peak in the middle
    peak_summits = (coords["start"] + coords["end"]) // 2
    summits_offsets = np.random.randint(-max_length / 2, max_length / 2, size=len(coords))
    new_peak_summits = peak_summits + summits_offsets
    coords["start"] = new_peak_summits - max_length // 2
    coords["end"] = new_peak_summits + max_length // 2

    # Add chrom column to dataframe
    coords.insert(0, "chrom", chrom)

    seqs = []
    sigs = []

    # Extract sequences and signals
    for idx, (chr, start, stop) in tqdm(coords.iterrows(), desc="Extracting seqs...", total=len(coords)):
        sequence = str(ref_genome[chr][start:stop]).upper()
        signal = bw.stats(chr, start, stop, nBins=len(sequence))

        # Handle None cases
        signal = [0.0 if s is None else s for s in signal]

        # Update lists
        seqs.append(sequence)
        sigs.append(signal)

    return seqs, sigs, coords, original_coords
    
def get_predictions_from_model(
        sequences: List[str],  
        model_path: str, 
        max_length: int, 
        num_layers: int, 
        hidden_dim: int, 
        use_MLP: bool, 
        use_pos_embs: bool, 
        metadata: str,
        bigwig: str,
        use_conv: bool,
        use_bidirectionality: bool
    ):
    """Run the model on extracted sequences"""

    print("Making predictions...")

    # Initilize model
    cfg = TrackMambaConfig(
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        use_MLP=use_MLP,
        use_pos_embs=use_pos_embs,
        head_index=metadata,
        use_conv=use_conv,
        use_bidirectionality=use_bidirectionality
    )

    m = TrackMamba.from_pretrained(model_path, cfg).to(device)

    # Initialize tokenizer
    tok = CharacterTokenizer(
        characters=['A', 'C', 'G', 'T', 'N'],
        model_max_length=max_length
    )

    # Tokenize sequences
    input_ids = tok(sequences, add_special_tokens=False, return_tensors='pt', padding=True)['input_ids']

    # Define tensors for holding predictions
    regression_preds = torch.zeros((len(sequences), max_length))
    classification_preds = torch.zeros((len(sequences), max_length))

    # Run the model
    bs = 0
    head_idx = m.metadata.get_exp_index(os.path.splitext(os.path.basename(bigwig))[0])

    # Progress bar
    progress_bar = tqdm(total=len(sequences) // BATCH_SIZE, desc="TrackMamba is predicting...")

    with torch.no_grad():
        while(bs < len(sequences)):
            # Get current batch
            input_ids_batch = input_ids[bs:bs+BATCH_SIZE].to(device)
            # Make predictions
            pred_regs, class_preds = m(input_ids_batch, attention_mask=None)

            # Extract predictions from right head
            current_batch_size = pred_regs.shape[0]
            batches = torch.arange(0, current_batch_size, device=pred_regs.device)
            pred_regs = pred_regs[batches, :, head_idx]
            class_preds = class_preds[batches, :, head_idx]

            # Convert class preds into binary
            pred_labels = (sigmoid(class_preds) > 0.5).int()

            # Add to the returned tensors
            regression_preds[bs:bs+BATCH_SIZE] = pred_regs.cpu()
            classification_preds[bs:bs+BATCH_SIZE] = pred_labels.cpu()

            # Update bs
            bs += current_batch_size
            progress_bar.update()

    print("Predictions done!")

    return regression_preds.cpu().detach(), classification_preds.cpu().detach()

def get_predictions_ag(coords: pd.DataFrame, GO_term: str, max_length: int, chrom: str):
    "Run alphagenome on coords and get DNase track"

    # Load API KEY
    load_dotenv()
    API_KEY = os.getenv("API_KEY")

    # Load AlphaGenome model
    dna_model = dna_client.create(API_KEY)

    # Load coords information
    coords = pd.DataFrame(coords, columns=["start", "end"]) # chr | start | stop
    coords.insert(0, "chr", chrom)

    # Initialize output matrix
    out_AG = np.zeros((len(coords), max_length))

    # Calculate predictions
    for i in tqdm(range(len(coords)), total=len(coords), desc="Making AG predictions..."):
        
        # Define the interval
        interval = genome.Interval(chromosome=coords.at[i, "chr"], start = coords.at[i, "start"], end = coords.at[i, "end"])

        # Resize to desired input len
        interval = interval.resize(dna_client.SEQUENCE_LENGTH_16KB)

        # Make prediction
        outputs = dna_model.predict_interval(
            interval=interval,
            requested_outputs=[dna_client.OutputType.CHIP_TF],
            ontology_terms=[GO_term] # term of the cell line
        )

        outputs = outputs.chip_tf.values

        # Extract input range and add to out vector
        out = outputs[coords.at[i, "start"] - interval.start:coords.at[i, "end"] - interval.start,:]

        # Pad to desired length
        out = np.pad(out.squeeze(), (0, max_length - len(out)), mode="constant")
        out_AG[i, :] = out

    return out_AG

def save_plot(
        regr_preds: torch.Tensor, 
        class_preds: torch.Tensor, 
        signals: List[List[float]], 
        ag_preds: np.array, 
        save_path: str, 
        orig_coords: List[Tuple[int, int]],
        coords: pd.DataFrame,
        model_name: str, 
        chrom: str,
        head_name: str,
        ncols: int = 5
    ):
    """Create plots and save them."""
    
    # Calculate lengths
    # seq_lenghts = (coords["end"] - coords["start"]).tolist()

    # Calculate number of rows and columns
    nseqs = len(signals)
    nrows = int(np.ceil(nseqs / ncols))

    # Compute RMSEs
    pearsons = []
    spearmans = []

    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows), squeeze=False)

    for i in tqdm(range(nseqs), total=nseqs, desc='Plotting tracks...'):
        ax = axes[i // ncols, i % ncols]

        x = np.arange(len(signals[i]))
        y_true = signals[i]
        y_pred = regr_preds[i].squeeze()#[:seq_lenghts[i]]
        cls_pred = class_preds[i].squeeze()#[:seq_lenghts[i]]
        # ag_pred = ag_preds[i]#[:seq_lenghts[i]] we padded till model_length #FIXME

        # Compute pearson and spearman
        pearsons.append(pearsonr(y_true, y_pred)[0])
        spearmans.append(spearmanr(y_true, y_pred)[0])
        
        # plot regression
        ax.plot(x, y_true, label="True", color="blue", linewidth=2)
        ax.plot(x, y_pred, label="Pred", color="red", linestyle="--", linewidth=2)
        # ax.plot(x, ag_pred.squeeze(), label="AG", color="green", linestyle="--", linewidth=2, alpha=0.3) #FIXME

        # highlight classification=1
        # for j in np.where(cls_pred == 1)[0]:
        #     ax.axvspan(j - 0.5, j + 0.5, color="yellow", alpha=0.3)

        # Plot vline for peak identification
        ax.axvline(x=orig_coords[i][0] - coords.at[i, "start"], color="black", linewidth=2)
        ax.axvline(x=orig_coords[i][1] - coords.at[i, "start"], color="black", linewidth=2)

        ax.set_title(f"{chrom}:{orig_coords[i][0]}-{orig_coords[i][1]}")
        ax.set_xlabel("Position")
        ax.set_ylabel("DNAse signal")
        if i == 0:  # only show legend once
            ax.legend()

    # remove unused subplots if nseqs < nrows*ncols
    for k in range(nseqs, nrows * ncols):
        fig.delaxes(axes[k // ncols, k % ncols])

    # Add title for model
    plot_title = f"{os.path.splitext(os.path.basename(model_name))[0]}_{chrom}_{head_name}"

    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f'{plot_title}.png'), dpi=300, format='png')
    
    # Clear current axes
    plt.show()
    plt.close()
    plt.cla()

    with open(os.path.join(save_path, f'{plot_title}_MSEs.tsv'), "w") as fout:
        for pears, spear in zip(pearsons, spearmans):
            fout.write(f"{pears}\t{spear}\n")


def main():
    p = argparse.ArgumentParser(description="Plot predicted TrackMamba tracks.")
    p.add_argument("--model_path", type=str, required=True, help="Path to the trained TrackMamba model.")
    # p.add_argument("--input_coords", type=str, required=True, help="Path to the input coordinates file (tsv). The file must be chr | start | stop.")
    p.add_argument("--bigbed", type=str, required=True, help="Bigbed file from which we make predictions.")
    p.add_argument("--chrom", type=str, required=True, help="Chromosome from which peaks are extracted.")
    p.add_argument("--metadata", type=str, required=True, help="Path to the metadata file for TrackMamba.")
    p.add_argument("--bigwig", type=str, required=True, help="Path to bigWig file.")
    p.add_argument("--fasta", type=str, required=True, help="Path to reference genome.")
    p.add_argument("--max_length", type=int, required=True, help="TrackMamba context length.")
    p.add_argument("--save_path", type=str, required=False, help="Where to put images.", default=".")
    p.add_argument("--num_layers", type=int, required=True, help="Number of TrackMamba layers.")
    p.add_argument("--use_MLP", action='store_true', default=False)
    p.add_argument("--use_pos_embs", action='store_true', default=False)
    p.add_argument("--use_conv", action='store_true', default=False)
    p.add_argument("--use_bidirectionality", action='store_true', default=False)
    p.add_argument("--hidden_dim", type=int, required=True, help="Dimensionality of TrackMamba residual stream.")
    p.add_argument("--GO_term", type=str, required=True, help="GO Term associated with the input experiment.")
    p.add_argument("--max_sequences", type=int, required=False, default=None, help="How many sequences from chrom to process.")
    p.add_argument("--is_custom", action="store_true", default=False)

    args = p.parse_args()

    # Open coords and extract sequences and signal
    sequences, signals, coords, original_coords = extract_sig_and_seq(args.fasta, args.bigwig, args.bigbed, args.chrom, args.max_length, args.max_sequences, args.is_custom)
    
    # Run the model and obtain predictions
    regr_preds, class_preds = get_predictions_from_model(
        sequences, 
        args.model_path, 
        args.max_length, 
        args.num_layers, 
        args.hidden_dim, 
        args.use_MLP, 
        args.use_pos_embs, 
        args.metadata, 
        args.bigwig,
        args.use_conv,
        args.use_bidirectionality
    )

    # Get predictions from alphagenome
    # ag_preds = get_predictions_ag(coords, args.GO_term, args.max_length, args.chrom)

    # Plot everything
    save_plot(
        regr_preds, 
        class_preds, 
        signals, 
        None, 
        args.save_path, 
        original_coords,
        coords, 
        args.model_path, 
        args.chrom,
        os.path.splitext(os.path.basename(args.bigwig))[0]
    )

if __name__ == '__main__':
    main()