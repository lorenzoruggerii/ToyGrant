import torch
import argparse
import matplotlib.pyplot as plt
from config import TrackMambaConfig
from TrackMamba import TrackMamba
from pyfaidx import Fasta
import pyBigWig
import pandas as pd
from typing import List
from tokenizer import CharacterTokenizer
from torch.nn.functional import sigmoid
import numpy as np
import os
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def extract_sig_and_seq(fasta_path: str, bigwig_path: str, input_coords: str, max_length: int):
    """Extract signal and sequences from input cooordinates"""
    
    # Open all the inputs
    ref_genome = Fasta(fasta_path)
    bw = pyBigWig.open(bigwig_path)
    coords = pd.read_csv(input_coords, sep=r"\s+", header=None) # chr | start | stop

    # Assert everything is inside context length
    assert ((coords[2] - coords[1]) <= max_length).all(), f"Every sequence must be < {max_length} nucleotides."

    seqs = []
    sigs = []

    # Extract sequences and signals
    for idx, (chr, start, stop) in tqdm(coords.iterrows(), desc="Extracting seqs...", total=len(coords)):
        sequence = str(ref_genome[chr][start:stop]).upper()
        signal = bw.stats(chr, start, stop, nBins=len(sequence))

        # Update lists
        seqs.append(sequence)
        sigs.append(signal)

    return seqs, sigs
    
def get_predictions_from_model(sequences: List[str],  model_path: str, max_length: int):
    """Run the model on extracted sequences"""

    # Initilize model
    cfg = TrackMambaConfig()
    m = TrackMamba.from_pretrained(model_path, cfg).to(device)

    # Initialize tokenizer
    tok = CharacterTokenizer(
        characters=['A', 'C', 'G', 'T', 'N'],
        model_max_length=max_length
    )

    # Tokenize sequences
    input_ids = tok(sequences, add_special_tokens=False, return_tensors='pt', padding=True)['input_ids'].to(device)

    # Run the model
    pred_regrs, class_preds = m(input_ids)

    # Convert class preds into binary
    pred_labels = (sigmoid(class_preds) > 0.5).int()

    return pred_regrs.cpu().detach(), pred_labels.cpu().detach()

def save_plot(regr_preds: torch.Tensor, class_preds: torch.Tensor, signals: List[List[float]], save_path: str, coords: str, ncols=5):
    """Create plots and save them."""

    # Load coords information
    coords = pd.read_csv(coords, sep=r"\s+", header=None) # chr | start | stop
    
    # Calculate lengths
    seq_lenghts = (coords[2] - coords[1]).tolist()

    # Calculate number of rows and columns
    nseqs = len(signals)
    nrows = int(np.ceil(nseqs / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows), squeeze=False)

    for i in range(nseqs):
        ax = axes[i // ncols, i % ncols]

        x = np.arange(len(signals[i]))
        y_true = signals[i]
        y_pred = regr_preds[i][:seq_lenghts[i]]
        cls_pred = class_preds[i][:seq_lenghts[i]]

        # plot regression
        ax.plot(x, y_true, label="True", color="blue", linewidth=2)
        ax.plot(x, y_pred, label="Pred", color="red", linestyle="--", linewidth=2)

        # highlight classification=1
        for j in np.where(cls_pred == 1)[0]:
            ax.axvspan(j - 0.5, j + 0.5, color="yellow", alpha=0.3)

        ax.set_title(f"{coords[0][i]}: {coords[1][i]}-{coords[2][i]}")
        ax.set_xlabel("Position")
        ax.set_ylabel("DNAse signal")
        if i == 0:  # only show legend once
            ax.legend()

    # remove unused subplots if nseqs < nrows*ncols
    for k in range(nseqs, nrows * ncols):
        fig.delaxes(axes[k // ncols, k % ncols])

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'plot_coords.png'), dpi=300, format='png')



def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True, help="Path to the trained TrackMamba model.")
    p.add_argument("--input_coords", type=str, required=True, help="Path to the input coordinates file (tsv). The file must be chr | start | stop.")
    p.add_argument("--bigwig", type=str, required=True, help="Path to bigWig file.")
    p.add_argument("--fasta", type=str, required=True, help="Path to reference genome.")
    p.add_argument("--max_length", type=int, required=True, help="TrackMamba context length.")
    p.add_argument("--save_path", type=str, required=False, help="Where to put images.", default=".")

    args = p.parse_args()

    # Open coords and extract sequences and signal
    sequences, signals = extract_sig_and_seq(args.fasta, args.bigwig, args.input_coords, args.max_length)
    
    # Run the model and obtain predictions
    regr_preds, class_preds = get_predictions_from_model(sequences, args.model_path, args.max_length)

    # Plot everything
    save_plot(regr_preds, class_preds, signals, args.save_path, args.input_coords)

if __name__ == '__main__':
    main()