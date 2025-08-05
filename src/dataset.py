"""Basic implementation for creating a dataset from DNAse seq peaks"""

from pyfaidx import Fasta
import pandas as pd
import pyBigWig
from config import DatasetCfg
import pyfaidx
import json
from tqdm import tqdm
import random
from torch.utils.data import Dataset
from datasets import Dataset
from bisect import bisect_left
import numpy as np

def chunk_list_truncate(lst, real_context_length):
    assert real_context_length > 0
    assert (len(lst) % real_context_length) == 0, "Real context length does not divide pseudo context length."
    n = (len(lst) // real_context_length) * real_context_length
    return [lst[i:i+real_context_length] for i in range(0, n, real_context_length)]

CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"] 

class DatasetCreator:
    def __init__(self, cfg: DatasetCfg):
        self.cfg = cfg
        self.fasta = pyfaidx.Fasta(cfg.fasta_path)
        self.bw = pyBigWig.open(cfg.bigwig_path)
        self.bb = pyBigWig.open(cfg.bigbed_path)
        self.chroms = self.bw.chroms()
        self.chroms = {k: v for k, v in self.chroms.items() if k in CHROMS}
        self.dataset = []

    def get_seq_and_signal(self, chrom, start):
        end = start + self.cfg.pseudo_context
        seq = str(self.fasta[chrom][start:end])
        signal = self.bw.stats(chrom, start, end, nBins = self.cfg.pseudo_context // self.cfg.bin_size)
        return seq.upper(), signal
            
    def _save_dataset(self):
        print(f"Saving dataset to {self.cfg.save_path}...")
        with open(self.cfg.save_path, "w") as out:
            json.dump(self.dataset, out)

    def create_dataset(self):

        # Pre-calculate outermost coords
        outer_coords = self.bb.chroms()
        
        for chrom in self.chroms:
            peaks = self.bb.entries(chrom, 0, outer_coords[chrom], withString=False)
            peaks = sorted(peaks, key=lambda x: x[0])  # sort by start

            # Filter on the number of peaks per chromosome
            peaks = peaks[:self.cfg.num_peaks_per_chrom] if self.cfg.num_peaks_per_chrom > 0 else peaks

            peak_intervals = [(start, end) for (start, end) in peaks]
            peak_starts = [start for (start, end) in peak_intervals]

            progress_bar = tqdm(total=len(peaks)*(self.cfg.negative_fold + 1), colour="GREEN")
            progress_bar.set_description(f"{chrom}: pos")
            starts = []

            def overlaps_peak(start, end):
                idx = bisect_left(peak_starts, start)
                for i in [idx - 1, idx, idx + 1]:
                    if 0 <= i < len(peak_intervals):
                        p_start, p_end = peak_intervals[i]
                        if not (end <= p_start or start >= p_end):
                            return True
                return False

            # Retrieve positive sequences
            for (start, end) in peaks:
                center = (end + start) // 2
                cand_new_start = center - self.cfg.pseudo_context // 2
                new_start = cand_new_start + random.randint(-self.cfg.pseudo_context // 2, self.cfg.pseudo_context // 2)
                if new_start < 0 or new_start + self.cfg.pseudo_context > self.chroms[chrom]:
                    continue
                seq, sig = self.get_seq_and_signal(chrom, new_start)

                # Define label for classification task
                label = np.zeros((self.cfg.pseudo_context,))
                label_idxs = np.arange(start=new_start, stop=new_start + self.cfg.pseudo_context)
                label = np.where((label_idxs >= start) & (label_idxs < end), 1, 0).tolist()

                # Make sure they have the same shape
                assert len(label) == len(seq) == len(sig) == self.cfg.pseudo_context

                # Chunk to real_context size
                seq = chunk_list_truncate(seq, self.cfg.real_context)
                sig = chunk_list_truncate(sig, self.cfg.real_context)
                label = chunk_list_truncate(label, self.cfg.real_context)

                if 'N' not in seq:
                    self.dataset.append({
                        "sequence": seq,
                        "signal": sig,
                        "chrom": chrom,
                        "start": new_start,
                        "end": new_start + self.cfg.pseudo_context,
                        "peak_start": start,
                        "peak_end": end,
                        "label": label,
                        "is_positive": True
                    })
                    starts.append(new_start)
                progress_bar.update()

            # Retrieve negative sequences of the same number as positive
            n_negatives = len(peaks) * self.cfg.negative_fold
            neg_sampled = 0
            attempts = 0
            max_attempts = n_negatives * 10
            progress_bar.set_description(f"{chrom}: neg")

            while neg_sampled < n_negatives and attempts < max_attempts:
                start = random.randint(0, self.chroms[chrom] - self.cfg.pseudo_context)
                end = start + self.cfg.pseudo_context
                if not overlaps_peak(start, end):
                    seq, sig = self.get_seq_and_signal(chrom, start)

                    # Define label as all zeros
                    label = np.zeros((self.cfg.pseudo_context,)).tolist()

                    # Chunk to real_context size
                    seq = chunk_list_truncate(seq, self.cfg.real_context)
                    sig = chunk_list_truncate(sig, self.cfg.real_context)
                    label = chunk_list_truncate(label, self.cfg.real_context)

                    if 'N' not in seq:
                        self.dataset.append({
                            "sequence": seq,
                            "signal": sig,
                            "chrom": chrom,
                            "start": start,
                            "end": end,
                            "peak_start": -1,
                            "peak_end": -1,
                            "label": label,
                            "is_positive": False
                        })
                        neg_sampled += 1
                        progress_bar.update()
                attempts += 1

        # Save final dataset
        self._save_dataset()

def get_hf_dataset(dataset_path: str) -> Dataset:
    """Load json file and returns hf dataset"""
    df = pd.read_json(dataset_path)
    hf_df = Dataset.from_pandas(df)
    return hf_df

class DNAseSeqDataset(Dataset):
    """Wrapper of torch.Dataset for training. To use after tokenization."""
    
    def __init__(self, tokenized_dataset):
        self.input_ids = tokenized_dataset['input_ids']
        self.labels = tokenized_dataset['signal']
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):

        return {
            "input_ids": self.input_ids[idx],
            "labels": self.labels[idx]
        }
    
if __name__ == '__main__':
    dataset_cfg = DatasetCfg()
    dataset = DatasetCreator(dataset_cfg)
    dataset.create_dataset()

    hf_dataset = get_hf_dataset(dataset_cfg.save_path)
    print(hf_dataset)