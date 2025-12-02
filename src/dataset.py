"""Basic implementation for creating a dataset from DNAse seq peaks"""

from pyfaidx import Fasta
import os
import argparse
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
from multiprocessing import Manager, RLock
from concurrent.futures import ProcessPoolExecutor, as_completed

tqdm.set_lock(RLock())

def process_chromosome(chrom, bw_f, bb_f, cfg, is_positive, progress_queue, idx):
    """Worker function to process positives or negatives for a single chromosome."""
    fasta = pyfaidx.Fasta(cfg.fasta_path)
    bw = pyBigWig.open(bw_f)
    bb = pyBigWig.open(bb_f)
    dataset = []

    bw_chroms = bw.chroms()
    if chrom not in bw_chroms:
        print(f"Warning: {chrom} not found in BigWig file {bw_f}")
        return []

    outer_coords = bb.chroms()[chrom]
    peaks = bb.entries(chrom, 0, outer_coords, withString=False)
    peaks = sorted(peaks, key=lambda x: x[0])
    peaks = peaks[:cfg.num_peaks_per_chrom] if cfg.num_peaks_per_chrom > 0 else peaks

    peak_intervals = [(start, end) for (start, end) in peaks]
    peak_starts = [start for (start, end) in peak_intervals]

    def overlaps_peak(start, end):
        idx = bisect_left(peak_starts, start)
        for i in [idx - 1, idx, idx + 1]:
            if 0 <= i < len(peak_intervals):
                p_start, p_end = peak_intervals[i]
                if not (end <= p_start or start >= p_end):
                    return True
        return False

    def get_seq_and_signal(chrom, start):
        end = start + cfg.pseudo_context
        seq = str(fasta[chrom][start:end])
        signal = bw.stats(chrom, start, end, nBins=cfg.pseudo_context // cfg.bin_size)
        
        # Handle None cases
        signal = [0.0 if s is None else s for s in signal]

        return seq.upper(), signal
    
    # Define the progress queue
    total = len(peaks) if is_positive else len(peaks) * cfg.negative_fold
    name = f"{os.path.splitext(os.path.basename(bb_f))[0]}_{chrom}:{'pos' if is_positive else 'neg'}"
    progress_queue.put(("init", name, total))

    if is_positive:
        for (start, end) in peaks:
            center = (end + start) // 2
            cand_new_start = center - cfg.pseudo_context // 2
            new_start = cand_new_start + random.randint(-cfg.pseudo_context // 2, cfg.pseudo_context // 2)
            if new_start < 0 or new_start + cfg.pseudo_context > outer_coords:
                continue
            seq, sig = get_seq_and_signal(chrom, new_start)
            label = np.zeros((cfg.pseudo_context,))
            label_idxs = np.arange(start=new_start, stop=new_start + cfg.pseudo_context)
            label = np.where((label_idxs >= start) & (label_idxs < end), 1, 0).tolist()

            if 'N' not in seq:
                dataset.append({
                    "sequence": seq,
                    "signal": sig,
                    "chrom": chrom,
                    "start": new_start,
                    "end": new_start + cfg.pseudo_context,
                    "peak_start": start,
                    "peak_end": end,
                    "label": label,
                    "is_positive": True,
                    "idx": idx
                })
            progress_queue.put(("update", name, 1))
    else:
        # Negatives
        n_negatives = len(peaks) * cfg.negative_fold
        neg_sampled = 0
        attempts = 0
        max_attempts = n_negatives * 10
        while neg_sampled < n_negatives and attempts < max_attempts:
            start = random.randint(0, outer_coords - cfg.pseudo_context)
            end = start + cfg.pseudo_context
            if not overlaps_peak(start, end):
                seq, sig = get_seq_and_signal(chrom, start)
                if 'N' not in seq:
                    label = np.zeros((cfg.pseudo_context,)).tolist()
                    dataset.append({
                        "sequence": seq,
                        "signal": sig,
                        "chrom": chrom,
                        "start": start,
                        "end": end,
                        "peak_start": -1,
                        "peak_end": -1,
                        "label": label,
                        "is_positive": False,
                        "idx": idx
                    })
                    neg_sampled += 1
                    progress_queue.put(("update", name, 1))
            attempts += 1

    fasta.close()
    bw.close()
    bb.close()
    progress_queue.put(("done", name, None))
    return dataset

def create_dataset_parallel(cfg, bws, bbs, chroms):
    manager = Manager()
    progress_queue = manager.Queue()
    all_datasets = []

    bars = {}
    from threading import Thread

    def progress_updater():
        while True:
            try:
                msg_type, name, val = progress_queue.get(timeout=1)
            except:
                continue

            if msg_type == "init":
                bars[name] = tqdm(total=val, desc=name, position=len(bars), leave=True, colour="green")
            
            elif msg_type == "update":
                if name in bars:
                    bars[name].update(val)

            elif msg_type == "stop":
                break

    thread = Thread(target=progress_updater, daemon=True)
    thread.start()

    # Now start the parallel execution
    with ProcessPoolExecutor(max_workers=min(len(chroms) * 2, 10)) as executor:
        # Cycle over DNase-seq Datasets
        futures = []
        for idx, (bw_f, bb_f) in enumerate(zip(bws, bbs)): 
            for chrom in chroms:
                futures.append(executor.submit(process_chromosome, chrom, bw_f, bb_f, cfg, True, progress_queue, idx))
                futures.append(executor.submit(process_chromosome, chrom, bw_f, bb_f, cfg, False, progress_queue, idx))
            for f in as_completed(futures):
                all_datasets.extend(f.result())

    # Stop Master Thread
    progress_queue.put(("stop", None, None))
    thread.join(timeout=2)
    return all_datasets

def chunk_list_truncate(lst, real_context_length):
    assert real_context_length > 0
    assert (len(lst) % real_context_length) == 0, "Real context length does not divide pseudo context length."
    n = (len(lst) // real_context_length) * real_context_length
    return [lst[i:i+real_context_length] for i in range(0, n, real_context_length)]

CHROMS = [f"chr{i}" for i in range(1, 6)] # + ["chrX", "chrY"]

class DatasetCreator:
    def __init__(self, cfg: DatasetCfg):
        self.cfg = cfg
        self.fasta = pyfaidx.Fasta(cfg.fasta_path)
        self.bws = cfg.bigwig_list
        self.bbs = cfg.bigbed_list
        self.dataset = []
            
    def _save_dataset(self):
        print(f"Saving dataset to {self.cfg.save_path}...")
        # with open(self.cfg.save_path, "w") as out:
        #     json.dump(self.dataset, out)
        hf_dataset = Dataset.from_list(self.dataset)
        hf_dataset.save_to_disk(self.cfg.save_path)

    def _save_metadata(self):
        print(f"Saving metadatata to {self.cfg.save_path}...")
        
        with open(f"{self.cfg.save_path}/index.csv", "w") as out:
            for i, exp in enumerate(self.bws):
                exp_name = os.path.splitext(os.path.basename(exp))[0]
                out.write(f"{i},{exp_name}\n")
    
    def create_dataset(self):
        print(f"Processing chromosomes in parallel using {len(CHROMS)} workers...")
        dataset = create_dataset_parallel(self.cfg, self.bws, self.bbs, CHROMS)
        self.dataset = dataset
        self._save_dataset()
        self._save_metadata()


def get_hf_dataset(dataset_path: str) -> Dataset:
    """Load json file and returns hf dataset"""
    df = pd.read_json(dataset_path)
    hf_df = Dataset.from_pandas(df)
    return hf_df

class DNAseSeqDataset(Dataset):
    """Wrapper of torch.Dataset for training. To use after tokenization."""
    
    def __init__(self, tokenized_dataset):
        self.input_ids = tokenized_dataset['input_ids']
        self.attention_mask = tokenized_dataset['attention_mask']
        self.signal = tokenized_dataset['signal']
        self.labels = tokenized_dataset['label']
        self.idxs = tokenized_dataset['idx']
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):

        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "signal": self.signal[idx],
            "label": self.labels[idx],
            'idxs': self.idxs[idx]
        }
    
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bigwig_list", type=str, required=True, help="List of comma separated bigwig identifiers.")
    p.add_argument("--bigbed_list", type=str, required=True, help="List of comma separated bigbed files.")
    p.add_argument("--bb_folder", type=str, required=True, help="Where to find bigbeds")
    p.add_argument("--bw_folder", type=str, required=True, help="Where to find bigwigs")

    args = p.parse_args()

    bigbed_list = [os.path.join(args.bb_folder, bb) for bb in args.bigbed_list.split(",")]
    bigwig_list = [os.path.join(args.bw_folder, bw) for bw in args.bigwig_list.split(",")]

    dataset_cfg = DatasetCfg(
        bigwig_list=bigwig_list,
        bigbed_list=bigbed_list
    )

    dataset = DatasetCreator(dataset_cfg)
    dataset.create_dataset()
    
if __name__ == '__main__':
    main()