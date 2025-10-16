from tqdm import tqdm
from dataset import DatasetCreator
from config import DatasetCfg
import random
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

# Required for parallelization
# concurrent can't pickle file handles, like self.fasta ...
# So we need to reopen them inside the function
def process_peak(chrom, start, end, cfg_dict, num_chunks, new_context_len, mean_noise):
    from pyfaidx import Fasta
    import pyBigWig
    import random

    fasta = Fasta(cfg_dict["fasta_path"])
    bw = pyBigWig.open(cfg_dict["bw_path"])
    bb = pyBigWig.open(cfg_dict["bb_path"])

    results = []
    peak_center = (end + start) // 2

    step_size = new_context_len // num_chunks
    new_lengths = [x for x in range(step_size, new_context_len + step_size, step_size) if x > cfg_dict["pseudo_context"]]
    new_lengths = [0, cfg_dict["pseudo_context"]] + new_lengths

    new_centers = []
    for s, e in zip(new_lengths[:-1], new_lengths[1:]):
        if (e - s) <= cfg_dict["pseudo_context"]:
            new_centers.append(
                random.randint(s, e)
            )
        else:
            new_centers.append(
                random.randint(e - cfg_dict["pseudo_context"], e)
            )

    for seq_len, center in zip(new_lengths[1:], new_centers):
        new_start = peak_center - center
        new_end = new_start + seq_len
        if new_start < 0 or new_end > cfg_dict["chrom_sizes"][chrom]:
            continue
        seq = str(fasta[chrom][new_start:new_end]).upper()
        sig = bw.stats(chrom, new_start, new_end, nBins=seq_len // cfg_dict["bin_size"])
        if "N" not in seq:
            results.append({
                "sequence": seq,
                "signal": sig,
                "chrom": chrom,
                "start": new_start,
                "end": new_end,
                "center": center,
                "peak_start": start,
                "peak_end": end,
                "idx": f"{chrom}:{new_start}-{new_end}"
            })
    return results

class ScalingDatasetCreator(DatasetCreator):

    def __init__(self, num_chunks: int, new_context_len: int, mean_noise: float, cfg: DatasetCfg):
        super().__init__(cfg)

        assert (new_context_len % num_chunks == 0), f"{new_context_len} is not divided evenly by {num_chunks}."

        self.num_chunks = num_chunks
        self.new_context_len = new_context_len
        self.mean_noise = mean_noise
    
    def get_seq_and_signal(self, chrom, start, length):
        end = start + length
        seq = str(self.fasta[chrom][start:end])
        signal = self.bw.stats(chrom, start, end, nBins = length // self.cfg.bin_size)
        return seq.upper(), signal
     
    def create_dataset(self, num_workers: int):
        
        outer_coords = self.bb.chroms()

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            cfg_dict = {
                "fasta_path": self.cfg.fasta_path,
                "bw_path": self.cfg.bigwig_path,
                "bb_path": self.cfg.bigbed_path,
                "pseudo_context": self.cfg.pseudo_context,
                "bin_size": self.cfg.bin_size,
                "chrom_sizes": self.chroms,
            }

            for chrom in self.chroms:

                peaks = self.bb.entries(chrom, 0, outer_coords[chrom], withString=False)
                peaks = sorted(peaks, key = lambda x: x[0]) # sort peaks by start

                # Filter on the number of peaks per chromosome
                peaks = peaks[:self.cfg.num_peaks_per_chrom] if self.cfg.num_peaks_per_chrom > 0 else peaks

                progress_bar = tqdm(total=len(peaks)*self.num_chunks, colour="GREEN")
                progress_bar.set_description(f"{chrom}")

                # Retrieve positive sequences
                for (start, end) in peaks:
                    futures.append(executor.submit(
                        process_peak, chrom, start, end, cfg_dict,
                        self.num_chunks, self.new_context_len, self.mean_noise
                    ))

                progress_bar = tqdm(total=len(futures), colour="GREEN")

                for future in as_completed(futures):
                    results = future.result()
                    self.dataset.extend(results)
                    progress_bar.update()
                
        # Save final dataset
        self._save_dataset()

if __name__ == "__main__":

    p = argparse.ArgumentParser()
    p.add_argument("--save_path", type=str, required=True, help="Where to save dataset.")
    p.add_argument("--new_context_len", type=int, required=True, help="New context len to scale")
    p.add_argument("--num_chunks", type=int, required=True, help="Number of chunks to evenly divide context_len.")
    p.add_argument("--mean_noise", type=int, required=False, default=100, help="Mean noise to add to the center")
    p.add_argument("--num_workers", type=int, required=True, help="Maximum number of concurrent threads.")

    args = p.parse_args()
    
    cfg = DatasetCfg(save_path=args.save_path)
    ds = ScalingDatasetCreator(num_chunks=args.num_chunks, new_context_len=args.new_context_len, mean_noise=args.mean_noise, cfg=cfg)

    ds.create_dataset(args.num_workers)

