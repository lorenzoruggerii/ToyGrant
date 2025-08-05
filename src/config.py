from dataclasses import dataclass

@dataclass
class DatasetCfg:
    """
    Config class for dataset creation

    Args:
     - pseudo_context: pseudo-context length for Mamboros (eg. 4K)
     - real_context: real context length for Mamboros (eg. 1K)
     - num_peaks_per_chrom: number of peaks extracted per chromosome
     - negative_fold: ratio between number of negatives and the total number of sequences
     - fasta_path: path to reference genome
     - bigwig_path: path to bigwig file
     - bigbed_path: path to bigbed file
     - save_path: where to save dataset
     - bin_size: how many NTs to use to average the DNAse signal values? 
    """
    pseudo_context: int = 4_000
    real_context: int = 1_000
    num_peaks_per_chrom: int = 20
    negative_fold: int = 1 # balanced dataset
    fasta_path: str = "../Mamba_TFBS/genome/hg38.fa"
    bigwig_path: str = "files/bigWig/ENCFF972GVB.bigWig"
    bigbed_path: str = "files/bed/ENCFF070TML.bigBed"
    save_path: str = "datasets/tracks_full_test.json"
    bin_size: int = 1 # consider 1 value for each nucleotide
