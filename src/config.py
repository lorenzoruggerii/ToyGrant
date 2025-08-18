import torch
from dataclasses import dataclass, field
from tokenizer import CharacterTokenizer
from typing import List, Callable, Tuple
from torch.nn import MSELoss, BCEWithLogitsLoss


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
     - is_mamboros_dataset: whether to return a mamboros dataset (signal, labels and sequence are a list of lists)
    """
    pseudo_context: int = 4_000
    real_context: int = 1_000
    num_peaks_per_chrom: int = -1
    negative_fold: int = 1 # balanced dataset
    fasta_path: str = "../Mamba_TFBS/genome/hg38.fa"
    bigwig_path: str = "files/bigWig/ENCFF972GVB.bigWig"
    bigbed_path: str = "files/bed/ENCFF070TML.bigBed"
    save_path: str = "datasets/tracks_1_22_4K.json"
    bin_size: int = 1 # consider 1 value for each nucleotide
    is_mamboros_dataset: bool = False 
    

@dataclass
class TrackMambaConfig:
    """
    Config class for TrackMamba.

    Args:
        - context_len: input sequence length
        - tokenizer: tokenizer used by the model
        - vocab_size: tokenizer vocabulary size
        - hidden_dim: residual stream dimensionality
        - num_layers: model's input layers
        
    """
    context_len: int = 4_000
    tokenizer: CharacterTokenizer = CharacterTokenizer(
        ['A', 'C', 'G', 'T', 'N'],
        model_max_length=context_len
    )
    num_layers: int = 4
    vocab_size: int = tokenizer.vocab_size
    hidden_dim: int = 256

@dataclass
class TrackMambaTrainerCfg:
    """
    Config class for trainer

    Args: 
        - lr: learning rate used for training
        - batch_size: batch size used for training
        - regression_loss: loss function used for regression task
        - classification_loss: loss function used for classification task
        - optimizer: optimizer used for training
        - data_path: path to dataset's dir 
        - data_file: dataset's filename
        - wandb_project: wandb project name
        - wandb_name: wandb run name
        - num_epochs: number of epochs used for training
        - train_chroms: chromosomes used for training
        - test_chroms: chromosomes used for testing
        - comb_factor: relative weight for classification loss
        - save_path: where to save trained model
        - seed: for reproducibility
    """
    lr: float = 3e-5
    batch_size: int = 4
    regression_loss: Callable = MSELoss()
    classification_loss: Callable = BCEWithLogitsLoss()
    optimizer: torch.optim.Optimizer = torch.optim.AdamW
    data_path: str = "datasets"
    data_file: str = "tracks_1_22_4K.json"
    wandb_project: str = "ToyGrant"
    num_epochs: int = 10
    train_chroms: List[str] = field(default_factory=lambda: ["chr1"])
    test_chroms: List[str] = field(default_factory=lambda: ["chr22"])
    comb_factor: float = 1.0
    wandb_name: str = f"TrackMambaNorm_{batch_size}BS_{num_epochs}Epochs_{comb_factor}lambda"
    save_path: str = f"models/ToyGrant/TrackMamba_{batch_size}BS_{num_epochs}Epochs_{comb_factor}lambda.pt"
    seed: int = 42