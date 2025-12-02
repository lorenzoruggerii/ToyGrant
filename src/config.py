import torch
from dataclasses import dataclass, field, asdict
from tokenizer import CharacterTokenizer
from typing import List, Callable, Tuple
from torch.nn import MSELoss, BCEWithLogitsLoss
import pandas as pd
import json


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
    bigwig_list: List[str]
    bigbed_list: List[str]
    pseudo_context: int = 4_000
    real_context: int = 1_000
    num_peaks_per_chrom: int = 10_000 # FIXME
    negative_fold: int = 1 # balanced dataset
    fasta_path: str = "../../Mamba_TFBS/genome/hg38.fa"
    save_path: str = "datasets/CHROMS_1_5_GATA1_CTCF_10_000"
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
        - use_MLP: whether to use MLP block after Mamba
        
    """
    context_len: int = 4_000
    tokenizer: CharacterTokenizer = CharacterTokenizer(
        ['A', 'C', 'G', 'T', 'N'],
        model_max_length=context_len
    )
    num_layers: int = 4
    vocab_size: int = tokenizer.vocab_size
    hidden_dim: int = 512
    head_index: str = "datasets/CHROMS_1_5_GATA1_CTCF_10_000/index.csv"
    use_MLP: bool = True
    use_pos_embs: bool = True
    use_MoE: bool = False
    num_experts: int = 4
    use_conv: bool = False
    use_bidirectionality: bool = False

    def save_cfg(self, path: str):

        data = asdict(self)

        data["tokenizer"] = {
            "class": self.tokenizer.__class__.__name__,
            "alphabet": self.tokenizer.get_vocab(),
            "model_max_length": self.tokenizer.model_max_length,
            "vocab_size": self.tokenizer.vocab_size
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

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
    lr: float = 1e-3
    batch_size: int = 32
    regression_loss: Callable = MSELoss()
    classification_loss: Callable = BCEWithLogitsLoss()
    optimizer: torch.optim.Optimizer = torch.optim.AdamW
    data_path: str = "datasets/CHROMS_1_5_GATA1_CTCF_10_000"
    data_file: str = "tracks_1_22_4K.json"
    wandb_project: str = "ToyGrant"
    num_epochs: int = 20
    train_chroms: List[str] = field(default_factory=lambda: ["chr1"])
    test_chroms: List[str] = field(default_factory=lambda: ["chr2"])
    comb_factor: float = 1
    wandb_name: str = f"TrackMambaNorm_{batch_size}BS_{num_epochs}Epochs_{comb_factor}lambda_2Heads"
    save_path: str = f"models/abl_bidir/TrackMamba_{batch_size}BS_{num_epochs}Epochs_{comb_factor}lambda_2Heads.pt"
    seed: int = 42

class TrackMambaMetadata:
    """
    This class represents Metadata for TrackMamba. Includes information about heads.
    """
    def __init__(self, head_index: str, **kwargs):
        """
        Args:
            - head_index : str. Is a text file indicating, for each prediction track, its head index
            - kwargs : Dict. Optional parameters for parsing input file.
        """
        self.index_file = head_index
        self.index: pd.DataFrame = pd.read_csv(head_index, **kwargs)
        self.num_heads = len(self.index)

    def get_head_index(self, track_id: str):
        return self.index.at[track_id, 1] # first col is experiment name
    
    def heads(self):
        return self.index[1].tolist()
    
    def get_exp_index(self, exp_name: str):
        return self.index[self.index[1] == exp_name].index.values[0].item()
    
    def save_metadata(self, path: str):
        self.index.to_csv(path, header=False)
    
        
