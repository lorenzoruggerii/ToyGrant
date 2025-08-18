import torch
import torch.nn as nn
from mamba_ssm import Mamba2
from config import TrackMambaConfig

class TrackMamba(nn.Module):
    """Mamba for genomic tracks prediction"""

    def __init__(self, cfg: TrackMambaConfig):
        super().__init__()
        self.cfg = cfg

        self.embedding = nn.Embedding(self.cfg.vocab_size, self.cfg.hidden_dim)
        self.pos_emb = nn.Embedding(self.cfg.context_len, self.cfg.hidden_dim)

        self.MambaBloks = nn.ModuleList([
            Mamba2(
                d_model=self.cfg.hidden_dim,
                d_state=64,
                d_conv=4,
                expand=2
            )
            for _ in range(self.cfg.num_layers)
        ])

        # Regression head for tracks
        self.track_head = nn.Linear(self.cfg.hidden_dim, 1)

        # Classification head for peak/no peak
        self.class_head = nn.Linear(self.cfg.hidden_dim, 1)

    def forward(self, x):
        
        # Add embeddings to positional embeddings
        embs = self.embedding(x)
        pos_embs = self.pos_emb(x)
        x = embs + pos_embs

        # Run into Mamba2 blocks
        for block in self.MambaBloks:
            x = block(x)

        # Get track value
        track_out = self.track_head(x).squeeze(-1)

        # Get peak value
        peak_out = self.class_head(x).squeeze(-1)

        return track_out, peak_out
        



