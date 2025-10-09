import torch
import torch.nn as nn
from mamba_ssm import Mamba2
from config import TrackMambaConfig
from mamba_ssm.utils.generation import InferenceParams

class TrackMamba(nn.Module):
    """Mamba for genomic tracks prediction"""

    def __init__(self, cfg: TrackMambaConfig):
        super().__init__()
        self.cfg = cfg

        self.embedding = nn.Embedding(self.cfg.vocab_size, self.cfg.hidden_dim)
        self.pos_emb = nn.Embedding(self.cfg.context_len, self.cfg.hidden_dim)

        self.MambaBloks = nn.ModuleList([])

        for i in range(self.cfg.num_layers):
            block = nn.ModuleDict({
                "mamba": Mamba2(
                d_model=self.cfg.hidden_dim,
                d_state=64,
                d_conv=4,
                expand=2,
                layer_idx=i
                ),
                "norm": nn.LayerNorm(self.cfg.hidden_dim),
            })

            if self.cfg.use_MLP:
                block["mlp"] = nn.Sequential(
                    nn.Linear(self.cfg.hidden_dim, 4 * self.cfg.hidden_dim),
                    nn.SiLU(),
                    nn.Linear(4 * self.cfg.hidden_dim, self.cfg.hidden_dim),
                )

            self.MambaBloks.append(block)

        # Regression head for tracks
        self.track_head = nn.Linear(self.cfg.hidden_dim, 1)

        # Classification head for peak/no peak
        self.class_head = nn.Linear(self.cfg.hidden_dim, 1)

    def forward(self, x):
        
        # Add embeddings to positional embeddings
        embs = self.embedding(x)

        # Create position indices: 0, 1, 2, ..., seq_len-1
        seq_len = x.shape[1]
        positions = torch.arange(seq_len, device=x.device)
        
        # Get embeddings
        pos_embs = (
            self.pos_emb(positions) if self.cfg.use_pos_embs else torch.zeros_like(embs)
        )

        x = embs + pos_embs

        # Run into Mamba2 blocks
        for blk in self.MambaBloks:
            x = blk["mamba"](x)
            x = blk["norm"](x)
            
            # Residual layer if MLP
            if "mlp" in blk:
                res = x
                x = blk["mlp"](x)
                x = res + x

        # Get track value
        track_out = self.track_head(x).squeeze(-1)

        # Get peak value
        peak_out = self.class_head(x).squeeze(-1)

        return track_out, peak_out
    
    @torch.no_grad()
    def scale_context(self, x: torch.Tensor, use_forward: bool = True):
        """
        Use pretrained TrackMamba model to scale to context_to_scale len.
        
        For now Mamba2 allows only inference step by step, not parallel.
        """

        context_to_scale = x.shape[1]
        batch_size = x.shape[0]

        if (context_to_scale <= self.cfg.context_len):
            if use_forward:
                track_out, _ = self.forward(x)
                return track_out
        
        self.eval()

        # Initialize inference params
        inference_params = InferenceParams(max_seqlen=context_to_scale, max_batch_size=batch_size, seqlen_offset=0)

        # Embed the input
        res = self.embedding(x)

        # Add pos encoding if necessary
        if self.cfg.use_pos_embs:
            num_chunks = (context_to_scale + self.cfg.context_len - 1) // self.cfg.context_len
            
            
            for i in range(num_chunks):
                positions = torch.arange(self.cfg.context_len, device=res.device)
                pos_embs = self.pos_emb(positions)

                # Define start and end points
                start = i*self.cfg.context_len
                end = min((i+1)*self.cfg.context_len, context_to_scale)

                res[:, start:end, :] += pos_embs[:(end-start), :]
        
        # Run everything in sequential mode
        for i, blk in enumerate(self.MambaBloks):
            
            # Store activations from Mamba layer
            out_mamba = torch.zeros_like(res)
            inference_params.reset(max_seqlen=context_to_scale, max_batch_size=batch_size)

            # Cycle for obtaining the activations using previous hidden states
            # https://github.com/state-spaces/mamba/issues/536

            for token_idx in range(context_to_scale):
                in_mamba = res[:, token_idx:token_idx+1, :]
                x = blk["mamba"](in_mamba, inference_params=inference_params)
                inference_params.seqlen_offset += 1
                out_mamba[:, token_idx, :] = x.squeeze()
            
            # Normalization layer
            res = blk["norm"](out_mamba) # (B, L, H)

            # MLP layer
            if "mlp" in blk:
                in_mlp = res
                out_mlp = blk["mlp"](in_mlp)
                res = res + out_mlp

        # Get track value
        track_value = self.track_head(res).squeeze(-1) # (B, L)

        return track_value
    
    @classmethod
    def from_pretrained(self, weight_path: str, config: TrackMambaConfig):
        """Load TrackMamba checkpoint"""
        model = TrackMamba(config)
        model.load_state_dict(torch.load(weight_path))
        return model
        



