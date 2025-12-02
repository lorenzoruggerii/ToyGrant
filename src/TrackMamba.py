import torch
import torch.nn as nn
from mamba_ssm import Mamba2
from moe import MoE
from config import TrackMambaConfig, TrackMambaMetadata
from mamba_ssm.utils.generation import InferenceParams

def rc_tokens(x, rc_map):
    """
    x: (B, T) tensor of token ids
    rc_map: tensor of shape (V,) mapping each token to its RC token
    """
    rc_map = rc_map.to(x.device)
    return rc_map[x].flip(dims=(-1,)).contiguous()

def rc_flip_preds(preds, attention_mask=None):
    # preds: (B, T, C)
    # mask: (B, T)
    preds_flipped = torch.flip(preds, dims=[1])
    return preds_flipped


class TrackMamba(nn.Module):
    """Mamba for genomic tracks prediction"""

    def __init__(self, cfg: TrackMambaConfig):
        super().__init__()
        self.cfg = cfg        

        self.metadata = TrackMambaMetadata(
            cfg.head_index,
            sep=",",
            header=None,
            index_col=0
        )

        # Store map for reverse complement
        self.rc_map = torch.tensor([
        #   A, C, G, T, N, PAD
            3, 2, 1, 0, 4, 5
        ])

        # self.embedding = nn.Embedding(self.cfg.vocab_size, self.cfg.hidden_dim)
        self.pos_emb = nn.Embedding(self.cfg.context_len, self.cfg.hidden_dim)

        # Init conv
        if self.cfg.use_conv:
            self.input_proj = (
                nn.Conv1d(
                    in_channels=4,
                    out_channels=self.cfg.hidden_dim,
                    kernel_size=3,
                    padding=1
                )
            )
        else:
            self.input_proj = (
                nn.Linear(
                    in_features=4,
                    out_features=self.cfg.hidden_dim
                )
            )

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

            if self.cfg.use_MoE:
                block["MoE"] = MoE(
                    d_model=self.cfg.hidden_dim,
                    d_hidden=4 * self.cfg.hidden_dim,
                    num_experts=self.cfg.num_experts
                )

            self.MambaBloks.append(block)

        # Regression head for tracks
        self.regression_heads = nn.ModuleDict(
            {head_idx: nn.Linear(self.cfg.hidden_dim, 1) for head_idx in self.metadata.heads()}
        )

        # Classification head for peak/no peak
        self.class_heads = nn.ModuleDict(
            {head_idx: nn.Linear(self.cfg.hidden_dim, 1) for head_idx in self.metadata.heads()}
        ) 

    def _forward(self, x):
        
        # Make one hot encoding
        # embs = self.embedding(x) # (B, T, hidden_dim)
        embs = self.one_hot(x)

        # Now expand to hidden_dim
        if self.cfg.use_conv:
            # Conv1d expects input of shape (B, C, T)
            x_proj = self.input_proj(embs.transpose(-1, -2))
            x_proj = x_proj.transpose(-1, -2)
        else:
            x_proj = self.input_proj(embs)


        # Create position indices: 0, 1, 2, ..., seq_len-1
        seq_len = x.shape[1]
        positions = torch.arange(seq_len, device=x.device)
        
        # Get embeddings
        pos_embs = (
            self.pos_emb(positions) if self.cfg.use_pos_embs else torch.zeros_like(embs)
        )

        x = x_proj + pos_embs

        # Run into Mamba2 blocks
        for blk in self.MambaBloks:

            # Residual mamba block: mamba
            res_x = x
            x = blk["mamba"](x)
            x = blk["norm"](x)
            x = res_x + x
            
            # Residual layer if MLP
            if "mlp" in blk:
                res = x
                x = blk["mlp"](x)
                x = res + x

            # Residual layer if MoE
            if "MoE" in blk:
                res = x
                x = blk["MoE"](x)
                x = res + x

        # Get track value
        track_outs = torch.cat(
            [head(x) for head in self.regression_heads.values()], dim=-1
        ) # (B, T, num_heads)
        
        # Get peak value
        peaks_out = torch.cat(
            [head(x) for head in self.class_heads.values()], dim=-1
        )

        return track_outs, peaks_out

    def forward(self, x, attention_mask=None):
        """
        The forward pass is mono or bi-directional.
        """

        # Get predictions from forward string
        y_regr, y_class = self._forward(x)

        # Compute reverse complement
        if self.cfg.use_bidirectionality:
            x_rc = rc_tokens(x, self.rc_map)
            
            # Make predictions on RC
            y_rc_reg, y_rc_class = self._forward(x_rc)

            # Flip RC predictions back
            y_rc_flip_reg = rc_flip_preds(y_rc_reg, attention_mask=attention_mask)
            y_rc_flip_class = rc_flip_preds(y_rc_class, attention_mask=attention_mask)

            # Average
            y_regr = 0.5 * (y_regr + y_rc_flip_reg)
            y_class = 0.5 * (y_class + y_rc_flip_class)

        return y_regr, y_class
    
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

            # Residual layer if MoE
            if "MoE" in blk:
                res = x
                x = blk["MoE"](x)
                x = res + x

        # Get track value
        track_value = self.track_head(res).squeeze(-1) # (B, L)

        return track_value
    
    @classmethod
    def from_pretrained(self, weight_path: str, config: TrackMambaConfig):
        """Load TrackMamba checkpoint"""
        model = TrackMamba(config)
        model.load_state_dict(torch.load(weight_path))
        return model
    
    def one_hot(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implements one hot encoding for a sequence.

        If sequence contains N it gets mapped to [0, 0, 0, 0]
        The vocabulary is: {A: 0, C: 1, G: 2, T: 3, N: 4, [PAD]: 5}
        """

        B, T = x.shape
        one_hot_full = torch.nn.functional.one_hot(x, num_classes=6).float()

        # Keep only first four channels
        one_hot = one_hot_full[..., :4]

        return one_hot
        



