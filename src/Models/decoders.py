from __future__ import annotations

import torch
import torch.nn as nn


class HNNLinearDecoder(nn.Module):
    """
    Baseline-faithful HNN decoder path.

    Shape flow:
        z_sys                : [B, latent_dim]
        dense expansion      : [B, window_size * hidden_dim]
        reshape              : [B, window_size, hidden_dim]
        output projection    : [B, window_size, output_dim]
    """

    def __init__(self, latent_dim: int, window_size: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.expand = nn.Linear(latent_dim, window_size * hidden_dim)
        self.activation = nn.Tanh()
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, z: torch.Tensor, context: dict | None = None) -> torch.Tensor:
        x = self.expand(z)
        x = self.activation(x)
        x = x.view(z.shape[0], self.window_size, self.hidden_dim)
        x = self.output_proj(x)
        return x


class ChannelLayerNorm(nn.Module):
    """LayerNorm over channel dimension for [B, C, T] tensors."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2)
        return x


class TemporalConvBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int = 3,
        dilation: int = 1,
        residual: bool = False,
    ):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.residual = residual
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.norm = ChannelLayerNorm(channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        y = self.norm(y)
        y = self.act(y)
        if self.residual:
            y = y + x
        return y


class HNNRepeatTCNDecoder(nn.Module):
    """
    Latent -> repeated temporal token -> temporal conv stack -> output.
    """

    def __init__(
        self,
        latent_dim: int,
        window_size: int,
        hidden_dim: int,
        output_dim: int,
        *,
        depth: int,
        residual: bool = False,
    ):
        super().__init__()
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.token_proj = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [
                TemporalConvBlock(
                    hidden_dim,
                    kernel_size=3,
                    dilation=1,
                    residual=residual,
                )
                for _ in range(depth)
            ]
        )
        self.temporal_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, z: torch.Tensor, context: dict | None = None) -> torch.Tensor:
        token = self.token_proj(z)
        x = token.unsqueeze(1).expand(-1, self.window_size, -1).contiguous()
        x = x.transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        x = x.transpose(1, 2)
        x = self.temporal_norm(x)
        x = self.output_proj(x)
        return x


class DirectRolloutReadoutDecoder(nn.Module):
    """
    Map rollout summary features directly to the reconstructed window.
    """

    def __init__(self, feature_dim: int, window_size: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.window_size = window_size
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.LayerNorm(6 * feature_dim),
            nn.Linear(6 * feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, window_size * output_dim),
        )

    def forward(self, z: torch.Tensor, context: dict | None = None) -> torch.Tensor:
        if context is None or 'rollout_features' not in context:
            raise ValueError('direct_rollout_readout requires rollout_features in decoder context')
        summary = context['rollout_features']
        x = self.net(summary)
        return x.view(summary.shape[0], self.window_size, self.output_dim)


def build_hnn_decoder(
    decoder_type: str,
    *,
    latent_dim: int,
    window_size: int,
    hidden_dim: int,
    output_dim: int,
) -> nn.Module:
    decoder_type = str(decoder_type).strip().lower()
    feature_dim = output_dim
    if decoder_type in {'linear', 'baseline', 'linear_baseline'}:
        return HNNLinearDecoder(
            latent_dim=latent_dim,
            window_size=window_size,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
        )
    if decoder_type == 'repeat_res_tcn_2':
        return HNNRepeatTCNDecoder(
            latent_dim=latent_dim,
            window_size=window_size,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            depth=2,
            residual=True,
        )
    if decoder_type == 'direct_rollout_readout':
        return DirectRolloutReadoutDecoder(
            feature_dim=feature_dim,
            window_size=window_size,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
        )
    raise ValueError(f'unsupported decoder_type: {decoder_type}')
