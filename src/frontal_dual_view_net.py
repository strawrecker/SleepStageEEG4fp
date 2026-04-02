from __future__ import annotations

import torch
from torch import nn


class FrontalDualViewNet(nn.Module):
    def __init__(
        self,
        channel_feature_dim: int,
        total_feature_dim: int,
        context_size: int,
        num_classes: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        abs_dim = channel_feature_dim * 2 * context_size
        diff_dim = channel_feature_dim * context_size
        aux_dim = max(total_feature_dim - channel_feature_dim * 3, 0) * context_size
        diff_hidden = max(hidden_dim // 2, 32)
        aux_hidden = max(hidden_dim // 4, 16) if aux_dim > 0 else 0

        self.absolute_branch = nn.Sequential(
            nn.LayerNorm(abs_dim),
            nn.Linear(abs_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.diff_branch = nn.Sequential(
            nn.LayerNorm(diff_dim),
            nn.Linear(diff_dim, diff_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(diff_hidden, diff_hidden),
            nn.GELU(),
        )
        self.aux_branch = (
            nn.Sequential(
                nn.LayerNorm(aux_dim),
                nn.Linear(aux_dim, aux_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(aux_hidden, aux_hidden),
                nn.GELU(),
            )
            if aux_dim > 0
            else None
        )
        fusion_dim = hidden_dim + diff_hidden + (aux_hidden if aux_dim > 0 else 0)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self.channel_feature_dim = channel_feature_dim
        self.total_feature_dim = total_feature_dim
        self.aux_per_epoch_dim = max(total_feature_dim - channel_feature_dim * 3, 0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, context_size, feature_dim = inputs.shape
        split_index = self.channel_feature_dim * 2
        diff_end = self.channel_feature_dim * 3
        absolute = inputs[:, :, :split_index].reshape(batch_size, context_size * split_index)
        differential = inputs[:, :, split_index:diff_end].reshape(
            batch_size,
            context_size * self.channel_feature_dim,
        )
        parts = [self.absolute_branch(absolute), self.diff_branch(differential)]
        if self.aux_branch is not None and self.aux_per_epoch_dim > 0:
            aux = inputs[:, :, diff_end:feature_dim].reshape(batch_size, context_size * self.aux_per_epoch_dim)
            parts.append(self.aux_branch(aux))
        fused = torch.cat(parts, dim=-1)
        return self.classifier(fused)
