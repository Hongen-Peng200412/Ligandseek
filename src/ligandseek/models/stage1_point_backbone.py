from __future__ import annotations

from typing import Any

import torch
from torch import nn


def _activation(name: str) -> nn.Module:
    name = str(name).lower()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"Unsupported activation: {name}")


class Stage1PointBackbone(nn.Module):
                                                                              

    def __init__(
        self,
        atom_feature_dim: int,
        recycle_feature_dim: int = 96,
        input_embed_hidden_dim: int = 128,
        input_embed_dim: int = 64,
        out_channels: int = 96,
        recycle_in_norm_mode: str = "layernorm",
        use_separate_input_proj: bool = True,
        act_layer_name: str = "gelu",
        **_: Any,
    ) -> None:
        super().__init__()
        self.backend = "zeros"
        self.atom_feature_dim = int(atom_feature_dim)
        self.recycle_feature_dim = int(recycle_feature_dim)
        self.input_embed_dim = int(input_embed_dim)
        self.out_channels = int(out_channels)
        self.feature_channels_by_name = {"point_feat": self.out_channels}
        self.recycle_input_norm = (
            nn.LayerNorm(self.recycle_feature_dim)
            if str(recycle_in_norm_mode) == "layernorm"
            else None
        )
        self.recycle_input_proj = nn.Linear(self.recycle_feature_dim, self.atom_feature_dim)

        if use_separate_input_proj:
            self.atom_input_proj_real = self._make_input_proj(input_embed_hidden_dim, act_layer_name)
            self.atom_input_proj_pseudo = self._make_input_proj(input_embed_hidden_dim, act_layer_name)
            self.atom_input_proj = None
        else:
            self.atom_input_proj = self._make_input_proj(input_embed_hidden_dim, act_layer_name)
            self.atom_input_proj_real = None
            self.atom_input_proj_pseudo = None

    def _make_input_proj(self, hidden_dim: int, act_layer_name: str) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(self.atom_feature_dim, int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            _activation(act_layer_name),
            nn.Linear(int(hidden_dim), self.input_embed_dim),
        )

    def build_zeros_output(
        self,
        atom_feat: torch.Tensor,
        atom_coord_centered_world: torch.Tensor | None = None,
        atom_batch_index: torch.Tensor | None = None,
        atom_offsets: torch.Tensor | None = None,
        return_feature_names: tuple[str, ...] | list[str] = (),
        **_: Any,
    ) -> dict[str, Any]:
        n_points = int(atom_feat.shape[0])
        point_feat = atom_feat.new_zeros((n_points, self.out_channels))
        feature_dict = {
            name: point_feat
            for name in return_feature_names
            if name == "point_feat"
        }
        return {
            "point_feat": point_feat,
            "point_state": {
                "coord": atom_coord_centered_world,
                "batch": atom_batch_index,
                "offset": atom_offsets,
            },
            "point_recycle_out": point_feat,
            "point_feature_dict": feature_dict,
        }

    def forward(self, batch: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        return self.build_zeros_output(
            atom_feat=batch["atom_feat"],
            atom_coord_centered_world=batch.get("atom_coord_centered_world"),
            atom_batch_index=batch.get("atom_batch_index"),
            atom_offsets=batch.get("atom_offsets"),
            **kwargs,
        )
