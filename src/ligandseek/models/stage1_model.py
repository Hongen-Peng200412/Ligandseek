"""Build the four paper architectures and run their three voxel recycling passes."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ligandseek.models.stage1_embed_head import Stage1EmbedHead
from ligandseek.models.stage1_point_backbone import Stage1PointBackbone
from ligandseek.models.stage1_voxel_backbone import Stage1VoxelBackbone


class VolumePointStage1Model(nn.Module):

    def __init__(
        self,
        voxel_backbone: Stage1VoxelBackbone,
        point_backbone: Stage1PointBackbone,
        embed_head: Stage1EmbedHead,
        max_recycles: int = 3,
        detach_recycle_states: bool = True,
        **_: Any,
    ) -> None:
        super().__init__()
        self.voxel_backbone = voxel_backbone
        self.point_backbone = point_backbone
        self.embed_head = embed_head
        self.max_recycles = int(max_recycles)
        self.detach_recycle_states = bool(detach_recycle_states)

    def _build_voxel_input(
        self, batch: dict[str, Any], embed_output: dict[str, Any]
    ) -> torch.Tensor:
        voxel_grid = batch["voxel_grid"]
        embed_grid = embed_output.get("voxel_pdb_embed_grid")
        if embed_grid is None:
            return voxel_grid
        return torch.cat(
            [
                voxel_grid,
                embed_grid.to(device=voxel_grid.device, dtype=voxel_grid.dtype),
            ],
            dim=1,
        )

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        embed_output = self.embed_head(batch)
        voxel_input = self._build_voxel_input(batch, embed_output)
        voxel_recycle = None
        voxel_output = None
        for recycle_idx in range(max(1, self.max_recycles)):
            voxel_output = self.voxel_backbone(voxel_input, recycle_in=voxel_recycle)
            if recycle_idx < self.max_recycles - 1:
                voxel_recycle = voxel_output["voxel_recycle_out"]
                if self.detach_recycle_states:
                    voxel_recycle = voxel_recycle.detach()
        assert voxel_output is not None
        return {
            **voxel_output,
            "embed_output": embed_output,
            "recycle_passes_used": max(1, self.max_recycles),
        }


def build_release_model(variant: str) -> VolumePointStage1Model:
    variant = str(variant)
    configs: dict[str, dict[str, Any]] = {
        "unet_raw": {
            "in_channels": 1,
            "point_atom_dim": 49,
            "point_separate": True,
            "embed_hidden": 64,
            "embed_voxel": 0,
            "embed_point": 0,
            "embed_blocks": 0,
            "aux_hidden": 1,
            "aux_convs": 0,
        },
        "unet_diff": {
            "in_channels": 2,
            "point_atom_dim": 49,
            "point_separate": True,
            "embed_hidden": 64,
            "embed_voxel": 0,
            "embed_point": 0,
            "embed_blocks": 0,
            "aux_hidden": 1,
            "aux_convs": 0,
        },
        "unet_full": {
            "in_channels": 56,
            "point_atom_dim": 49,
            "point_separate": False,
            "embed_hidden": 64,
            "embed_voxel": 0,
            "embed_point": 0,
            "embed_blocks": 0,
            "aux_hidden": 1,
            "aux_convs": 0,
        },
        "ligandseek_pv": {
            "in_channels": 107,
            "point_atom_dim": 64,
            "point_separate": True,
            "embed_hidden": 128,
            "embed_voxel": 49,
            "embed_point": 64,
            "embed_blocks": 3,
            "aux_hidden": 32,
            "aux_convs": 2,
        },
    }
    if variant not in configs:
        raise KeyError(f"Unknown release variant: {variant}")
    cfg = configs[variant]

    voxel_backbone = Stage1VoxelBackbone(
        in_channels=cfg["in_channels"],
        feature_channels=48,
        planes=[64, 256, 256, 256, 256, 256, 128, 64, 64],
        gradient_checkpoint=False,
        return_feature_keys=[],
        aux_head_hidden_channels=cfg["aux_hidden"],
        num_conv3d_aux=cfg["aux_convs"],
        ligand_head_hidden_channels=32,
        num_conv3d_ligand=2,
        voxel_aux_logit_dim=1,
        voxel_ligand_logit_dim=1,
    )
    point_backbone = Stage1PointBackbone(
        atom_feature_dim=cfg["point_atom_dim"],
        recycle_feature_dim=96,
        input_embed_hidden_dim=128,
        input_embed_dim=64,
        out_channels=96,
        use_separate_input_proj=cfg["point_separate"],
        act_layer_name="gelu",
    )
    embed_head = Stage1EmbedHead(
        atom_feature_dim=49,
        embed_hidden_dim=cfg["embed_hidden"],
        embed_voxel_out_channels=cfg["embed_voxel"],
        embed_point_out_channels=cfg["embed_point"],
        num_trunk_blocks=0,
        num_voxel_blocks=cfg["embed_blocks"],
        num_point_blocks=0,
        trunk_buffer_radii=[],
        voxel_buffer_radii=[8.0, 4.0, 0.0] if variant == "ligandseek_pv" else [],
        point_buffer_radii=[],
        num_heads=4,
        patch_size=256,
        serialization_orders=["z", "z-trans", "hilbert", "hilbert-trans"],
        shuffle_orders=(variant == "ligandseek_pv"),
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.1 if variant == "ligandseek_pv" else 0.0,
        proj_drop=0.1 if variant == "ligandseek_pv" else 0.0,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        add_occupancy_channels=(variant == "ligandseek_pv"),
        use_centroid_encoding=(variant == "ligandseek_pv"),
        use_soft_splatting=(variant == "ligandseek_pv"),
        embed_residual_enabled=(variant == "ligandseek_pv"),
        embed_point_gate_enabled=False,
        embed_voxel_gate_enabled=False,
        act_layer_name="gelu",
        ffn_type="gated",
        mlp_ratio=3,
        scatter_reduce="sum",
        point_grid_size=0.25,
        cpe_impl="pointconv" if variant == "ligandseek_pv" else "none",
        cpe_kernel_size=5,
        cpe_receptive_field=2.0,
        pointconv_block_max_neighbors=64 if variant == "ligandseek_pv" else 16,
        drop_path=0.2 if variant == "ligandseek_pv" else 0.0,
        pre_norm=True,
    )
    return VolumePointStage1Model(
        voxel_backbone=voxel_backbone,
        point_backbone=point_backbone,
        embed_head=embed_head,
        max_recycles=3,
        detach_recycle_states=True,
    )
