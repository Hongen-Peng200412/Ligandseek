from __future__ import annotations

from typing import Sequence
import math

import torch
from torch import nn

from .raunet import SimpleUnet


class Stage1VoxelBackbone(SimpleUnet):
    def __init__(
        self,
        in_channels: int | None,
        feature_channels: int,
        planes: Sequence[int],
        gradient_checkpoint: bool,
        return_feature_keys: Sequence[str],
        aux_head_hidden_channels: int,
        num_conv3d_aux: int,
        ligand_head_hidden_channels: int,
        num_conv3d_ligand: int,
        prior_prob: float | None = None,
        voxel_aux_logit_dim: int = 1,
        voxel_ligand_logit_dim: int = 1,
        prior_probs: Sequence[float] | None = None,
        prior_prob_voxel_receptor: float | None = None,
        prior_prob_voxel_ligand: float | None = None,
    ) -> None:
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\

        super().__init__(
            in_channels=in_channels,
            out_channels=int(feature_channels),
            planes=planes,
            gradient_checkpoint=gradient_checkpoint,
        )

        self.feature_channels = int(feature_channels)
        self.voxel_aux_logit_dim = int(voxel_aux_logit_dim)
        self.voxel_ligand_logit_dim = int(voxel_ligand_logit_dim)
        self.return_feature_keys = tuple(str(key_name) for key_name in return_feature_keys)

        enc0, enc1, enc2, enc3, bottleneck, dec3, dec2, dec1, dec0 = [int(value) for value in planes]

        self.feature_channels_by_name = {
            "voxel_ds_0": int(enc0 * 4),
            "voxel_ds_1": enc1,
            "voxel_ds_2": enc2,
            "voxel_ds_3": enc3,
            "voxel_ds_4": bottleneck,
            "voxel_c4": bottleneck,
            "voxel_c3": dec3,
            "voxel_c2": dec2,
            "voxel_c1": dec1,
            "voxel_c0": dec0,
            "voxel_final": self.feature_channels,
        }


        if int(aux_head_hidden_channels) > 0:
            aux_layers: list[nn.Module] = []
            for _ in range(int(num_conv3d_aux)):
                aux_layers.append(nn.Conv3d(self.feature_channels if len(aux_layers) == 0 else int(aux_head_hidden_channels), int(aux_head_hidden_channels), kernel_size=3, padding=1))
                aux_layers.append(nn.ReLU())

            _last_in = int(aux_head_hidden_channels) if len(aux_layers) > 0 else self.feature_channels
            aux_layers.append(nn.Conv3d(_last_in, self.voxel_aux_logit_dim, kernel_size=1))
            self.voxel_aux_head = nn.Sequential(*aux_layers)
        else:
            if self.feature_channels != self.voxel_aux_logit_dim:
                raise ValueError(
                    "aux_head_hidden_channels <= 0 requires feature_channels == voxel_aux_logit_dim, "
                    f"actual feature_channels={self.feature_channels}, voxel_aux_logit_dim={self.voxel_aux_logit_dim}"
                )
            self.voxel_aux_head = None


        if int(ligand_head_hidden_channels) > 0:
            ligand_layers: list[nn.Module] = []
            for _ in range(int(num_conv3d_ligand)):
                ligand_layers.append(nn.Conv3d(self.feature_channels if len(ligand_layers) == 0 else int(ligand_head_hidden_channels), int(ligand_head_hidden_channels), kernel_size=3, padding=1))
                ligand_layers.append(nn.ReLU())
            _last_in_lig = int(ligand_head_hidden_channels) if len(ligand_layers) > 0 else self.feature_channels
            ligand_layers.append(nn.Conv3d(_last_in_lig, self.voxel_ligand_logit_dim, kernel_size=1))
            self.voxel_ligand_head = nn.Sequential(*ligand_layers)
        else:
            self.voxel_ligand_head = None

        if prior_probs is not None:

            self._init_multiclass_prior_bias(self.voxel_aux_head, self.voxel_aux_logit_dim, prior_probs)
            self._init_multiclass_prior_bias(self.voxel_ligand_head, self.voxel_ligand_logit_dim, prior_probs)
        else:


            aux_prior = prior_prob_voxel_receptor if prior_prob_voxel_receptor is not None else prior_prob
            ligand_prior = prior_prob_voxel_ligand if prior_prob_voxel_ligand is not None else prior_prob
            if aux_prior is not None:
                if self.voxel_aux_logit_dim != 1:
                    raise ValueError("multi-channel softmax heads should use prior_probs, not single-channel voxel aux priors")

                aux_bias_val = -math.log((1.0 - float(aux_prior)) / float(aux_prior))
                if self.voxel_aux_head is not None:
                    nn.init.constant_(self.voxel_aux_head[2 * int(num_conv3d_aux)].bias, aux_bias_val)
                elif self.conv_end.bias is not None:
                    nn.init.constant_(self.conv_end.bias, aux_bias_val)
            if ligand_prior is not None:
                if self.voxel_ligand_logit_dim != 1:
                    raise ValueError("multi-channel softmax heads should use prior_probs, not single-channel voxel ligand priors")

                ligand_bias_val = -math.log((1.0 - float(ligand_prior)) / float(ligand_prior))
                if self.voxel_ligand_head is not None:
                    nn.init.constant_(self.voxel_ligand_head[2 * int(num_conv3d_ligand)].bias, ligand_bias_val)


        self.voxel_recycle_proj = nn.Conv3d(
            in_channels=self.feature_channels,
            out_channels=int(self.shortconvadd.output_channels),
            kernel_size=1,
        )

    @staticmethod
    def _init_multiclass_prior_bias(head: nn.Module | None, logit_dim: int, prior_probs: Sequence[float]) -> None:
\
\
\
\
\
\
\
\
\
\

        if head is None:
            return
        if int(logit_dim) <= 1:
            raise ValueError("prior_probs only applies to multi-channel softmax heads")

        probs = torch.as_tensor(list(prior_probs), dtype=torch.float32)
        if probs.numel() != int(logit_dim):
            raise ValueError(f"prior_probs length {probs.numel()} and logit_dim={logit_dim} mismatch")
        if torch.any(probs <= 0):
            raise ValueError("all prior_probs entries must be greater than 0")
        if not torch.isclose(probs.sum(), torch.tensor(1.0), rtol=1e-4, atol=1e-6):
            raise ValueError(f"prior_probs sum must be 1, actual {float(probs.sum())}")

        last_layer = head[-1]
        if not isinstance(last_layer, nn.Conv3d) or last_layer.bias is None:
            raise TypeError("multiclass prior initialization requires the final head layer to be Conv3d with bias")
        with torch.no_grad():

            last_layer.bias.copy_(probs.log().to(device=last_layer.bias.device, dtype=last_layer.bias.dtype))


    def _forward_single_pass(
        self,
        voxel_grid: torch.Tensor,
        recycle_in: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
\
\
\
\
\
\
\
\
\

        batch_size, _, depth, height, width = voxel_grid.shape
        if recycle_in is None:

            voxel_recycle = torch.zeros(
                (batch_size, int(self.shortconvadd.output_channels), depth, height, width),
                device=voxel_grid.device,
                dtype=voxel_grid.dtype,
            )
        else:

            voxel_recycle = recycle_in.to(device=voxel_grid.device, dtype=voxel_grid.dtype)


        fused_input = self.shortconvadd(voxel_grid, voxel_recycle)


        ds_0 = self.shortconv1(fused_input)

        ds_1 = self._checkpoint_call(self.downsample1, ds_0)

        ds_2 = self._checkpoint_call(self.downsample2, ds_1)

        ds_3 = self._checkpoint_call(self.downsample3, ds_2)

        ds_4 = self._checkpoint_call(self.downsample4, ds_3)


        c4 = self._checkpoint_call(self.A_block, ds_4)


        c3 = self._checkpoint_call(self.main1, self.upsample_add(c4, ds_3))

        c2 = self._checkpoint_call(self.main2, self._checkpoint_call(self.attn2, c3, ds_2))

        c1 = self._checkpoint_call(self.main3, self._checkpoint_call(self.attn3, c2, ds_1))

        c0 = self._checkpoint_call(self.main4, self._checkpoint_call(self.attn4, c1, ds_0))


        f3 = self.conv_end_3(c0)

        f5 = self.conv_end_5(c0)

        f7 = self.conv_end_7(c0)

        fused_multiscale = self.relu1(torch.cat((f3, f5, f7), dim=1))

        final = self.conv_end(fused_multiscale)


        all_feature_dict = {
            "voxel_fused_input": fused_input,
            "voxel_ds_0": ds_0,
            "voxel_ds_1": ds_1,
            "voxel_ds_2": ds_2,
            "voxel_ds_3": ds_3,
            "voxel_ds_4": ds_4,
            "voxel_c4": c4,
            "voxel_c3": c3,
            "voxel_c2": c2,
            "voxel_c1": c1,
            "voxel_c0": c0,
            "voxel_f3": f3,
            "voxel_f5": f5,
            "voxel_f7": f7,
            "voxel_fused_multiscale": fused_multiscale,
            "voxel_final": final,
        }
        return all_feature_dict


    def forward_features(
        self,
        voxel_grid: torch.Tensor,
        recycle_in: torch.Tensor | None,
        return_feature_keys: Sequence[str],
    ) -> dict[str, torch.Tensor]:
\
\
\
\
\
\
\
\
\
\

        all_feature_dict = self._forward_single_pass(voxel_grid=voxel_grid, recycle_in=recycle_in)
        requested_feature_keys = tuple(str(key_name) for key_name in return_feature_keys)

        return {key_name: all_feature_dict[key_name] for key_name in requested_feature_keys}


    def forward(
        self,
        voxel_grid: torch.Tensor,
        recycle_in: torch.Tensor | None = None,
        return_feature_keys: Sequence[str] | None = None,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
\
\
\
\
\
\
\
\
\
\
\
\
\

        requested_feature_keys = (
            self.return_feature_keys
            if return_feature_keys is None
            else tuple(str(key_name) for key_name in return_feature_keys)
        )
        all_feature_dict = self._forward_single_pass(voxel_grid=voxel_grid, recycle_in=recycle_in)
        selected_feature_dict = {key_name: all_feature_dict[key_name] for key_name in requested_feature_keys}



        if self.voxel_aux_head is not None:
            voxel_logits_aux = self.voxel_aux_head(all_feature_dict["voxel_final"])
        else:
            voxel_logits_aux = all_feature_dict["voxel_final"]

        voxel_logits_ligand = None
        if self.voxel_ligand_head is not None:
            voxel_logits_ligand = self.voxel_ligand_head(all_feature_dict["voxel_final"])


        voxel_recycle_out = self.voxel_recycle_proj(all_feature_dict["voxel_final"])

        return {
            "voxel_features": selected_feature_dict,
            "voxel_logits_aux": voxel_logits_aux,
            "voxel_logits_ligand": voxel_logits_ligand,
            "voxel_recycle_out": voxel_recycle_out,
        }
