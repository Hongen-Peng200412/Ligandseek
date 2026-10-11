from __future__ import annotations

from collections.abc import Mapping

from dataclasses import dataclass, fields

from typing import Any

import torch

from torch import nn


@dataclass(frozen=True)
class TypedPointConfig:

    enabled: bool = True

    separate_qkv: bool = True

    separate_attn_proj: bool = True

    separate_ffn: bool = True

    separate_cpe: bool = True

    separate_embedding: bool = True

    separate_pooling_proj: bool = True

    separate_unpooling_proj: bool = True

    separate_fusion: bool = True

    separate_point_input_proj: bool = True

    separate_atom_token_proj: bool = True

    @property
    def use_separate_qkv(self) -> bool:

        return self.enabled and self.separate_qkv

    @property
    def use_separate_attn_proj(self) -> bool:

        return self.enabled and self.separate_attn_proj

    @property
    def use_separate_ffn(self) -> bool:

        return self.enabled and self.separate_ffn

    @property
    def use_separate_cpe(self) -> bool:

        return self.enabled and self.separate_cpe

    @property
    def use_separate_embedding(self) -> bool:

        return self.enabled and self.separate_embedding

    @property
    def use_separate_pooling_proj(self) -> bool:

        return self.enabled and self.separate_pooling_proj

    @property
    def use_separate_unpooling_proj(self) -> bool:

        return self.enabled and self.separate_unpooling_proj

    @property
    def use_separate_fusion(self) -> bool:

        return self.enabled and self.separate_fusion

    @property
    def use_separate_point_input_proj(self) -> bool:

        return self.enabled and self.separate_point_input_proj

    @property
    def use_separate_atom_token_proj(self) -> bool:

        return self.enabled and self.separate_atom_token_proj


def normalize_typed_point_cfg(
    cfg: Mapping[str, Any] | TypedPointConfig | None
) -> TypedPointConfig:

    if cfg is None:

        return TypedPointConfig()

    if isinstance(cfg, TypedPointConfig):

        return cfg

    valid_keys = {field.name for field in fields(TypedPointConfig)}

    cfg_dict = dict(cfg)

    unknown_keys = sorted(set(cfg_dict) - valid_keys)

    if unknown_keys:

        raise ValueError(f"typed_point_cfg contains unknown fields: {unknown_keys}")

    for key, value in cfg_dict.items():

        if not isinstance(value, bool):

            raise ValueError(
                f"typed_point_cfg.{key} must be bool, got {type(value).__name__}"
            )

    return TypedPointConfig(**cfg_dict)


def validate_pseudo_mask(
    pseudo_mask: torch.Tensor | None,
    point_count: int,
    *,
    name: str,
) -> torch.Tensor | None:

    if pseudo_mask is None:

        return None

    if pseudo_mask.dtype != torch.bool:

        raise RuntimeError(f"{name}: pseudo_mask must have bool dtype.")

    if pseudo_mask.ndim != 1:

        raise RuntimeError(f"{name}: pseudo_mask must be a 1D tensor.")

    if int(pseudo_mask.shape[0]) != int(point_count):

        raise RuntimeError(
            f"{name}: pseudo_mask length must be {point_count}, got {pseudo_mask.shape[0]}."
        )

    return pseudo_mask


def split_mask_state(
    pseudo_mask: torch.Tensor | None,
    point_count: int,
    *,
    name: str,
) -> tuple[torch.Tensor | None, bool, bool]:

    validated_mask = validate_pseudo_mask(pseudo_mask, point_count, name=name)

    if validated_mask is None:

        return None, point_count > 0, False

    has_pseudo = bool(validated_mask.any().item())

    has_real = bool((~validated_mask).any().item())

    return validated_mask, has_real, has_pseudo


def merge_type_aware_tensor_outputs(
    real_y: torch.Tensor,
    pseudo_y: torch.Tensor,
    pseudo_mask: torch.Tensor,
) -> torch.Tensor:

    output_dtype = torch.promote_types(real_y.dtype, pseudo_y.dtype)

    y = torch.empty(
        (int(pseudo_mask.shape[0]),) + tuple(real_y.shape[1:]),
        dtype=output_dtype,
        device=real_y.device,
    )

    y[~pseudo_mask] = real_y.to(dtype=output_dtype)

    y[pseudo_mask] = pseudo_y.to(dtype=output_dtype)

    return y


def apply_type_aware_tensor_module(
    x: torch.Tensor,
    pseudo_mask: torch.Tensor | None,
    real_module: nn.Module,
    pseudo_module: nn.Module,
) -> torch.Tensor:

    if int(x.shape[0]) == 0:

        return real_module(x)

    pseudo_mask = validate_pseudo_mask(
        pseudo_mask,
        int(x.shape[0]),
        name="apply_type_aware_tensor_module",
    )

    if pseudo_mask is None or not bool(pseudo_mask.any().item()):

        return real_module(x)

    if bool(pseudo_mask.all().item()):

        return pseudo_module(x)

    real_x = x[~pseudo_mask]

    pseudo_x = x[pseudo_mask]

    real_y = real_module(real_x)

    pseudo_y = pseudo_module(pseudo_x)

    return merge_type_aware_tensor_outputs(real_y, pseudo_y, pseudo_mask)


def make_typed_linear_norm_act(
    in_channels: int,
    out_channels: int,
    act_layer: type[nn.Module] | None,
) -> nn.Sequential:

    modules: list[nn.Module] = [
        nn.Linear(in_channels, out_channels),
        nn.LayerNorm(out_channels),
    ]

    if act_layer is not None:

        modules.append(act_layer())

    return nn.Sequential(*modules)
