import sys

from typing import List

from functools import partial

import math

import torch

import torch.nn as nn

from collections import OrderedDict

try:

    from addict import Dict

except ImportError:

    class Dict(dict):

        def __getattr__(self, name):

            try:

                return self[name]

            except KeyError as exc:

                raise AttributeError(name) from exc

        def __setattr__(self, name, value):

            self[name] = value

        def __delattr__(self, name):

            try:

                del self[name]

            except KeyError as exc:

                raise AttributeError(name) from exc


try:

    import torch_scatter

except ImportError:

    class _TorchScatterFallback:

        @staticmethod
        def scatter_mean(src, index, dim=0, dim_size=None):

            if dim != 0:

                raise NotImplementedError("scatter_mean fallback only supports dim=0.")

            if dim_size is None:

                dim_size = int(index.max().item()) + 1 if index.numel() else 0

            out = src.new_zeros((int(dim_size),) + tuple(src.shape[1:]))

            count_shape = (int(dim_size),) + (1,) * (src.ndim - 1)

            count = src.new_zeros(count_shape)

            if index.numel():

                out.index_add_(0, index.long(), src)

                one_shape = (int(index.numel()),) + (1,) * (src.ndim - 1)

                count.index_add_(0, index.long(), src.new_ones(one_shape))

            return out / count.clamp_min(1)

        @staticmethod
        def segment_csr(src, indptr, reduce="sum"):

            chunks = []

            for start, end in zip(indptr[:-1].tolist(), indptr[1:].tolist()):

                segment = src[int(start) : int(end)]

                if segment.numel() == 0:

                    chunks.append(src.new_zeros(src.shape[1:]))

                elif reduce == "mean":

                    chunks.append(segment.mean(dim=0))

                elif reduce == "sum":

                    chunks.append(segment.sum(dim=0))

                elif reduce == "max":

                    chunks.append(segment.max(dim=0).values)

                else:

                    raise NotImplementedError(
                        f"segment_csr fallback does not support reduce={reduce!r}."
                    )

            if not chunks:

                return src.new_empty((0,) + tuple(src.shape[1:]))

            return torch.stack(chunks, dim=0)

    torch_scatter = _TorchScatterFallback()

try:

    import torch_cluster

except ImportError:

    torch_cluster = None

try:

    from timm.models.layers import DropPath

except ImportError:

    class DropPath(nn.Module):

        def __init__(self, drop_prob=0.0):

            super().__init__()

            self.drop_prob = float(drop_prob)

        def forward(self, x):

            if self.drop_prob == 0.0 or not self.training:

                return x

            keep_prob = 1.0 - self.drop_prob

            shape = (x.shape[0],) + (1,) * (x.ndim - 1)

            mask = x.new_empty(shape).bernoulli_(keep_prob)

            return x.div(keep_prob) * mask


try:

    import flash_attn

except ImportError:

    flash_attn = None

from .serialization import encode

from ligandseek.models.typed_point import (
    apply_type_aware_tensor_module,
    make_typed_linear_norm_act,
    merge_type_aware_tensor_outputs,
    split_mask_state,
    validate_pseudo_mask,
)


@torch._dynamo.disable
def _radius_graph_eager(coord, receptive_field, batch, max_neighbors):

    if torch_cluster is None:

        rows = []

        cols = []

        for batch_id in batch.unique(sorted=True):

            local = (batch == batch_id).nonzero(as_tuple=True)[0]

            if local.numel() == 0:

                continue

            dist = torch.cdist(coord[local], coord[local])

            within = (dist <= float(receptive_field)) & (dist > 0)

            for local_i in range(int(local.numel())):

                neigh = within[local_i].nonzero(as_tuple=True)[0]

                if neigh.numel() > int(max_neighbors):

                    order = torch.argsort(dist[local_i, neigh])[: int(max_neighbors)]

                    neigh = neigh[order]

                if neigh.numel():

                    rows.append(local[local_i].expand_as(neigh))

                    cols.append(local[neigh])

        if not rows:

            return torch.empty((2, 0), dtype=torch.long, device=coord.device)

        return torch.stack([torch.cat(rows), torch.cat(cols)], dim=0)

    return torch_cluster.radius_graph(
        x=coord,
        r=receptive_field,
        batch=batch,
        max_num_neighbors=max_neighbors,
        loop=False,
    )


@torch.inference_mode()
def offset2bincount(offset):

    return torch.diff(
        offset, prepend=torch.tensor([0], device=offset.device, dtype=torch.long)
    )


@torch.inference_mode()
def offset2batch(offset):

    bincount = offset2bincount(offset)

    return torch.arange(
        len(bincount), device=offset.device, dtype=torch.long
    ).repeat_interleave(bincount)


@torch.inference_mode()
def batch2offset(batch):

    return torch.cumsum(batch.bincount(), dim=0).long()


class Point(Dict):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        if "batch" not in self.keys() and "offset" in self.keys():

            self["batch"] = offset2batch(self.offset)

        elif "offset" not in self.keys() and "batch" in self.keys():

            self["offset"] = batch2offset(self.batch)

    def serialization(self, order="z", depth=None, shuffle_orders=False):

        assert "batch" in self.keys()

        if "grid_coord" not in self.keys():

            assert {"grid_size", "coord"}.issubset(self.keys())

            self["grid_coord"] = torch.div(
                self.coord - self.coord.min(0)[0], self.grid_size, rounding_mode="trunc"
            ).int()

        if depth is None:

            depth = max(int(self.grid_coord.max()).bit_length(), 1)

        self["serialized_depth"] = depth

        assert depth * 3 + len(self.offset).bit_length() <= 63

        assert depth <= 16

        code = [
            encode(self.grid_coord, self.batch, depth, order=order_) for order_ in order
        ]

        code = torch.stack(code)

        order = torch.argsort(code)

        inverse = torch.zeros_like(order).scatter_(
            dim=1,
            index=order,
            src=torch.arange(0, code.shape[1], device=order.device).repeat(
                code.shape[0], 1
            ),
        )

        if shuffle_orders:

            perm = torch.randperm(code.shape[0])

            code = code[perm]

            order = order[perm]

            inverse = inverse[perm]

        self["serialized_code"] = code

        self["serialized_order"] = order

        self["serialized_inverse"] = inverse

    def sparsify(self, _pad=96):

        assert {"feat", "batch"}.issubset(self.keys())

        if "grid_coord" not in self.keys():

            assert {"grid_size", "coord"}.issubset(self.keys())

            self["grid_coord"] = torch.div(
                self.coord - self.coord.min(0)[0], self.grid_size, rounding_mode="trunc"
            ).int()


class PointModule(nn.Module):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)


class PointSequential(PointModule):

    def __init__(self, *args, **kwargs):

        super().__init__()

        if len(args) == 1 and isinstance(args[0], OrderedDict):

            for key, module in args[0].items():

                self.add_module(key, module)

        else:

            for idx, module in enumerate(args):

                self.add_module(str(idx), module)

        for name, module in kwargs.items():

            if sys.version_info < (3, 6):

                raise ValueError("kwargs only supported in py36+")

            if name in self._modules:

                raise ValueError("name exists.")

            self.add_module(name, module)

    def __getitem__(self, idx):

        if not (-len(self) <= idx < len(self)):

            raise IndexError("index {} is out of range".format(idx))

        if idx < 0:

            idx += len(self)

        it = iter(self._modules.values())

        for i in range(idx):

            next(it)

        return next(it)

    def __len__(self):

        return len(self._modules)

    def add(self, module, name=None):

        if name is None:

            name = str(len(self._modules))

            if name in self._modules:

                raise KeyError("name exists")

        self.add_module(name, module)

    def forward(self, input):

        for module in self._modules.values():

            if isinstance(module, PointModule):

                input = module(input)

            else:

                if isinstance(input, Point):

                    input.feat = module(input.feat)

                else:

                    input = module(input)

        return input


class PDNorm(PointModule):

    def __init__(
        self,
        num_features,
        norm_layer,
        context_channels=256,
        conditions=("ScanNet", "S3DIS", "Structured3D"),
        decouple=True,
        adaptive=False,
    ):

        super().__init__()

        self.conditions = conditions

        self.decouple = decouple

        self.adaptive = adaptive

        if self.decouple:

            self.norm = nn.ModuleList([norm_layer(num_features) for _ in conditions])

        else:

            self.norm = norm_layer

        if self.adaptive:

            self.modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(context_channels, 2 * num_features, bias=True)
            )

    def forward(self, point):

        assert {"feat", "condition"}.issubset(point.keys())

        if isinstance(point.condition, str):

            condition = point.condition

        else:

            condition = point.condition[0]

        if self.decouple:

            assert condition in self.conditions

            norm = self.norm[self.conditions.index(condition)]

        else:

            norm = self.norm

        point.feat = norm(point.feat)

        if self.adaptive:

            assert "context" in point.keys()

            shift, scale = self.modulation(point.context).chunk(2, dim=1)

            point.feat = point.feat * (1.0 + scale) + shift

        return point


class RPE(torch.nn.Module):

    def __init__(self, patch_size, num_heads):

        super().__init__()

        self.patch_size = patch_size

        self.num_heads = num_heads

        self.pos_bnd = int((4 * patch_size) ** (1 / 3) * 2)

        self.rpe_num = 2 * self.pos_bnd + 1

        self.rpe_table = torch.nn.Parameter(torch.zeros(3 * self.rpe_num, num_heads))

        torch.nn.init.trunc_normal_(self.rpe_table, std=0.02)

    def forward(self, coord):

        idx = (
            coord.clamp(-self.pos_bnd, self.pos_bnd)
            + self.pos_bnd
            + torch.arange(3, device=coord.device) * self.rpe_num
        )

        out = self.rpe_table.index_select(dim=0, index=idx.reshape(-1))

        out = out.view(idx.shape + (-1,)).sum(3)

        out = out.permute(0, 3, 1, 2)

        return out


class MLP(nn.Module):

    def __init__(
        self,
        in_channels,
        hidden_channels=None,
        out_channels=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):

        super().__init__()

        out_channels = out_channels or in_channels

        hidden_channels = hidden_channels or in_channels

        self.fc1 = nn.Linear(in_channels, hidden_channels)

        self.act = act_layer()

        self.fc2 = nn.Linear(hidden_channels, out_channels)

        self.drop = nn.Dropout(drop)

    def forward(self, x):

        x = self.fc1(x)

        x = self.act(x)

        x = self.drop(x)

        x = self.fc2(x)

        x = self.drop(x)

        return x


_ACT_LAYER_MAP = {
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "relu": nn.ReLU,
    "leakyrelu": nn.LeakyReLU,
}


def resolve_act_layer(act_layer_name: str):

    key = str(act_layer_name).lower()

    if key not in _ACT_LAYER_MAP:

        raise ValueError(
            f"Unsupported act_layer_name={act_layer_name!r}; choices: {list(_ACT_LAYER_MAP)}"
        )

    return _ACT_LAYER_MAP[key]


class GatedTransition(nn.Module):

    def __init__(self, in_channels: int, mlp_ratio: int, act_layer, drop: float):

        super().__init__()

        hidden = int(in_channels) * int(mlp_ratio)

        self.w1 = nn.Linear(in_channels, hidden, bias=False)

        self.w2 = nn.Linear(in_channels, hidden, bias=False)

        self.w3 = nn.Linear(hidden, in_channels, bias=False)

        self.act = act_layer()

        self.drop = nn.Dropout(drop)

    def forward(self, x):

        y = self.w3(self.act(self.w1(x)) * self.w2(x))

        return self.drop(y)


class SerializedAttention(PointModule):

    def __init__(
        self,
        channels,
        num_heads,
        patch_size,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        order_index=0,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=True,
        upcast_softmax=True,
        separate_qkv=False,
        separate_attn_proj=False,
    ):

        super().__init__()

        assert channels % num_heads == 0

        self.channels = channels

        self.num_heads = num_heads

        self.scale = qk_scale or (channels // num_heads) ** -0.5

        self.order_index = order_index

        self.upcast_attention = upcast_attention

        self.upcast_softmax = upcast_softmax

        self.enable_rpe = enable_rpe

        self.enable_flash = bool(enable_flash and flash_attn is not None)

        if self.enable_flash:

            assert (
                enable_rpe is False
            ), "Set enable_rpe to False when enable Flash Attention"

            assert (
                upcast_attention is False
            ), "Set upcast_attention to False when enable Flash Attention"

            assert (
                upcast_softmax is False
            ), "Set upcast_softmax to False when enable Flash Attention"

            self.patch_size = patch_size

            self.attn_drop = attn_drop

        else:

            self.patch_size_max = patch_size

            self.patch_size = 0

            self.attn_drop = torch.nn.Dropout(attn_drop)

        self.separate_qkv = bool(separate_qkv)

        self.separate_attn_proj = bool(separate_attn_proj)

        if self.separate_qkv:

            self.qkv_real = torch.nn.Linear(channels, channels * 3, bias=qkv_bias)

            self.qkv_pseudo = torch.nn.Linear(channels, channels * 3, bias=qkv_bias)

        else:

            self.qkv = torch.nn.Linear(channels, channels * 3, bias=qkv_bias)

        if self.separate_attn_proj:

            self.proj_real = torch.nn.Linear(channels, channels)

            self.proj_pseudo = torch.nn.Linear(channels, channels)

        else:

            self.proj = torch.nn.Linear(channels, channels)

        self.proj_drop = torch.nn.Dropout(proj_drop)

        self.softmax = torch.nn.Softmax(dim=-1)

        self.rpe = RPE(patch_size, num_heads) if self.enable_rpe else None

    @torch.no_grad()
    def get_rel_pos(self, point, order):

        K = self.patch_size

        rel_pos_key = f"rel_pos_{self.order_index}"

        if rel_pos_key not in point.keys():

            grid_coord = point.grid_coord[order]

            grid_coord = grid_coord.reshape(-1, K, 3)

            point[rel_pos_key] = grid_coord.unsqueeze(2) - grid_coord.unsqueeze(1)

        return point[rel_pos_key]

    @torch.no_grad()
    def get_padding_and_inverse(self, point):

        pad_key = f"pad_{self.patch_size}"

        unpad_key = f"unpad_{self.patch_size}"

        cu_seqlens_key = f"cu_seqlens_{self.patch_size}"

        if (
            pad_key not in point.keys()
            or unpad_key not in point.keys()
            or cu_seqlens_key not in point.keys()
        ):

            offset = point.offset

            bincount = offset2bincount(offset)

            bincount_pad = (
                torch.div(
                    bincount + self.patch_size - 1,
                    self.patch_size,
                    rounding_mode="trunc",
                )
                * self.patch_size
            )

            mask_pad = bincount > self.patch_size

            bincount_pad = ~mask_pad * bincount + mask_pad * bincount_pad

            _offset = nn.functional.pad(offset, (1, 0))

            _offset_pad = nn.functional.pad(torch.cumsum(bincount_pad, dim=0), (1, 0))

            pad = torch.arange(_offset_pad[-1], device=offset.device)

            unpad = torch.arange(_offset[-1], device=offset.device)

            cu_seqlens = []

            for i in range(len(offset)):

                unpad[_offset[i] : _offset[i + 1]] += _offset_pad[i] - _offset[i]

                if bincount[i] != bincount_pad[i]:

                    pad[
                        _offset_pad[i + 1]
                        - self.patch_size
                        + (bincount[i] % self.patch_size) : _offset_pad[i + 1]
                    ] = pad[
                        _offset_pad[i + 1]
                        - 2 * self.patch_size
                        + (bincount[i] % self.patch_size) : _offset_pad[i + 1]
                        - self.patch_size
                    ]

                pad[_offset_pad[i] : _offset_pad[i + 1]] -= _offset_pad[i] - _offset[i]

                cu_seqlens.append(
                    torch.arange(
                        _offset_pad[i],
                        _offset_pad[i + 1],
                        step=self.patch_size,
                        dtype=torch.int32,
                        device=offset.device,
                    )
                )

            point[pad_key] = pad

            point[unpad_key] = unpad

            point[cu_seqlens_key] = nn.functional.pad(
                torch.concat(cu_seqlens), (0, 1), value=_offset_pad[-1]
            )

        return point[pad_key], point[unpad_key], point[cu_seqlens_key]

    def forward(self, point):

        if not self.enable_flash:

            bincount = offset2bincount(point.offset)

            positive_bincount = bincount[bincount > 0]

            min_points = (
                positive_bincount.min().tolist() if positive_bincount.numel() else 1
            )

            self.patch_size = min(int(min_points), self.patch_size_max)

        H = self.num_heads

        K = self.patch_size

        C = self.channels

        pad, unpad, cu_seqlens = self.get_padding_and_inverse(point)

        order = point.serialized_order[self.order_index][pad]

        inverse = unpad[point.serialized_inverse[self.order_index]]

        pseudo_mask = validate_pseudo_mask(
            point.get("pseudo_mask", None),
            int(point.feat.shape[0]),
            name="SerializedAttention.forward",
        )

        if self.separate_qkv:

            qkv_feat = apply_type_aware_tensor_module(
                point.feat, pseudo_mask, self.qkv_real, self.qkv_pseudo
            )

        else:

            qkv_feat = self.qkv(point.feat)

        qkv = qkv_feat[order]

        if not self.enable_flash:

            q, k, v = (
                qkv.reshape(-1, K, 3, H, C // H).permute(2, 0, 3, 1, 4).unbind(dim=0)
            )

            if self.upcast_attention:

                q = q.float()

                k = k.float()

            attn = (q * self.scale) @ k.transpose(-2, -1)

            if self.enable_rpe:

                attn = attn + self.rpe(self.get_rel_pos(point, order))

            if self.upcast_softmax:

                attn = attn.float()

            attn = self.softmax(attn)

            attn = self.attn_drop(attn).to(qkv.dtype)

            feat = (attn @ v).transpose(1, 2).reshape(-1, C)

        else:

            feat = flash_attn.flash_attn_varlen_qkvpacked_func(
                qkv.half().reshape(-1, 3, H, C // H),
                cu_seqlens,
                max_seqlen=self.patch_size,
                dropout_p=self.attn_drop if self.training else 0,
                softmax_scale=self.scale,
            ).reshape(-1, C)

            feat = feat.to(qkv.dtype)

        feat = feat[inverse]

        if self.separate_attn_proj:

            feat = apply_type_aware_tensor_module(
                feat, pseudo_mask, self.proj_real, self.proj_pseudo
            )

        else:

            feat = self.proj(feat)

        feat = self.proj_drop(feat)

        point.feat = feat

        return point


class PointConvCPE(PointModule):

    _BOTTLENECK_RATIO: int = 4

    def __init__(
        self,
        channels: int,
        receptive_field: float,
        max_neighbors: int,
        cache_key: str | None,
        norm_layer=None,
    ):

        super().__init__()

        self.channels = int(channels)

        self.receptive_field = float(receptive_field)

        self.max_neighbors = int(max_neighbors)

        self.cache_key = cache_key

        c_r = max(self.channels // self._BOTTLENECK_RATIO, 16)

        self.c_r = c_r

        self.w_v = nn.Linear(self.channels, c_r)

        self.mlp_w = nn.Sequential(
            nn.Linear(3, c_r),
            nn.GELU(),
            nn.Linear(c_r, c_r),
        )

        self.w_o = nn.Linear(c_r, self.channels)

        self.norm = (
            norm_layer(self.channels) if norm_layer is not None else nn.Identity()
        )

    def _get_or_build_graph(self, point):

        graph_cache_attr = (
            f"_pointconv_graph_{self.cache_key}" if self.cache_key else None
        )

        if graph_cache_attr is not None and graph_cache_attr in point.keys():

            return point[graph_cache_attr]

        edge_index = _radius_graph_eager(
            coord=point.coord,
            receptive_field=self.receptive_field,
            batch=point.batch,
            max_neighbors=self.max_neighbors,
        )

        if graph_cache_attr is not None:

            point[graph_cache_attr] = edge_index

        return edge_index

    def _compute_delta(
        self,
        feat: torch.Tensor,
        coord: torch.Tensor,
        batch: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:

        del batch

        idx_i = edge_index[0]

        idx_j = edge_index[1]

        rel_coord = (coord[idx_j] - coord[idx_i]) / self.receptive_field

        w_ij = self.mlp_w(rel_coord)

        v_j = self.w_v(feat[idx_j])

        m_ij = w_ij * v_j

        agg_i = torch_scatter.scatter_mean(m_ij, idx_i, dim=0, dim_size=feat.shape[0])

        delta_i = self.w_o(agg_i)

        delta_i = self.norm(delta_i)

        return delta_i

    def forward_subset(
        self,
        point: Point,
        keep_mask: torch.Tensor,
        *,
        type_name: str,
    ) -> torch.Tensor:

        if type_name not in {"real", "pseudo"}:

            raise ValueError(f"type_name must be 'real' or 'pseudo', got {type_name}")

        if (
            keep_mask.dtype != torch.bool
            or keep_mask.ndim != 1
            or int(keep_mask.shape[0]) != int(point.feat.shape[0])
        ):

            raise RuntimeError(
                "PointConvCPE.forward_subset: keep_mask must be a 1D bool tensor with the same length as point.feat."
            )

        keep_count = int(keep_mask.sum().item())

        if keep_count == 0:

            return point.feat.new_empty((0, self.channels))

        graph_cache_attr = (
            f"_pointconv_graph_{self.cache_key}_{type_name}" if self.cache_key else None
        )

        if graph_cache_attr is not None and graph_cache_attr in point.keys():

            edge_index = point[graph_cache_attr]

        else:

            edge_index = _radius_graph_eager(
                coord=point.coord[keep_mask],
                receptive_field=self.receptive_field,
                batch=point.batch[keep_mask],
                max_neighbors=self.max_neighbors,
            )

            if graph_cache_attr is not None:

                point[graph_cache_attr] = edge_index

        return self._compute_delta(
            feat=point.feat[keep_mask],
            coord=point.coord[keep_mask],
            batch=point.batch[keep_mask],
            edge_index=edge_index,
        )

    def forward(self, point):

        edge_index = self._get_or_build_graph(point)

        point.feat = self._compute_delta(
            feat=point.feat,
            coord=point.coord,
            batch=point.batch,
            edge_index=edge_index,
        )

        return point


class Block(PointModule):

    def __init__(
        self,
        channels,
        num_heads,
        patch_size=48,
        cpe_kernel_size=5,
        cpe_impl="pointconv",
        cpe_receptive_field=2.0,
        pointconv_block_max_neighbors=16,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        pre_norm=True,
        order_index=0,
        cpe_indice_key=None,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=True,
        upcast_softmax=True,
        ffn_type="mlp",
        separate_qkv=False,
        separate_attn_proj=False,
        separate_ffn=False,
        separate_cpe=False,
    ):

        super().__init__()

        self.channels = channels

        self.cpe_impl = str(cpe_impl)

        self.cpe_kernel_size = int(cpe_kernel_size)

        self.pre_norm = pre_norm

        self.separate_cpe = bool(separate_cpe)

        self.separate_ffn = bool(separate_ffn)

        if self.cpe_impl == "none":

            self.cpe = None

            self.cpe_real = None

            self.cpe_pseudo = None

        elif self.cpe_impl == "pointconv":

            if self.separate_cpe:

                self.cpe = None

                self.cpe_real = PointConvCPE(
                    channels=channels,
                    receptive_field=float(cpe_receptive_field),
                    max_neighbors=int(pointconv_block_max_neighbors),
                    cache_key=cpe_indice_key,
                    norm_layer=norm_layer,
                )

                self.cpe_pseudo = PointConvCPE(
                    channels=channels,
                    receptive_field=float(cpe_receptive_field),
                    max_neighbors=int(pointconv_block_max_neighbors),
                    cache_key=cpe_indice_key,
                    norm_layer=norm_layer,
                )

            else:

                self.cpe = PointConvCPE(
                    channels=channels,
                    receptive_field=float(cpe_receptive_field),
                    max_neighbors=int(pointconv_block_max_neighbors),
                    cache_key=cpe_indice_key,
                    norm_layer=norm_layer,
                )

                self.cpe_real = None

                self.cpe_pseudo = None

        else:

            raise ValueError(
                f"Block: cpe_impl must be 'pointconv' or 'none', got {self.cpe_impl!r}"
            )

        self.norm1 = PointSequential(norm_layer(channels))

        self.attn = SerializedAttention(
            channels=channels,
            patch_size=patch_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            order_index=order_index,
            enable_rpe=enable_rpe,
            enable_flash=enable_flash,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
            separate_qkv=separate_qkv,
            separate_attn_proj=separate_attn_proj,
        )

        self.ffn_type = str(ffn_type).lower()

        if self.ffn_type == "none":

            self.norm2 = None

            self.mlp = None

            self.mlp_real = None

            self.mlp_pseudo = None

        elif self.ffn_type == "gated":

            self.norm2 = PointSequential(norm_layer(channels))

            if self.separate_ffn:

                self.mlp_real = GatedTransition(
                    in_channels=channels,
                    mlp_ratio=int(mlp_ratio),
                    act_layer=act_layer,
                    drop=proj_drop,
                )

                self.mlp_pseudo = GatedTransition(
                    in_channels=channels,
                    mlp_ratio=int(mlp_ratio),
                    act_layer=act_layer,
                    drop=proj_drop,
                )

                self.mlp = None

            else:

                self.mlp = PointSequential(
                    GatedTransition(
                        in_channels=channels,
                        mlp_ratio=int(mlp_ratio),
                        act_layer=act_layer,
                        drop=proj_drop,
                    )
                )

                self.mlp_real = None

                self.mlp_pseudo = None

        else:

            self.norm2 = PointSequential(norm_layer(channels))

            if self.separate_ffn:

                self.mlp_real = MLP(
                    in_channels=channels,
                    hidden_channels=int(channels * mlp_ratio),
                    out_channels=channels,
                    act_layer=act_layer,
                    drop=proj_drop,
                )

                self.mlp_pseudo = MLP(
                    in_channels=channels,
                    hidden_channels=int(channels * mlp_ratio),
                    out_channels=channels,
                    act_layer=act_layer,
                    drop=proj_drop,
                )

                self.mlp = None

            else:

                self.mlp = PointSequential(
                    MLP(
                        in_channels=channels,
                        hidden_channels=int(channels * mlp_ratio),
                        out_channels=channels,
                        act_layer=act_layer,
                        drop=proj_drop,
                    )
                )

                self.mlp_real = None

                self.mlp_pseudo = None

        self.drop_path = PointSequential(
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )

    def _run_cpe(
        self, point: Point, pseudo_mask: torch.Tensor | None
    ) -> torch.Tensor | None:

        if self.cpe is not None:

            input_feat = point.feat

            cpe_point = self.cpe(point)

            cpe_delta = cpe_point.feat

            point.feat = input_feat

            return cpe_delta

        if self.cpe_real is None or self.cpe_pseudo is None:

            return None

        pseudo_mask, has_real, has_pseudo = split_mask_state(
            pseudo_mask,
            int(point.feat.shape[0]),
            name="Block._run_cpe",
        )

        if pseudo_mask is None or not has_pseudo:

            return self.cpe_real.forward_subset(
                point,
                torch.ones(
                    int(point.feat.shape[0]), dtype=torch.bool, device=point.feat.device
                ),
                type_name="real",
            )

        if not has_real:

            return self.cpe_pseudo.forward_subset(
                point, pseudo_mask, type_name="pseudo"
            )

        real_delta = self.cpe_real.forward_subset(point, ~pseudo_mask, type_name="real")

        pseudo_delta = self.cpe_pseudo.forward_subset(
            point, pseudo_mask, type_name="pseudo"
        )

        return merge_type_aware_tensor_outputs(real_delta, pseudo_delta, pseudo_mask)

    def _run_ffn(
        self, feat: torch.Tensor, pseudo_mask: torch.Tensor | None
    ) -> torch.Tensor:

        if self.mlp is not None:

            return self.mlp(feat)

        return apply_type_aware_tensor_module(
            feat, pseudo_mask, self.mlp_real, self.mlp_pseudo
        )

    def forward(self, point: Point):

        cpe_shortcut = point.feat

        cpe_delta = self._run_cpe(
            point,
            validate_pseudo_mask(
                point.get("pseudo_mask", None),
                int(point.feat.shape[0]),
                name="Block.forward",
            ),
        )

        if cpe_delta is not None:

            point.feat = cpe_shortcut + cpe_delta

        shortcut = point.feat

        if self.pre_norm:

            point = self.norm1(point)

        point = self.drop_path(self.attn(point))

        point.feat = shortcut + point.feat

        if not self.pre_norm:

            point = self.norm1(point)

        if self.ffn_type != "none":

            shortcut = point.feat

            if self.pre_norm:

                point = self.norm2(point)

            ffn_feat = self._run_ffn(
                point.feat,
                validate_pseudo_mask(
                    point.get("pseudo_mask", None),
                    int(point.feat.shape[0]),
                    name="Block.forward",
                ),
            )

            point.feat = ffn_feat

            point = self.drop_path(point)

            point.feat = shortcut + point.feat

            if not self.pre_norm:

                point = self.norm2(point)

        return point


class SerializedPooling(PointModule):

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=2,
        norm_layer=None,
        act_layer=None,
        reduce="max",
        shuffle_orders=True,
        traceable=True,
        separate_pooling_proj=False,
    ):

        super().__init__()

        assert stride == 2 ** (math.ceil(stride) - 1).bit_length()

        assert reduce in ["sum", "mean", "min", "max"]

        self.in_channels = in_channels

        self.out_channels = out_channels

        self.stride = stride

        self.reduce = reduce

        self.shuffle_orders = shuffle_orders

        self.traceable = traceable

        self.separate_pooling_proj = bool(separate_pooling_proj)

        self.norm = None

        self.act = None

        self.pool_post_real = None

        self.pool_post_pseudo = None

        if self.separate_pooling_proj:

            self.pool_proj_real = nn.Linear(in_channels, out_channels)

            self.pool_proj_pseudo = nn.Linear(in_channels, out_channels)

            real_post_modules: list[nn.Module] = [nn.LayerNorm(out_channels)]

            pseudo_post_modules: list[nn.Module] = [nn.LayerNorm(out_channels)]

            if act_layer is not None:

                real_post_modules.append(act_layer())

                pseudo_post_modules.append(act_layer())

            self.pool_post_real = nn.Sequential(*real_post_modules)

            self.pool_post_pseudo = nn.Sequential(*pseudo_post_modules)

        else:

            self.proj = nn.Linear(in_channels, out_channels)

            if norm_layer is not None:

                self.norm = PointSequential(norm_layer(out_channels))

            if act_layer is not None:

                self.act = PointSequential(act_layer())

    def forward(self, point: Point):

        pooling_depth = (math.ceil(self.stride) - 1).bit_length()

        if pooling_depth > point.serialized_depth:

            pooling_depth = 0

        assert {
            "serialized_code",
            "serialized_order",
            "serialized_inverse",
            "serialized_depth",
        }.issubset(
            point.keys()
        ), "Run point.serialization() point cloud before SerializedPooling"

        pseudo_mask = validate_pseudo_mask(
            point.get("pseudo_mask", None),
            int(point.feat.shape[0]),
            name="SerializedPooling.forward",
        )

        spatial_code = point.serialized_code >> pooling_depth * 3

        spatial_code_primary = spatial_code[0]

        if pseudo_mask is None:

            cluster_key = spatial_code_primary

        else:

            if bool((spatial_code_primary < 0).any().item()) or bool(
                (spatial_code_primary >= (1 << 62)).any().item()
            ):

                raise RuntimeError(
                    "SerializedPooling.forward: typed pooling spatial_code_primary must be in [0, 1 << 62)."
                )

            cluster_key = spatial_code_primary * 2 + pseudo_mask.to(
                spatial_code_primary.dtype
            )

        _cluster_key_unique, cluster, counts = torch.unique(
            cluster_key,
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )

        _, indices = torch.sort(cluster)

        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])

        head_indices = indices[idx_ptr[:-1]]

        code = spatial_code[:, head_indices]

        tie_breaker = head_indices.to(dtype=code.dtype)

        secondary_order = torch.argsort(tie_breaker, stable=True)

        order = torch.stack(
            [
                secondary_order[torch.argsort(code[i][secondary_order], stable=True)]
                for i in range(code.shape[0])
            ]
        )

        inverse = torch.zeros_like(order).scatter_(
            dim=1,
            index=order,
            src=torch.arange(0, code.shape[1], device=order.device).repeat(
                code.shape[0], 1
            ),
        )

        if self.shuffle_orders:

            perm = torch.randperm(code.shape[0])

            code = code[perm]

            order = order[perm]

            inverse = inverse[perm]

        if self.separate_pooling_proj:

            projected_feat = apply_type_aware_tensor_module(
                point.feat,
                pseudo_mask,
                self.pool_proj_real,
                self.pool_proj_pseudo,
            )

        else:

            projected_feat = self.proj(point.feat)

        pooled_feat = torch_scatter.segment_csr(
            projected_feat[indices], idx_ptr, reduce=self.reduce
        )

        if self.separate_pooling_proj:

            pooled_mask = pseudo_mask[head_indices] if pseudo_mask is not None else None

            pooled_feat = apply_type_aware_tensor_module(
                pooled_feat,
                pooled_mask,
                self.pool_post_real,
                self.pool_post_pseudo,
            )

        point_dict = Dict(
            feat=pooled_feat,
            coord=torch_scatter.segment_csr(
                point.coord[indices], idx_ptr, reduce="mean"
            ),
            grid_coord=point.grid_coord[head_indices] >> pooling_depth,
            serialized_code=code,
            serialized_order=order,
            serialized_inverse=inverse,
            serialized_depth=point.serialized_depth - pooling_depth,
            batch=point.batch[head_indices],
        )

        if "condition" in point.keys():

            point_dict["condition"] = point.condition

        if "context" in point.keys():

            point_dict["context"] = point.context

        if pseudo_mask is not None:

            point_dict["pseudo_mask"] = pseudo_mask[head_indices]

        if self.traceable:

            point_dict["pooling_inverse"] = cluster

            point_dict["pooling_parent"] = point

        point = Point(point_dict)

        if self.norm is not None:

            point = self.norm(point)

        if self.act is not None:

            point = self.act(point)

        return point


class SerializedUnpooling(PointModule):

    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        norm_layer=None,
        act_layer=None,
        traceable=False,
        separate_unpooling_proj=False,
    ):

        super().__init__()

        self.separate_unpooling_proj = bool(separate_unpooling_proj)

        if self.separate_unpooling_proj:

            self.proj_real = make_typed_linear_norm_act(
                in_channels, out_channels, act_layer
            )

            self.proj_pseudo = make_typed_linear_norm_act(
                in_channels, out_channels, act_layer
            )

            self.proj_skip_real = make_typed_linear_norm_act(
                skip_channels, out_channels, act_layer
            )

            self.proj_skip_pseudo = make_typed_linear_norm_act(
                skip_channels, out_channels, act_layer
            )

            self.proj = None

            self.proj_skip = None

        else:

            self.proj = PointSequential(nn.Linear(in_channels, out_channels))

            self.proj_skip = PointSequential(nn.Linear(skip_channels, out_channels))

            if norm_layer is not None:

                self.proj.add(norm_layer(out_channels))

                self.proj_skip.add(norm_layer(out_channels))

            if act_layer is not None:

                self.proj.add(act_layer())

                self.proj_skip.add(act_layer())

        self.traceable = traceable

    def forward(self, point):

        assert "pooling_parent" in point.keys()

        assert "pooling_inverse" in point.keys()

        child_mask = validate_pseudo_mask(
            point.get("pseudo_mask", None),
            int(point.feat.shape[0]),
            name="SerializedUnpooling.forward.child",
        )

        parent = point.pop("pooling_parent")

        inverse = point.pop("pooling_inverse")

        parent_mask = validate_pseudo_mask(
            parent.get("pseudo_mask", None),
            int(parent.feat.shape[0]),
            name="SerializedUnpooling.forward.parent",
        )

        if self.separate_unpooling_proj:

            point.feat = apply_type_aware_tensor_module(
                point.feat, child_mask, self.proj_real, self.proj_pseudo
            )

            parent.feat = apply_type_aware_tensor_module(
                parent.feat,
                parent_mask,
                self.proj_skip_real,
                self.proj_skip_pseudo,
            )

        else:

            point = self.proj(point)

            parent = self.proj_skip(parent)

        parent.feat = parent.feat + point.feat[inverse]

        if self.traceable:

            parent["unpooling_parent"] = point

        return parent


class PointConvEmbedding(PointModule):

    def __init__(
        self,
        in_channels: int,
        embed_channels: int,
        receptive_field: float,
        max_neighbors: int,
        norm_layer=None,
        act_layer=None,
    ):

        super().__init__()

        self.in_channels = int(in_channels)

        self.embed_channels = int(embed_channels)

        self.receptive_field = float(receptive_field)

        self.max_neighbors = int(max_neighbors)

        self.w_v = nn.Linear(self.in_channels, self.embed_channels)

        self.self_proj = nn.Linear(self.in_channels, self.embed_channels)

        self.mlp_w = nn.Sequential(
            nn.Linear(3, self.embed_channels),
            nn.GELU(),
            nn.Linear(self.embed_channels, self.embed_channels),
        )

        self.norm = norm_layer(self.embed_channels) if norm_layer is not None else None

        self.act = act_layer() if act_layer is not None else None

    def _forward_tensors(
        self,
        feat: torch.Tensor,
        coord: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:

        point_count = int(feat.shape[0])

        if point_count == 0:

            return feat.new_empty((0, self.embed_channels))

        edge_index = _radius_graph_eager(
            coord=coord,
            receptive_field=self.receptive_field,
            batch=batch,
            max_neighbors=self.max_neighbors,
        )

        idx_i = edge_index[0]

        idx_j = edge_index[1]

        rel_coord = (coord[idx_j] - coord[idx_i]) / self.receptive_field

        w_ij = self.mlp_w(rel_coord)

        v_j = self.w_v(feat[idx_j])

        m_ij = w_ij * v_j

        agg_i = torch_scatter.scatter_mean(m_ij, idx_i, dim=0, dim_size=point_count)

        embed_feat = agg_i + self.self_proj(feat)

        if self.norm is not None:

            embed_feat = self.norm(embed_feat)

        if self.act is not None:

            embed_feat = self.act(embed_feat)

        return embed_feat

    def forward_subset(self, point: Point, keep_mask: torch.Tensor) -> torch.Tensor:

        if (
            keep_mask.dtype != torch.bool
            or keep_mask.ndim != 1
            or int(keep_mask.shape[0]) != int(point.feat.shape[0])
        ):

            raise RuntimeError(
                "PointConvEmbedding.forward_subset: keep_mask must be a 1D bool tensor with the same length as point.feat."
            )

        return self._forward_tensors(
            feat=point.feat[keep_mask],
            coord=point.coord[keep_mask],
            batch=point.batch[keep_mask],
        )

    def forward(self, point):

        point.feat = self._forward_tensors(point.feat, point.coord, point.batch)

        return point


class Embedding(PointModule):

    def __init__(
        self,
        in_channels,
        embed_channels,
        embedding_kernel_size=7,
        embedding_impl="pointconv",
        embedding_receptive_field=5.0,
        pointconv_embed_max_neighbors=32,
        norm_layer=None,
        act_layer=None,
        separate_embedding=False,
    ):

        super().__init__()

        self.in_channels = in_channels

        self.embed_channels = embed_channels

        self.embedding_impl = str(embedding_impl)

        self.embedding_kernel_size = int(embedding_kernel_size)

        self.separate_embedding = bool(separate_embedding)

        if self.embedding_impl != "pointconv":

            raise ValueError(
                f"Embedding: embedding_impl only supports 'pointconv', got {self.embedding_impl!r}"
            )

        if self.separate_embedding:

            self._pointconv_embed_real = PointConvEmbedding(
                in_channels=in_channels,
                embed_channels=embed_channels,
                receptive_field=float(embedding_receptive_field),
                max_neighbors=int(pointconv_embed_max_neighbors),
                norm_layer=nn.LayerNorm,
                act_layer=act_layer,
            )

            self._pointconv_embed_pseudo = PointConvEmbedding(
                in_channels=in_channels,
                embed_channels=embed_channels,
                receptive_field=float(embedding_receptive_field),
                max_neighbors=int(pointconv_embed_max_neighbors),
                norm_layer=nn.LayerNorm,
                act_layer=act_layer,
            )

            self._pointconv_embed = None

        else:

            self._pointconv_embed = PointConvEmbedding(
                in_channels=in_channels,
                embed_channels=embed_channels,
                receptive_field=float(embedding_receptive_field),
                max_neighbors=int(pointconv_embed_max_neighbors),
                norm_layer=norm_layer,
                act_layer=act_layer,
            )

            self._pointconv_embed_real = None

            self._pointconv_embed_pseudo = None

    def forward(self, point: Point):

        if not self.separate_embedding:

            point = self._pointconv_embed(point)

            return point

        pseudo_mask, has_real, has_pseudo = split_mask_state(
            point.get("pseudo_mask", None),
            int(point.feat.shape[0]),
            name="Embedding.forward",
        )

        if pseudo_mask is None or not has_pseudo:

            point.feat = self._pointconv_embed_real.forward_subset(
                point,
                torch.ones(
                    int(point.feat.shape[0]), dtype=torch.bool, device=point.feat.device
                ),
            )

            return point

        if not has_real:

            point.feat = self._pointconv_embed_pseudo.forward_subset(point, pseudo_mask)

            return point

        real_embed_feat = self._pointconv_embed_real.forward_subset(point, ~pseudo_mask)

        pseudo_embed_feat = self._pointconv_embed_pseudo.forward_subset(
            point, pseudo_mask
        )

        point.feat = merge_type_aware_tensor_outputs(
            real_embed_feat, pseudo_embed_feat, pseudo_mask
        )

        return point


class PointTransformerV3(PointModule):

    def __init__(
        self,
        in_channels=49,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(4, 2, 2, 2),
        embedding_kernel_size=7,
        embedding_impl="pointconv",
        cpe_impl="pointconv",
        embedding_receptive_field=5.0,
        pointconv_embed_max_neighbors=64,
        pointconv_block_max_neighbors=32,
        enc_cpe_kernel_size=(5, 5, 5, 5, 5),
        dec_cpe_kernel_size=(5, 5, 5, 5, 5),
        enc_cpe_receptive_field=(2.0, 4.0, 8.0, 12.0, 16.0),
        dec_cpe_receptive_field=(2.0, 4.0, 8.0, 12.0, 16.0),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(64, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(256, 128, 96, 72, 64),
        dec_depths=(2, 2, 2, 2, 0),
        dec_channels=(64, 64, 128, 256, 512),
        dec_num_head=(4, 4, 8, 16, 32),
        dec_patch_size=(128, 96, 72, 64, 64),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
        act_layer_name="gelu",
        ffn_type="mlp",
        separate_qkv=False,
        separate_attn_proj=False,
        separate_ffn=False,
        separate_cpe=False,
        separate_embedding=False,
        separate_pooling_proj=False,
        separate_unpooling_proj=False,
    ):

        super().__init__()

        self.num_stages = len(enc_depths)

        self.order = [order] if isinstance(order, str) else order

        self.cls_mode = cls_mode

        self.shuffle_orders = shuffle_orders

        self.separate_qkv = bool(separate_qkv)

        self.separate_attn_proj = bool(separate_attn_proj)

        self.separate_ffn = bool(separate_ffn)

        self.separate_cpe = bool(separate_cpe)

        self.separate_embedding = bool(separate_embedding)

        self.separate_pooling_proj = bool(separate_pooling_proj)

        self.separate_unpooling_proj = bool(separate_unpooling_proj)

        self.embedding_impl = str(embedding_impl)

        self.cpe_impl = str(cpe_impl)

        assert self.num_stages == len(stride) + 1

        assert self.num_stages == len(enc_depths)

        assert self.num_stages == len(enc_channels)

        assert self.num_stages == len(enc_num_head)

        assert self.num_stages == len(enc_patch_size)

        assert self.num_stages == len(
            enc_cpe_kernel_size
        ), f"enc_cpe_kernel_size length ({len(enc_cpe_kernel_size)}) must equal num_stages ({self.num_stages})"

        assert self.num_stages == len(
            enc_cpe_receptive_field
        ), f"enc_cpe_receptive_field length ({len(enc_cpe_receptive_field)}) must equal num_stages ({self.num_stages})"

        assert self.cls_mode or self.num_stages == len(dec_depths)

        assert self.cls_mode or self.num_stages == len(dec_channels)

        assert self.cls_mode or self.num_stages == len(dec_num_head)

        assert self.cls_mode or self.num_stages == len(dec_patch_size)

        assert self.cls_mode or self.num_stages == len(
            dec_cpe_kernel_size
        ), f"dec_cpe_kernel_size length ({len(dec_cpe_kernel_size)}) must equal num_stages ({self.num_stages})"

        assert self.cls_mode or self.num_stages == len(
            dec_cpe_receptive_field
        ), f"dec_cpe_receptive_field length ({len(dec_cpe_receptive_field)}) must equal num_stages ({self.num_stages})"

        if self.embedding_impl != "pointconv":

            raise ValueError(
                f"PointTransformerV3: embedding_impl only supports 'pointconv', got {self.embedding_impl!r}"
            )

        if self.cpe_impl not in {"pointconv", "none"}:

            raise ValueError(
                f"PointTransformerV3: cpe_impl must be 'pointconv' or 'none', got {self.cpe_impl!r}"
            )

        if self.cpe_impl == "pointconv":

            for s, rf in enumerate(enc_cpe_receptive_field):

                assert (
                    float(rf) > 0
                ), f"in pointconv mode enc_cpe_receptive_field[{s}]={rf} must be positive"

            for s, rf in enumerate(dec_cpe_receptive_field):

                assert (
                    float(rf) > 0
                ), f"in pointconv mode dec_cpe_receptive_field[{s}]={rf} must be positive"

        if pdnorm_bn:

            bn_layer = partial(
                PDNorm,
                norm_layer=partial(
                    nn.BatchNorm1d, eps=1e-3, momentum=0.01, affine=pdnorm_affine
                ),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
            )

        else:

            bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        if pdnorm_ln:

            ln_layer = partial(
                PDNorm,
                norm_layer=partial(nn.LayerNorm, elementwise_affine=pdnorm_affine),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
            )

        else:

            ln_layer = nn.LayerNorm

        act_layer = resolve_act_layer(act_layer_name)

        self.embedding = Embedding(
            in_channels=in_channels,
            embed_channels=enc_channels[0],
            embedding_kernel_size=int(embedding_kernel_size),
            embedding_impl=self.embedding_impl,
            embedding_receptive_field=float(embedding_receptive_field),
            pointconv_embed_max_neighbors=int(pointconv_embed_max_neighbors),
            norm_layer=bn_layer,
            act_layer=act_layer,
            separate_embedding=self.separate_embedding,
        )

        enc_drop_path = [
            x.item()
            for x in torch.linspace(start=0, end=drop_path, steps=sum(enc_depths))
        ]

        self.enc = PointSequential()

        for s in range(self.num_stages):

            enc_drop_path_ = enc_drop_path[
                sum(enc_depths[:s]) : sum(enc_depths[: s + 1])
            ]

            enc = PointSequential()

            if s > 0:

                enc.add(
                    SerializedPooling(
                        in_channels=enc_channels[s - 1],
                        out_channels=enc_channels[s],
                        stride=stride[s - 1],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                        separate_pooling_proj=self.separate_pooling_proj,
                    ),
                    name="down",
                )

            for i in range(enc_depths[s]):

                enc.add(
                    Block(
                        channels=enc_channels[s],
                        num_heads=enc_num_head[s],
                        patch_size=enc_patch_size[s],
                        cpe_kernel_size=int(enc_cpe_kernel_size[s]),
                        cpe_impl=self.cpe_impl,
                        cpe_receptive_field=float(enc_cpe_receptive_field[s]),
                        pointconv_block_max_neighbors=int(
                            pointconv_block_max_neighbors
                        ),
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=enc_drop_path_[i],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        order_index=i % len(self.order),
                        cpe_indice_key=f"stage{s}",
                        enable_rpe=enable_rpe,
                        enable_flash=enable_flash,
                        upcast_attention=upcast_attention,
                        upcast_softmax=upcast_softmax,
                        ffn_type=ffn_type,
                        separate_qkv=self.separate_qkv,
                        separate_attn_proj=self.separate_attn_proj,
                        separate_ffn=self.separate_ffn,
                        separate_cpe=self.separate_cpe,
                    ),
                    name=f"block{i}",
                )

            if len(enc) != 0:

                self.enc.add(module=enc, name=f"enc{s}")

        if not self.cls_mode:

            dec_drop_path = [
                x.item() for x in torch.linspace(0, drop_path, sum(dec_depths))
            ]

            self.dec = PointSequential()

            dec_channels = list(dec_channels)

            for s in reversed(range(self.num_stages)):

                if int(dec_depths[s]) == 0:

                    self.dec.add(module=nn.Identity(), name=f"dec{s}")

                    continue

                dec_drop_path_ = dec_drop_path[
                    sum(dec_depths[:s]) : sum(dec_depths[: s + 1])
                ]

                dec_drop_path_.reverse()

                dec = PointSequential()

                if s == self.num_stages - 1:

                    if enc_channels[-1] != dec_channels[s]:

                        dec.add(
                            nn.Linear(enc_channels[-1], dec_channels[s]), name="proj"
                        )

                        dec.add(ln_layer(dec_channels[s]), name="proj_norm")

                else:

                    up_in_channels = dec_channels[s + 1]

                    dec.add(
                        SerializedUnpooling(
                            in_channels=up_in_channels,
                            skip_channels=enc_channels[s],
                            out_channels=dec_channels[s],
                            norm_layer=bn_layer,
                            act_layer=act_layer,
                            separate_unpooling_proj=self.separate_unpooling_proj,
                        ),
                        name="up",
                    )

                for i in range(dec_depths[s]):

                    dec.add(
                        Block(
                            channels=dec_channels[s],
                            num_heads=dec_num_head[s],
                            patch_size=dec_patch_size[s],
                            cpe_kernel_size=int(dec_cpe_kernel_size[s]),
                            cpe_impl=self.cpe_impl,
                            cpe_receptive_field=float(dec_cpe_receptive_field[s]),
                            pointconv_block_max_neighbors=int(
                                pointconv_block_max_neighbors
                            ),
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            attn_drop=attn_drop,
                            proj_drop=proj_drop,
                            drop_path=dec_drop_path_[i],
                            norm_layer=ln_layer,
                            act_layer=act_layer,
                            pre_norm=pre_norm,
                            order_index=i % len(self.order),
                            cpe_indice_key=f"stage{s}",
                            enable_rpe=enable_rpe,
                            enable_flash=enable_flash,
                            upcast_attention=upcast_attention,
                            upcast_softmax=upcast_softmax,
                            ffn_type=ffn_type,
                            separate_qkv=self.separate_qkv,
                            separate_attn_proj=self.separate_attn_proj,
                            separate_ffn=self.separate_ffn,
                            separate_cpe=self.separate_cpe,
                        ),
                        name=f"block{i}",
                    )

                self.dec.add(module=dec, name=f"dec{s}")

    def forward(self, data_dict):

        point = Point(data_dict)

        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)

        point = self.embedding(point)

        point = self.enc(point)

        if not self.cls_mode:

            point = self.dec(point)

        return point
