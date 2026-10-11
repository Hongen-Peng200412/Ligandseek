"""Build the original density channel bank from each experimental/simulated window."""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from dataclasses import dataclass, field
from typing import Optional



@dataclass
class DensityChannelConfig:
\
\
\
\
\
\
\

    clip_percentile: tuple[float, float] = (0.001, 0.999)
    fit_mask_percentile: float = 0.003
    enabled_channels: list[str] = field(default_factory=lambda: ["all"])





def _fit_scale_params(
    exp: np.ndarray,
    sim: np.ndarray,
    percentile: float,
    receptor_mask: np.ndarray | None,
) -> tuple[float, float]:
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


    n_total = int(exp.size)

    min_voxels = max(int(n_total * percentile), 10)



    density_pct = min(5.0 * percentile, 1.0)

    t_exp = float(np.percentile(exp, (1.0 - density_pct) * 100))

    t_sim = float(np.percentile(sim, (1.0 - density_pct) * 100))

    density_filter = (exp >= t_exp) & (sim >= t_sim)

    if receptor_mask is not None:

        priority = np.asarray(receptor_mask, dtype=bool) & density_filter

        n_priority = int(priority.sum())

        if n_priority >= min_voxels:

            mask = priority
        else:

            mask = density_filter
    else:

        mask = density_filter


    n_masked = int(mask.sum())

    if n_masked < 10:

        return 1.0, 0.0


    sim_masked = sim[mask]

    exp_masked = exp[mask]


    if float(sim_masked.var()) < 1e-12:
        return 1.0, 0.0


    X = np.column_stack([sim_masked, np.ones_like(sim_masked)])

    params = np.linalg.lstsq(X, exp_masked, rcond=None)[0]
    return float(params[0]), float(params[1])

def _compute_ops(
    exp: np.ndarray,
    sim: np.ndarray,
    percentile: float,
    receptor_mask: np.ndarray | None,
) -> dict[str, np.ndarray]:
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


    a, b = _fit_scale_params(exp, sim, percentile, receptor_mask)

    diff = exp - (a * sim + b)

    return {
        "exp": exp,
        "sim": sim,
        "diff": diff,
        "posdiff": np.maximum(diff, 0),
    }








def _clip_and_norm(x: np.ndarray, clip_percentile: tuple[float, float]) -> np.ndarray:
\
\
\
\
\
\
\
\
\


    lo = float(np.percentile(x, clip_percentile[0] * 100))

    hi = float(np.percentile(x, clip_percentile[1] * 100))

    x_clipped = np.clip(x, lo, hi)

    mu = float(x_clipped.mean())

    sigma = float(x_clipped.std()) + 1e-8
    return (x_clipped - mu) / sigma

def _apply_norm(
    op_result: np.ndarray,
    norm: str,
    clip_percentile: tuple[float, float],
) -> np.ndarray:
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

    if norm == "nonorm":
        return op_result
    elif norm == "clipnorm":
        return _clip_and_norm(op_result, clip_percentile)
    else:
        raise ValueError(f"Unknown norm: {norm}")








def _gaussian_filter_3d(x: np.ndarray, sigma: float) -> np.ndarray:
\
\
\
\
\
\
\
\
\

    return ndimage.gaussian_filter(x, sigma=sigma, mode="reflect")

def _dog_3d(x: np.ndarray, sigma: float) -> np.ndarray:
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


    m = 1.6
    return ndimage.gaussian_filter(x, sigma=sigma * m, mode="reflect")\
           - ndimage.gaussian_filter(x, sigma=sigma, mode="reflect")


def _build_smooth_strength(
    receptor_mask: np.ndarray,
    sigma: float,
    radius: float,
) -> np.ndarray:
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


    mask = np.asarray(receptor_mask, dtype=bool)

    distance = ndimage.distance_transform_edt(~mask)

    smooth_strength = np.exp(-(distance ** 2) / (2.0 * sigma * sigma))
    smooth_strength[distance > radius] = 0.0
    return smooth_strength.astype(np.float32)



_POST_SIGMA_MAP: dict[str, Optional[float]] = {
    "nopost": None,
    "gauss1": 1.0,
    "gauss2": 2.0,
    "DoG1": 1.0,
    "DoG2": 2.0,
    "smooth1": 1.0,
    "smooth2": 2.0,
}

def _apply_post(x: np.ndarray, post: str) -> np.ndarray:
\
\
\
\
\
\
\
\
\

    if post == "nopost":
        return x
    if post not in _POST_SIGMA_MAP:
        raise ValueError(f"Unknown post: {post}")
    sigma = _POST_SIGMA_MAP[post]
    if sigma is None:
        return x
    if post.startswith("gauss"):
        return _gaussian_filter_3d(x, sigma)
    elif post.startswith("DoG"):
        return _dog_3d(x, sigma)
    elif post.startswith("smooth"):
        raise ValueError("smooth postprocessing requires receptor_mask in build_density_channels")
    else:
        raise ValueError(f"Unknown post: {post}")





















OPS: list[str] = ["exp", "sim", "diff", "posdiff"]

NORMS: list[str] = ["nonorm", "clipnorm"]

POSTS: list[str] = ["nopost", "gauss1", "gauss2", "DoG1", "DoG2", "smooth1", "smooth2"]


ALL_CHANNEL_NAMES: list[str] = [
    f"{op}_{norm}_{post}"
    for op in OPS
    for norm in NORMS
    for post in POSTS
]


_OPS_NEEDING_BOTH: set[str] = {"diff", "posdiff"}


def _parse_channel_name(name: str) -> tuple[str, str, str]:
\
\
\
\
\
\
\
\

    parts = name.split("_")
    if len(parts) < 3:
        raise ValueError(f"Invalid channel name: {name}")


    op = parts[0]
    if op not in OPS:
        raise ValueError(f"Unknown op in channel name '{name}': {op}")


    norm = parts[1]
    if norm not in NORMS:
        raise ValueError(f"Unknown norm in channel name '{name}': {norm}")


    post = "_".join(parts[2:])
    if post not in POSTS:
        raise ValueError(f"Unknown post in channel name '{name}': {post}")

    return op, norm, post
















def build_density_channels(
    exp_raw: Optional[np.ndarray],
    sim_raw: Optional[np.ndarray],
    config: DensityChannelConfig,
    receptor_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
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



    if "all" in config.enabled_channels:
        enabled = list(ALL_CHANNEL_NAMES)
    else:
        enabled = list(config.enabled_channels)
    if not enabled:
        raise ValueError("enabled_channels is empty; no density channels can be generated")



    needed_ops: set[str] = set()
    for name in enabled:
        op, _, _ = _parse_channel_name(name)
        needed_ops.add(op)


    has_exp = exp_raw is not None
    has_sim = sim_raw is not None
    needs_both = bool(needed_ops & _OPS_NEEDING_BOTH)
    if "exp" in needed_ops and not has_exp:
        raise ValueError("exp channels are enabled but exp_raw is missing")
    if "sim" in needed_ops and not has_sim:
        raise ValueError("sim channels are enabled but sim_raw is missing")
    if needs_both and (not has_exp or not has_sim):
        raise ValueError("diff/posdiff channels require both exp_raw and sim_raw")



    all_ops: dict[str, np.ndarray] = {}
    if has_exp and has_sim and needs_both:
        all_ops = _compute_ops(exp_raw, sim_raw, config.fit_mask_percentile, receptor_mask)
    else:
        if has_exp:
            all_ops["exp"] = exp_raw
        if has_sim:
            all_ops["sim"] = sim_raw





    norm_cache: dict[tuple[str, str], np.ndarray] = {}

    gaussian_cache: dict[tuple[str, str, float], np.ndarray] = {}

    smooth_cache: dict[float, np.ndarray] = {}

    def get_normed(op_name: str, norm_name: str) -> np.ndarray:
        cache_key = (op_name, norm_name)
        if cache_key not in norm_cache:
            norm_cache[cache_key] = _apply_norm(all_ops[op_name], norm_name, config.clip_percentile)
        return norm_cache[cache_key]

    def get_gaussian(op_name: str, norm_name: str, sigma: float) -> np.ndarray:
        cache_key = (op_name, norm_name, float(sigma))
        if cache_key not in gaussian_cache:
            gaussian_cache[cache_key] = _gaussian_filter_3d(get_normed(op_name, norm_name), sigma)
        return gaussian_cache[cache_key]

    def get_smooth_strength(sigma: float) -> np.ndarray:
        cache_key = float(sigma)
        if cache_key not in smooth_cache:
            if receptor_mask is None:
                raise ValueError("smooth postprocessing requires receptor_mask")
            smooth_cache[cache_key] = _build_smooth_strength(receptor_mask, cache_key, radius=2.0)
        return smooth_cache[cache_key]


    channels: list[np.ndarray] = []
    for name in enabled:
        op_name, norm_name, post_name = _parse_channel_name(name)
        if post_name == "nopost":
            final = get_normed(op_name, norm_name)
        elif post_name.startswith("gauss"):
            sigma = float(_POST_SIGMA_MAP[post_name])
            final = get_gaussian(op_name, norm_name, sigma)
        elif post_name.startswith("DoG"):
            sigma = float(_POST_SIGMA_MAP[post_name])
            final = get_gaussian(op_name, norm_name, sigma * 1.6) - get_gaussian(op_name, norm_name, sigma)
        elif post_name.startswith("smooth"):
            sigma = float(_POST_SIGMA_MAP[post_name])
            final = get_normed(op_name, norm_name) * (1.0 - get_smooth_strength(sigma))
        else:
            final = _apply_post(get_normed(op_name, norm_name), post_name)
        channels.append(final)

    if not channels:
        raise ValueError(
            "No density channels were generated; check enabled_channels and inputs."
            f" enabled_channels={config.enabled_channels},"
            f" has_exp={has_exp}, has_sim={has_sim}"
        )


    return np.stack(channels, axis=0).astype(np.float32)





def detect_diff_posdiff_indices(enabled_channels: list[str]) -> list[int]:
\
\
\
\
\
\
\
\


    if "all" in enabled_channels:
        resolved = list(ALL_CHANNEL_NAMES)
    else:
        resolved = list(enabled_channels)

    indices: list[int] = []
    for i, name in enumerate(resolved):
        op, _, _ = _parse_channel_name(name)
        if op in ("diff", "posdiff"):
            indices.append(i)
    return indices
