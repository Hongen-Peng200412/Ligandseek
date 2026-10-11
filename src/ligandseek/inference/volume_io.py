"""Read MRC density in ZYX order and reproduce the paper Fourier grid preprocessing."""

import numpy as np
import mrcfile as mrc
from pathlib import Path

def load_map(mrc_fn:str,multiply_global_origin:bool=True):


    mrc_file=mrc.open(mrc_fn,'r')
    voxel_size=mrc_file.voxel_size
    voxel_size=np.array([voxel_size.x,voxel_size.y,voxel_size.z])
    if voxel_size[0] <= 0:
        raise RuntimeError(f"Seems like the MRC file: {mrc_fn} does not have a header.")
    c = mrc_file.header["mapc"]
    r = mrc_file.header["mapr"]
    s = mrc_file.header["maps"]

    global_origin = mrc_file.header["origin"]
    global_origin = np.array([global_origin.x, global_origin.y, global_origin.z])
    nstart = np.array([mrc_file.header["nxstart"],mrc_file.header["nystart"],mrc_file.header["nzstart"]])
    temp1 = [c - 1, r - 1, s - 1]
    temp_start = np.zeros(3)
    for temp_index in range(3):
        temp_start[temp1[temp_index]] = nstart[temp_index]
    global_origin = global_origin + temp_start
    if multiply_global_origin:
        global_origin = global_origin * voxel_size
    if c == 1 and r == 2 and s == 3:
        grid = mrc_file.data.copy()
    elif c == 1 and r == 3 and s == 2:
        grid = np.moveaxis(mrc_file.data.copy(), [2, 0, 1], [2, 1, 0])
    elif c == 3 and r == 2 and s == 1:
        grid = np.moveaxis(mrc_file.data.copy(), [0, 1, 2], [2, 1, 0])
    elif c == 3 and r == 1 and s == 2:
        grid = np.moveaxis(mrc_file.data.copy(), [1, 0, 2], [2, 1, 0])
    elif c == 2 and r == 1 and s == 3:
        grid = np.moveaxis(mrc_file.data.copy(), [1, 2, 0], [2, 1, 0])
    elif c == 2 and r == 3 and s == 1:
        grid = np.moveaxis(mrc_file.data.copy(), [0, 2, 1], [2, 1, 0])
    else:
        raise RuntimeError("MRC file axis arrangement not supported!")

    mrc_file.close()
    return (grid, voxel_size, global_origin)

def make_cubic(box):
    bz = np.array(box.shape)
    s = bz + bz % 2
    if np.all(box.shape == s):
        return box, np.zeros(3, dtype=int), bz
    nbox = np.zeros(tuple(s))
    c = np.array(nbox.shape) // 2 - bz // 2
    nbox[c[0] : c[0] + bz[0], c[1] : c[1] + bz[1], c[2] : c[2] + bz[2]] = box
    return nbox, c, c + bz

def normalize_voxel_size(density, in_voxel_sz, target_voxel_size=1.0,check_point:bool=True):
    iz,iy,ix = np.shape(density)
    in_size = np.array([ix,iy,iz])
    out_size = np.ceil(in_size*in_voxel_sz/target_voxel_size)
    out_size = out_size.astype(int)
    for d_index,h in enumerate(out_size):
        if h % 2 != 0:
            vs_1 = in_voxel_sz[d_index] * in_size[d_index] / (h + 1)
            vs_2 = in_voxel_sz[d_index] * in_size[d_index] / (h - 1)
            if abs(vs_1-target_voxel_size) < abs(vs_2-target_voxel_size):
                h = h+1
            else:
                h = h-1
        out_size[d_index] = h
    out_voxel_sz = in_voxel_sz * in_size / out_size
    jx,jy,jz=out_size
    out_size=np.array([jz,jy,jx]).astype(int)
    if check_point:
        density = rescale_real(density, out_size)
    return density, out_voxel_sz

def rescale_real(box, out_sz):
    assert np.all(np.array(box.shape)%2==0) and np.all(np.array(out_sz)%2==0)
    # Preserve the original all-axis condition used for the paper grids.
    if np.all(out_sz != box.shape):
        f = np.fft.rfftn(box)
        f = rescale_fourier(f, out_sz)
        box = np.fft.irfftn(f)
    return box

def rescale_fourier(box, out_sz):
    temp_size = list(out_sz)
    temp_size[2] = temp_size[2]//2 + 1
    temp_size = tuple(temp_size)
    i_size = box.shape
    c=[]
    pads=[]
    for t_index in range(2):
        if i_size[t_index]>temp_size[t_index]:
            c.append(temp_size[t_index])
            pad = 0
        else:
            c.append(i_size[t_index])
            pad = (temp_size[t_index] - i_size[t_index])//2
        pads.append((pad,pad))
    if i_size[2] > temp_size[2]:
        c.append(temp_size[2])
        pads.append((0,0))
    else:
        c.append(i_size[2])
        pads.append((0,temp_size[2] - i_size[2]))
    pads = tuple(pads)
    ibox = np.fft.fftshift(box, axes=(0, 1))
    si = np.array(ibox.shape) // 2
    so = np.array(c) // 2
    lamb_box = ibox[si[0] - so[0]:si[0] + so[0], si[1] - so[1]:si[1] + so[1], :c[2]]
    obox = np.pad(lamb_box,pads,mode='constant')
    obox = np.fft.ifftshift(obox, axes=(0, 1))
    return obox

def make_model_grid(grid, voxel_size, global_origin, target_voxel_size=1.5):

    grid, shift, _ = make_cubic(grid)
    global_origin = global_origin - shift * voxel_size

    grid, voxel_size = normalize_voxel_size(
        grid, voxel_size, target_voxel_size=target_voxel_size
    )
    return grid, voxel_size, global_origin


def load_mrc_volume(path: str | Path, target_voxel_size: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the paper grid (float32 ZYX), actual spacing (XYZ), and historical origin (XYZ)."""
    grid, spacing, origin = load_map(str(path))
    grid, spacing, origin = make_model_grid(grid, spacing, origin, target_voxel_size)
    return np.asarray(grid, dtype=np.float32), np.asarray(spacing, dtype=np.float32), np.asarray(origin, dtype=np.float32)


def save_probability_npz(path: str | Path, ligand_probability: np.ndarray, voxel_size: np.ndarray | None = None, origin: np.ndarray | None = None) -> None:
    payload = {"ligand_probability": np.asarray(ligand_probability, dtype=np.float32)}
    if voxel_size is not None:
        payload["voxel_size"] = np.asarray(voxel_size, dtype=np.float32)
    if origin is not None:
        payload["origin"] = np.asarray(origin, dtype=np.float32)
    np.savez_compressed(path, **payload)
