"""Build receptor atom home-voxel masks in the paper XYZ-to-ZYX convention."""

import numpy as np

def build_hardmask_from_atom_coordinates(
    atom_coord_local_voxel: np.ndarray,
    atom_is_in_core_box: np.ndarray,
    box_shape_zyx: np.ndarray,
) -> np.ndarray:


    depth, height, width = [int(v) for v in np.asarray(box_shape_zyx, dtype=np.int64).tolist()]
    
    hardmask = np.zeros((depth, height, width), dtype=np.int64)


    core_mask = np.asarray(atom_is_in_core_box, dtype=bool)
    if atom_coord_local_voxel.shape[0] == 0 or not np.any(core_mask):
        return hardmask


    core_coord_local_voxel = np.asarray(atom_coord_local_voxel[core_mask], dtype=np.float32)
    
    voxel_idx_xyz = np.floor(core_coord_local_voxel).astype(np.int64)
    
    box_shape_xyz = np.asarray([width, height, depth], dtype=np.int64)


    valid_mask = np.all(voxel_idx_xyz >= 0, axis=1)
    valid_mask &= voxel_idx_xyz[:, 0] < box_shape_xyz[0]
    valid_mask &= voxel_idx_xyz[:, 1] < box_shape_xyz[1]
    valid_mask &= voxel_idx_xyz[:, 2] < box_shape_xyz[2]
    if not np.any(valid_mask):
        return hardmask


    voxel_idx_xyz = voxel_idx_xyz[valid_mask]
    hardmask[voxel_idx_xyz[:, 2], voxel_idx_xyz[:, 1], voxel_idx_xyz[:, 0]] = 1
    return hardmask

def build_hardmask_from_world_coordinates(
    atom_coords_world: np.ndarray,
    box_origin_world: np.ndarray,
    voxel_size_world: np.ndarray,
    box_shape_zyx: np.ndarray,
) -> np.ndarray:


    atom_coord_local_voxel = (
        (np.asarray(atom_coords_world, dtype=np.float32) - np.asarray(box_origin_world, dtype=np.float32)[None, :])
        / np.asarray(voxel_size_world, dtype=np.float32)[None, :]
    ).astype(np.float32, copy=False)


    box_shape_xyz = np.asarray(box_shape_zyx, dtype=np.float32)[[2, 1, 0]]
    
    atom_is_in_core_box = np.all(atom_coord_local_voxel >= 0.0, axis=1)
    atom_is_in_core_box &= np.all(atom_coord_local_voxel < box_shape_xyz[None, :], axis=1)

    return build_hardmask_from_atom_coordinates(
        atom_coord_local_voxel=atom_coord_local_voxel,
        atom_is_in_core_box=atom_is_in_core_box,
        box_shape_zyx=np.asarray(box_shape_zyx, dtype=np.int64),
    )
