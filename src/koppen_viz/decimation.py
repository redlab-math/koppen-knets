"""
Adaptive decimation for plotting large datasets.
Preserves critical features using multi-resolution sampling.
"""

import numpy as np
from typing import Tuple
import h5py


def compute_curvature_importance(y: np.ndarray, window: int = 5) -> np.ndarray:
    """
    Compute local curvature as importance metric.
    Higher curvature = more important to keep.
    """
    # Second derivative (discrete curvature)
    d2y = np.abs(np.diff(np.diff(y)))
    
    # Pad to match original length
    importance = np.concatenate([[0], d2y, [0]])
    
    # Smooth with rolling window
    if window > 1:
        kernel = np.ones(window) / window
        importance = np.convolve(importance, kernel, mode='same')
    
    return importance


def multiscale_decimation(x: np.ndarray, y: np.ndarray, 
                         target_points: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Intelligent multi-scale decimation preserving features.
    
    Strategy:
    1. Always keep endpoints
    2. Sample uniformly at multiple scales (dyadic)
    3. Add high-curvature regions
    4. Remove duplicates and sort
    
    Args:
        x, y: Full data arrays
        target_points: Desired number of points
    
    Returns:
        Decimated (x, y) arrays
    """
    n = len(x)
    
    if n <= target_points:
        return x, y
    
    indices = set()
    
    # 1. Always keep endpoints
    indices.add(0)
    indices.add(n - 1)
    
    # 2. Multi-scale uniform sampling (dyadic levels)
    n_scales = int(np.log2(n / target_points)) + 1
    points_per_scale = target_points // (n_scales + 1)
    
    for level in range(n_scales):
        step = max(1, n // (2 ** level))
        level_indices = np.arange(0, n, step)[:points_per_scale]
        indices.update(level_indices)
    
    # 3. Add high-curvature regions
    importance = compute_curvature_importance(y)
    
    # Sort by importance and take top ones
    n_curvature = min(target_points // 3, n)
    top_curvature_idx = np.argsort(importance)[-n_curvature:]
    indices.update(top_curvature_idx)
    
    # 4. Convert to sorted array
    indices = np.array(sorted(indices))
    
    # Trim to exact target
    if len(indices) > target_points:
        # Keep evenly spaced subset
        keep_idx = np.linspace(0, len(indices) - 1, target_points, dtype=int)
        indices = indices[keep_idx]
    
    return x[indices], y[indices]


def decimate_from_hdf5(h5file: h5py.File, 
                       dataset_x: str,
                       dataset_y: str,
                       target_points: int = 10000,
                       load_strategy: str = 'sample') -> Tuple[np.ndarray, np.ndarray]:
    """
    Decimate data directly from HDF5 without loading full arrays.
    
    Args:
        h5file: Open HDF5 file
        dataset_x, dataset_y: Dataset paths
        target_points: Target number of points
        load_strategy: 'sample' (fast) or 'full' (accurate decimation)
    
    Returns:
        Decimated (x, y) arrays
    """
    x_ds = h5file[dataset_x]
    y_ds = h5file[dataset_y]
    n_total = len(x_ds)
    
    if n_total <= target_points:
        return x_ds[:], y_ds[:]
    
    if load_strategy == 'sample':
        # Fast: uniform sampling directly
        indices = np.linspace(0, n_total - 1, target_points, dtype=int)
        return x_ds[indices], y_ds[indices]
    
    elif load_strategy == 'full':
        # Accurate: load all and use smart decimation
        # Only feasible for moderate k
        print("  Loading full data for accurate decimation...")
        x_full = x_ds[:]
        y_full = y_ds[:]
        return multiscale_decimation(x_full, y_full, target_points)
    
    else:
        raise ValueError(f"Unknown load_strategy: {load_strategy}")


def adaptive_decimation_for_k(k: int, max_points: int = 50000) -> int:
    """
    Determine appropriate number of plot points based on k.
    
    For small k: use more points (capture detail)
    For large k: use fewer points (avoid memory issues)
    """
    if k <= 4:
        return min(max_points, 10000)
    elif k <= 6:
        return min(max_points, 20000)
    elif k <= 8:
        return min(max_points, 30000)
    else:
        return max_points