"""
HDF5 storage
"""

import h5py
import numpy as np
import jax.numpy as jnp
from pathlib import Path
import time

from config.params import KSTConfig
from koppen_core.recursion import koppen_phi_batch
from koppen_core.math import compute_lambdas


def create_checkpoint_file(filepath: Path, config: KSTConfig) -> h5py.File:
    """Create HDF5 file structure for checkpointing."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    h5f = h5py.File(filepath, 'w')
    
    meta = h5f.create_group('metadata')
    meta.attrs['n'], meta.attrs['gamma'], meta.attrs['k'], meta.attrs['epsilon'], meta.attrs['num_points'], meta.attrs['Q'] = \
        config.n, config.gamma, config.k, config.epsilon, config.num_points, config.Q
    
    lambdas = compute_lambdas(config.n)
    meta.create_dataset('lambdas', data=np.array(lambdas))
    
    for group in ['holder_function', 'reparametrization', 'lipschitz_function', 'verification', 'timings']:
        h5f.create_group(group)
        
    return h5f

def generate_holder_streaming(h5file: h5py.File, 
                              config: KSTConfig,
                              chunk_size: int = 100000,
                              verbose: bool = True) -> float:
    """Generate Hölder function ψ̃_k via streaming directly to HDF5."""
    if verbose:
        print(f"\n[STEP 1/4] Generating Hölder function ψ̃_{config.k}...")
        print(f"  Points: {config.num_points:,}\n  Algorithm: Actor Appendix E (Eq 5.3)")
    
    start_time = time.perf_counter()
    n_total, n_chunks = config.num_points, (config.num_points + chunk_size - 1) // chunk_size
    
    holder_grp = h5file['holder_function']
    x_ds = holder_grp.create_dataset('x_holder', shape=(n_total,), dtype='f8', chunks=(min(chunk_size, n_total),), compression='gzip')
    y_ds = holder_grp.create_dataset('y_holder', shape=(n_total,), dtype='f8', chunks=(min(chunk_size, n_total),), compression='gzip')
    
    if verbose: print(f"  Processing {n_chunks} chunks...")
    
    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, n_total)
        
        # Generar chunk de X en CPU con NumPy.
        x_chunk_np = np.linspace(
            start_idx / (n_total - 1),
            (end_idx - 1) / (n_total - 1),
            end_idx - start_idx,
            dtype=np.float64
        )
        
        # Calcular chunk de Y en CPU.
        y_chunk_np = koppen_phi_batch(x_chunk_np, config.n, config.gamma, config.k)
        
        # --- CORRECCIÓN CRÍTICA ---
        # Escribir los datos correctos en el disco.
        x_ds[start_idx:end_idx] = x_chunk_np
        y_ds[start_idx:end_idx] = y_chunk_np
        # --- FIN DE LA CORRECCIÓN ---
        
        if verbose and (i + 1) % max(1, n_chunks // 10) == 0:
            print(f"    Progress: {100 * (i + 1) / n_chunks:.0f}%")
    
    elapsed = time.perf_counter() - start_time
    if verbose: print(f"  ✓ Hölder function generated in {elapsed:.3f}s")
    h5file['timings'].attrs['holder_generation'] = elapsed
    return elapsed

def load_checkpoint(filepath: Path) -> tuple:
    """Loads a previously computed checkpoint file."""
    if not filepath.exists(): raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    h5f = h5py.File(filepath, 'r')
    meta = h5f['metadata']
    config = KSTConfig(n=int(meta.attrs['n']), gamma=int(meta.attrs['gamma']), k=int(meta.attrs['k']), epsilon=float(meta.attrs['epsilon']))
    return h5f, config

def get_memory_estimate(config: KSTConfig) -> dict:
    """Estimates memory requirements for a given configuration."""
    n_points, bytes_per_float64 = config.num_points, 8
    holder_full_gb = 2 * n_points * bytes_per_float64 / 1024**3
    sigma_gb = n_points * bytes_per_float64 / 1024**3
    chunk_size = 100000
    gpu_peak_gb = chunk_size * 2 * bytes_per_float64 / 1024**3
    return {
        'holder_full_ram_gb': holder_full_gb, 
        'sigma_ram_gb': sigma_gb, 
        'total_disk_gb': (holder_full_gb / 2) + sigma_gb,
        'gpu_peak_gb': gpu_peak_gb
    }

