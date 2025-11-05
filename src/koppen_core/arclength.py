"""
Arc length computation for reparametrization (Section 5.3).
Streaming implementation for large k.
"""
import jax.numpy as jnp
from jax import jit
import h5py
import numpy as np
from typing import Tuple

@jit
def compute_arclength_segment(x_seg: jnp.ndarray, y_seg: jnp.ndarray) -> float:
    """Computes arc length of a curve segment: L = Σ √(dx² + dy²)."""
    dx = jnp.diff(x_seg)
    dy = jnp.diff(y_seg)
    segment_lengths = jnp.sqrt(dx**2 + dy**2)
    return jnp.sum(segment_lengths)

def compute_arclength_streaming(h5_input: h5py.File, 
                                chunk_size: int = 100000) -> Tuple[np.ndarray, float]:
    """
    Compute cumulative arc length function σ(x) via streaming from HDF5.
    """
    # --- CORRECCIÓN ---
    # Se accede a los datasets usando la ruta completa, incluyendo el grupo.
    x_ds = h5_input['holder_function/x_holder']
    y_ds = h5_input['holder_function/y_holder']
    # --- FIN DE LA CORRECCIÓN ---
    
    n_total = len(x_ds)
    print(f"  Computing arc length via streaming ({n_total:,} points)...")
    
    # Pass 1: compute total length
    total_length = 0.0
    n_chunks = (n_total - 1 + chunk_size - 1) // chunk_size
    
    for i in range(n_chunks):
        start = i * chunk_size
        end = min(start + chunk_size + 1, n_total)
        x_chunk, y_chunk = jnp.array(x_ds[start:end]), jnp.array(y_ds[start:end])
        total_length += float(compute_arclength_segment(x_chunk, y_chunk))
        if (i + 1) % max(1, n_chunks // 10) == 0:
            print(f"    Arc length pass 1: {100*(i+1)/n_chunks:.0f}% complete")
    
    print(f"  Total arc length L = {total_length:.6f}")
    
    # Pass 2: compute cumulative normalized sigma
    sigma = np.zeros(n_total, dtype=np.float64)
    cumulative = 0.0
    
    for i in range(n_chunks):
        start = i * chunk_size
        end = min(start + chunk_size + 1, n_total)
        x_chunk, y_chunk = jnp.array(x_ds[start:end]), jnp.array(y_ds[start:end])
        
        lengths = np.array(jnp.sqrt(jnp.diff(x_chunk)**2 + jnp.diff(y_chunk)**2))
        chunk_cumulative = np.concatenate([[0], np.cumsum(lengths)])
        sigma[start:end] = cumulative + chunk_cumulative
        cumulative = sigma[end - 1]
        if (i + 1) % max(1, n_chunks // 10) == 0:
            print(f"    Arc length pass 2: {100*(i+1)/n_chunks:.0f}% complete")
            
    sigma = sigma / total_length
    return sigma, total_length
