"""
Memory management utilities for large k.
"""

import psutil
import jax
from typing import Dict
import gc


def get_memory_info() -> Dict[str, float]:
    """
    Get current memory usage statistics.
    
    Returns:
        Dict with RAM and GPU memory in MB
    """
    process = psutil.Process()
    mem_info = process.memory_info()
    
    info = {
        'ram_used_mb': mem_info.rss / 1024**2,
        'ram_available_mb': psutil.virtual_memory().available / 1024**2,
        'ram_percent': psutil.virtual_memory().percent,
    }
    
    # --- CORRECCIÓN ---
    # Inicializa la clave para asegurar que siempre exista, incluso si no hay GPU.
    info['gpu_allocated_mb'] = None
    
    # Intenta obtener la memoria de la GPU
    try:
        devices = jax.devices()
        if devices and hasattr(devices[0], 'memory_stats'):
            stats = devices[0].memory_stats()
            if stats:
                # Sobrescribe el valor solo si se encuentra una GPU con estadísticas.
                info['gpu_allocated_mb'] = stats.get('bytes_in_use', 0) / 1024**2
    except Exception:
        # En caso de un error inesperado, el valor ya está establecido en None.
        pass
    
    return info


def clear_jax_cache():
    """Clear JAX compilation cache and force garbage collection"""
    jax.clear_caches()
    gc.collect()


class MemoryMonitor:
    """Context manager for monitoring memory usage"""
    
    def __init__(self, label: str = "Operation"):
        self.label = label
        self.start_info = None
    
    def __enter__(self):
        clear_jax_cache()
        self.start_info = get_memory_info()
        print(f"\n[{self.label}] Memory at start:")
        print(f"  RAM: {self.start_info['ram_used_mb']:.1f} MB "
              f"({self.start_info['ram_percent']:.1f}% used)")
        if self.start_info['gpu_allocated_mb'] is not None:
            print(f"  GPU: {self.start_info['gpu_allocated_mb']:.1f} MB")
        return self
    
    def __exit__(self, *args):
        end_info = get_memory_info()
        
        ram_delta = end_info['ram_used_mb'] - self.start_info['ram_used_mb']
        
        print(f"\n[{self.label}] Memory at end:")
        print(f"  RAM: {end_info['ram_used_mb']:.1f} MB "
              f"(Δ{ram_delta:+.1f} MB)")
        
        if end_info['gpu_allocated_mb'] is not None and \
           self.start_info['gpu_allocated_mb'] is not None:
            gpu_delta = (end_info['gpu_allocated_mb'] - 
                        self.start_info['gpu_allocated_mb'])
            print(f"  GPU: {end_info['gpu_allocated_mb']:.1f} MB "
                  f"(Δ{gpu_delta:+.1f} MB)")
        
        clear_jax_cache()


def check_resources(config, min_ram_gb: float = 4.0, min_gpu_mb: float = 1000):
    """
    Check if system has sufficient resources.
    
    Raises RuntimeError if insufficient.
    """
    from .storage import get_memory_estimate
    
    mem = psutil.virtual_memory()
    available_ram_gb = mem.available / 1024**3
    
    if available_ram_gb < min_ram_gb:
        raise RuntimeError(
            f"Insufficient RAM: {available_ram_gb:.2f} GB available, "
            f"need at least {min_ram_gb:.2f} GB"
        )
    
    # Estimate requirements
    estimates = get_memory_estimate(config)
    
    print("\nResource estimates:")
    print(f"  Required disk space: {estimates['total_disk_gb']:.2f} GB")
    print(f"  Peak RAM usage: {estimates['holder_full_ram_gb']:.2f} GB")
    print(f"  Peak GPU memory: {estimates['gpu_peak_gb']:.2f} GB")
    print(f"  Available RAM: {available_ram_gb:.2f} GB")
    
    if estimates['holder_full_ram_gb'] > available_ram_gb * 0.8:
        raise RuntimeError(
            f"Configuration may exceed available RAM!\n"
            f"  Estimated peak: {estimates['holder_full_ram_gb']:.2f} GB\n"
            f"  Available: {available_ram_gb:.2f} GB\n"
            f"  Recommendation: Use streaming or reduce k"
        )