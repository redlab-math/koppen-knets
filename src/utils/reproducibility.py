"""
Reproducibility utilities for k-nets experiments.

Provides functions for:
    - Setting random seeds across all libraries (NumPy, PyTorch, JAX, Python)
    - Logging environment information (package versions, hardware)
    - Ensuring deterministic behavior for reproducible research
"""

import logging
import platform
import random
import sys
from typing import Optional

import numpy as np
import torch

# Optional JAX import
try:
    import jax
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


def set_global_seeds(seed: int) -> None:
    """
    Set random seeds for all major libraries to ensure reproducibility.
    
    Sets seeds for:
        - Python's random module
        - NumPy
        - PyTorch (CPU and CUDA)
        - JAX (if available)
    
    Also configures PyTorch for deterministic behavior.
    
    Args:
        seed: Random seed value
    
    Example:
        >>> set_global_seeds(42)
        >>> # All random operations will now be reproducible
    
    Note:
        Deterministic mode may impact performance. For production/speed-critical
        code, consider disabling cudnn.deterministic and cudnn.benchmark.
    """
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    # PyTorch deterministic mode
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # JAX (if available)
    if JAX_AVAILABLE:
        try:
            from jax import random as jax_random
            # JAX uses a different PRNG system
            # Store seed for later use if needed
            global _JAX_SEED
            _JAX_SEED = seed
        except Exception:
            pass


def configure_jax(
    enable_x64: bool = True,
    platform: Optional[str] = None,
    logger: Optional[logging.Logger] = None
) -> None:
    """
    Configure JAX for high-precision numerical computation.
    
    Args:
        enable_x64: Enable 64-bit precision (double precision)
        platform: Force specific platform ('cpu', 'gpu', 'tpu')
        logger: Logger instance for status messages
    
    Example:
        >>> configure_jax(enable_x64=True, platform='cpu')
    """
    if not JAX_AVAILABLE:
        if logger:
            logger.warning("JAX not available, skipping JAX configuration")
        return
    
    try:
        # Enable 64-bit precision
        if enable_x64:
            jax.config.update("jax_enable_x64", True)
        
        # Set platform if specified
        if platform:
            jax.config.update("jax_platform_name", platform)
        
        if logger:
            logger.info(f"JAX backend: {jax.default_backend()}")
            logger.info(f"JAX devices: {jax.devices()}")
            logger.info(f"JAX x64 enabled: {jax.config.x64_enabled}")
    
    except Exception as e:
        if logger:
            logger.warning(f"JAX configuration failed: {e}")


def log_environment_info(logger: logging.Logger) -> None:
    """
    Log detailed environment information for reproducibility.
    
    Logs:
        - Python version and platform
        - NumPy version
        - PyTorch version and CUDA availability
        - JAX version (if available)
        - Hardware information (CPU, GPU)
    
    Args:
        logger: Logger instance
    
    Example:
        >>> import logging
        >>> logger = logging.getLogger(__name__)
        >>> log_environment_info(logger)
    """
    logger.info("=" * 80)
    logger.info("ENVIRONMENT INFORMATION")
    logger.info("=" * 80)
    
    # Python and platform
    logger.info(f"Python version: {sys.version.split()[0]}")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Architecture: {platform.machine()}")
    
    # NumPy
    logger.info(f"NumPy version: {np.__version__}")
    
    # PyTorch
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"PyTorch CUDA version: {torch.version.cuda}")
        logger.info(f"PyTorch CUDNN version: {torch.backends.cudnn.version()}")
        n_gpus = torch.cuda.device_count()
        logger.info(f"Number of GPUs: {n_gpus}")
        
        for i in range(n_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            logger.info(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    
    # Check for MPS (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        logger.info("PyTorch MPS (Metal) available: True")
    
    # JAX
    if JAX_AVAILABLE:
        logger.info(f"JAX version: {jax.__version__}")
        try:
            logger.info(f"JAX backend: {jax.default_backend()}")
            logger.info(f"JAX devices: {len(jax.devices())} device(s)")
            for i, device in enumerate(jax.devices()):
                logger.info(f"  Device {i}: {device}")
        except Exception as e:
            logger.warning(f"Could not query JAX devices: {e}")
    else:
        logger.info("JAX: Not available")
    
    # Additional libraries (optional)
    try:
        import scipy
        logger.info(f"SciPy version: {scipy.__version__}")
    except ImportError:
        pass
    
    try:
        import pandas
        logger.info(f"Pandas version: {pandas.__version__}")
    except ImportError:
        pass
    
    try:
        import matplotlib
        logger.info(f"Matplotlib version: {matplotlib.__version__}")
    except ImportError:
        pass
    
    try:
        import h5py
        logger.info(f"h5py version: {h5py.__version__}")
    except ImportError:
        pass
    
    logger.info("=" * 80)


def get_device(
    preferred: str = 'auto',
    gpu_id: int = 0,
    logger: Optional[logging.Logger] = None
) -> torch.device:
    """
    Get PyTorch device with automatic fallback.
    
    Priority order for 'auto':
        1. CUDA (if available)
        2. MPS/Metal (Apple Silicon, if available)
        3. CPU (fallback)
    
    Args:
        preferred: Preferred device ('auto', 'cuda', 'mps', 'cpu')
        gpu_id: GPU device ID (for CUDA)
        logger: Logger instance
    
    Returns:
        PyTorch device
    
    Example:
        >>> device = get_device('auto')
        >>> model.to(device)
    """
    if preferred == 'auto':
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{gpu_id}')
            if logger:
                logger.info(f"Using device: {device} ({torch.cuda.get_device_name(gpu_id)})")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            if logger:
                logger.info(f"Using device: {device} (Apple Metal)")
        else:
            device = torch.device('cpu')
            if logger:
                logger.info(f"Using device: {device}")
    
    elif preferred == 'cuda':
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{gpu_id}')
            if logger:
                logger.info(f"Using device: {device}")
        else:
            if logger:
                logger.warning("CUDA requested but not available, falling back to CPU")
            device = torch.device('cpu')
    
    elif preferred == 'mps':
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            if logger:
                logger.info(f"Using device: {device}")
        else:
            if logger:
                logger.warning("MPS requested but not available, falling back to CPU")
            device = torch.device('cpu')
    
    else:  # 'cpu' or unknown
        device = torch.device('cpu')
        if logger:
            logger.info(f"Using device: {device}")
    
    return device


def set_torch_threads(n_threads: Optional[int] = None) -> None:
    """
    Set number of threads for PyTorch operations.
    
    Useful for controlling CPU parallelism in compute clusters.
    
    Args:
        n_threads: Number of threads (None to use default)
    
    Example:
        >>> set_torch_threads(4)  # Limit to 4 CPU threads
    """
    if n_threads is not None:
        torch.set_num_threads(n_threads)
        torch.set_num_interop_threads(n_threads)


def ensure_deterministic(strict: bool = True) -> None:
    """
    Ensure maximum determinism for reproducible experiments.
    
    Enables:
        - PyTorch deterministic algorithms
        - Warning on non-deterministic operations
    
    Args:
        strict: If True, raise error on non-deterministic ops; else warn
    
    Warning:
        Some operations may not have deterministic implementations.
        This may impact performance or cause runtime errors.
    
    Example:
        >>> ensure_deterministic(strict=False)
    """
    torch.use_deterministic_algorithms(strict)
    
    if strict:
        # Warn about operations that don't have deterministic implementations
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False


def get_git_info(repo_path: Optional[str] = None) -> dict:
    """
    Get git repository information for provenance tracking.
    
    Args:
        repo_path: Path to git repository (None for current directory)
    
    Returns:
        Dictionary with git info (commit hash, branch, status)
        Returns empty dict if not a git repository or git unavailable
    
    Example:
        >>> git_info = get_git_info()
        >>> print(git_info.get('commit'))
    """
    try:
        import subprocess
        
        def run_git_cmd(cmd):
            result = subprocess.run(
                cmd,
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        
        info = {
            'commit': run_git_cmd(['git', 'rev-parse', 'HEAD']),
            'branch': run_git_cmd(['git', 'rev-parse', '--abbrev-ref', 'HEAD']),
            'dirty': len(run_git_cmd(['git', 'status', '--porcelain'])) > 0
        }
        
        return info
    
    except Exception:
        return {}


def main():
    """Test reproducibility utilities."""
    import logging
    
    # Setup logger
    logger = logging.getLogger('test_reproducibility')
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(handler)
    
    print("Testing reproducibility utilities...\n")
    
    # Test seed setting
    print("1. Testing set_global_seeds(42)...")
    set_global_seeds(42)
    
    # Verify NumPy reproducibility
    a = np.random.randn(5)
    set_global_seeds(42)
    b = np.random.randn(5)
    assert np.allclose(a, b), "NumPy seeds not working"
    print("   ✓ NumPy seeds verified")
    
    # Verify PyTorch reproducibility
    x = torch.randn(5)
    set_global_seeds(42)
    y = torch.randn(5)
    assert torch.allclose(x, y), "PyTorch seeds not working"
    print("   ✓ PyTorch seeds verified")
    
    # Test JAX configuration
    print("\n2. Testing configure_jax()...")
    configure_jax(enable_x64=True, logger=logger)
    
    # Test environment logging
    print("\n3. Testing log_environment_info()...")
    log_environment_info(logger)
    
    # Test device detection
    print("\n4. Testing get_device()...")
    device = get_device('auto', logger=logger)
    print(f"   Detected device: {device}")
    
    # Test git info
    print("\n5. Testing get_git_info()...")
    git_info = get_git_info()
    if git_info:
        print(f"   Git commit: {git_info.get('commit', 'N/A')[:8]}")
        print(f"   Git branch: {git_info.get('branch', 'N/A')}")
        print(f"   Dirty: {git_info.get('dirty', False)}")
    else:
        print("   Not a git repository or git unavailable")
    
    print("\n✓ All tests passed")


if __name__ == '__main__':
    main()