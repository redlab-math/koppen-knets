

import numpy as np
import h5py
from pathlib import Path
from typing import Callable, Tuple, Union
import warnings
import logging

# Configuración de logging para este módulo
logging.basicConfig(level=logging.INFO, format='[%(levelname)s - KSTProjector] %(message)s')

class KSTProjector:
    """
    Efficient KST projection engine with Lipschitz continuous inner function.
    """
    
    def __init__(
        self,
        n: int,
        psi_checkpoint: Union[Path, str],
        vectorized: bool = True
    ):
        self.n = n
        self.Q = 2 * n + 1
        self.vectorized = vectorized
        
        logging.info(f"Initializing for n={n} with checkpoint '{psi_checkpoint}'")
        
        self.psi, self.psi_base_domain, self.s_vals, self.psi_vals = self._load_psi(psi_checkpoint)
        self.lambdas = self._compute_lambdas(n)
        self.epsilon = 1.0 / (2.0 * n)
        
        logging.info(f"Coefficients loaded: ε={self.epsilon:.4f}, λ_p range=[{self.lambdas.min():.4f}, {self.lambdas.max():.4f}]")

    def _compute_lambdas(self, n: int) -> np.ndarray:
        """Computes integrally independent coefficients."""
        lambdas = np.array([1.0 / np.sqrt(p + 2) for p in range(n)])
        return lambdas / lambdas.sum()
    
    def _load_psi(self, checkpoint_path: Union[Path, str]) -> Tuple[Callable, Tuple[float, float], np.ndarray, np.ndarray]:
        """
        Loads pre-computed Lipschitz continuous function ψ(s) from HDF5.
        
        Returns the function as a callable that evaluates using left-step lookup
        (no interpolation).
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            logging.error(f"Checkpoint file not found at '{checkpoint_path}'")
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logging.info("Loading Lipschitz ψ(s) from HDF5...")
        
        with h5py.File(checkpoint_path, 'r') as h5f:
            if 'lipschitz_function/s_lipschitz' not in h5f:
                raise KeyError("Dataset 'lipschitz_function/s_lipschitz' not found in HDF5")
            if 'lipschitz_function/psi_lipschitz' not in h5f:
                raise KeyError("Dataset 'lipschitz_function/psi_lipschitz' not found in HDF5")

            # Dominio uniforme [0,1] y valores de ψ(s) Lipschitz
            s_vals = h5f['lipschitz_function/s_lipschitz'][:]
            psi_vals = h5f['lipschitz_function/psi_lipschitz'][:]
            
            base_domain = (s_vals[0], s_vals[-1])
            
            logging.info(f"Loaded {len(s_vals):,} points from Lipschitz ψ(s) on domain [{base_domain[0]:.6f}, {base_domain[1]:.6f}]")
            
            # Validar dimensión
            meta = h5f['metadata']
            stored_n = int(meta.attrs['n'])
            if stored_n != self.n:
                logging.error(f"Dimension mismatch: Checkpoint is for n={stored_n}, projector is for n={self.n}")
                raise ValueError(f"Dimension mismatch in checkpoint")

        # ================================================================
        # EXTENSIÓN PERIÓDICA CON LOOKUP DE ESCALÓN IZQUIERDO
        # ================================================================
        def psi_extended(x_in: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
            """
            Applies the periodic extension ψ(x) = ψ(x - ⌊x⌋) + ⌊x⌋.
            
            Uses left-step lookup (no interpolation): for any query point,
            returns the value at the largest s_i ≤ query.
            """
            x = np.atleast_1d(x_in)
            
            x_floor = np.floor(x)
            x_frac = x - x_floor
            
            # Buscar índice del valor inmediato a la izquierda
            # searchsorted con side='right' da el primer índice > x_frac
            # Restamos 1 para obtener el índice del valor ≤ x_frac
            indices = np.searchsorted(s_vals, x_frac, side='right') - 1
            
            # Manejar edge case: si x_frac < s_vals[0], usar índice 0
            indices = np.clip(indices, 0, len(psi_vals) - 1)
            
            psi_frac = psi_vals[indices]
            
            result = psi_frac + x_floor
            
            return result[0] if isinstance(x_in, (float, np.float64)) else result

        return psi_extended, base_domain, s_vals, psi_vals

    def project_batch(self, X: np.ndarray) -> np.ndarray:
        """Projects a batch of points X ∈ R^{N×n} to KST space Z ∈ R^{N×Q}."""
        X = np.atleast_2d(X)
        N = len(X)
        
        if X.shape[1] != self.n:
            raise ValueError(f"Input shape mismatch: Expected {self.n} dims, got {X.shape[1]}")
        
        if np.any(X < 0) or np.any(X > 1):
            warnings.warn(
                f"Input X is outside the expected [0, 1] domain. Clipping values.",
                UserWarning
            )
            X = np.clip(X, 0.0, 1.0)
            
        return self._project_batch_vectorized(X)
    
    def _project_batch_vectorized(self, X: np.ndarray) -> np.ndarray:
        """Vectorized implementation for speed."""
        N = len(X)
        Z = np.zeros((N, self.Q), dtype=np.float64)
        
        for q in range(self.Q):
            shift = q * self.epsilon
            
            X_shifted = X + shift
            
            psi_vals = self.psi(X_shifted)  # Shape: (N, n)
            
            Z[:, q] = np.dot(psi_vals, self.lambdas)
        
        return Z
    
    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Convenience wrapper for projection."""
        return self.project_batch(X)