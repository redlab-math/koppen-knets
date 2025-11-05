
import logging
import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Callable

class KSTSprecherLorentzModel(nn.Module):
    """
    Modelo KST-Sprecher-Lorentz: f(x) = Σ_q Φ(Z_q)
    
    Donde:
        Z_q = Σ_p λ_qp · ψ(x_p + qε)
        Φ: red neuronal univariada compartida
        λ_qp: coeficientes entrenables
    """
    
    def __init__(
        self,
        projector,
        hidden_sizes: List[int] = [256, 128, 64],
        dropout: float = 0.1,
        lambda_noise: float = 0.1,
        activation: Callable = nn.GELU,
        logger: Optional[logging.Logger] = None
    ):
        """
        Args:
            projector: Instancia de KSTProjector con ψ pre-computada
            hidden_sizes: Arquitectura de la red externa Φ
            dropout: Tasa de dropout
            lambda_noise: Cantidad de ruido para inicializar λ
            activation: La función de activación a usar en la red Φ (e.g., nn.ReLU, torch.sin)
            logger: Logger opcional
        """
        super().__init__()
        
        self.projector = projector
        self.n = projector.n
        self.Q = projector.Q
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        
        # =================================================================
        # INICIALIZACIÓN DE λ (SIN CAMBIOS)
        # =================================================================
        lambda_init = self._initialize_lambda_matrix()
        
        noise = torch.randn_like(lambda_init) * lambda_noise
        lambda_perturbed = lambda_init + noise
        
        lambda_perturbed = lambda_perturbed / lambda_perturbed.sum(
            dim=1, keepdim=True
        )
        
        self.lambda_matrix = nn.Parameter(lambda_perturbed)
        
        # =================================================================
        # RED EXTERNA Φ CON ACTIVACIÓN CONFIGURABLE
        # =================================================================
        layers = []
        current_size = 1
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(current_size, hidden_size),
                activation(),
                nn.Dropout(dropout)
            ])
            current_size = hidden_size
        
        layers.append(nn.Linear(current_size, 1))
        self.outer_net = nn.Sequential(*layers)
        
        self._log_architecture()
    
    def _initialize_lambda_matrix(self) -> torch.Tensor:
        """
        Inicializa la matriz λ.
        """
        Q, n = self.Q, self.n
        
        if hasattr(self.projector, 'lambdas'):
            lambda_p = self.projector.lambdas
            if isinstance(lambda_p, np.ndarray):
                lambda_p = torch.from_numpy(lambda_p).float()
            lambda_init = lambda_p.unsqueeze(0).repeat(Q, 1)
        else:
            lambda_init = torch.zeros(Q, n)
            for q in range(Q):
                for p in range(n):
                    lambda_init[q, p] = np.sqrt(p + 1) / (2 * n - 1)
            lambda_init = lambda_init / lambda_init.sum(dim=1, keepdim=True)
        
        return lambda_init
    
    def _log_architecture(self):
        """Log detallado de la arquitectura del modelo."""
        n_params_lambda = self.lambda_matrix.numel()
        n_params_phi = sum(p.numel() for p in self.outer_net.parameters())
        n_params_total = n_params_lambda + n_params_phi
        
        self.logger.info("KST-SprecherLorentzModel inicializado:")
        self.logger.info(f"  - Dimensión de entrada (n): {self.n}")
        self.logger.info(f"  - Número de proyecciones (Q): {self.Q}")
        self.logger.info(f"  - Matriz λ_qp entrenable con tamaño: {tuple(self.lambda_matrix.shape)}")
        
        phi_arch = [1]
        for layer in self.outer_net:
            if isinstance(layer, nn.Linear):
                phi_arch.append(layer.out_features)
        phi_arch_str = " -> ".join(map(str, phi_arch))
        
        activation_name = "custom"
        if len(self.outer_net) > 1 and hasattr(self.outer_net[1], '__class__'):
            activation_name = self.outer_net[1].__class__.__name__
        
        self.logger.info(f"  - Arquitectura de red externa Φ (compartida): MLP {phi_arch_str} con activación {activation_name}")
        self.logger.info(f"  - Parámetros: λ={n_params_lambda:,}, Φ={n_params_phi:,}, Total={n_params_total:,}")
        
        lambda_mean = self.lambda_matrix.mean().item()
        lambda_std = self.lambda_matrix.std().item()
        self.logger.info(f"  - λ inicializado: mean={lambda_mean:.4f}, std={lambda_std:.4f}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass del modelo KST-SL (VECTORIZADO EFICIENTE).
        """
        # Calcular proyecciones Z usando los lambdas entrenables.
        Z_adjusted = self._apply_trainable_lambdas(x)
        
        # Aplicar la red externa Φ a cada proyección Z_q.
        # Z_adjusted tiene shape (batch, Q). Lo expandimos a (batch, Q, 1)
        # para que la red lineal (que espera un input de tamaño 1) pueda aplicarse.
        Z_reshaped = Z_adjusted.unsqueeze(-1)
        
        # El MLP se aplica a lo largo de la dimensión Q.
        # La red es compartida, por lo que se aplica el mismo MLP a cada Z_q.
        chi = self.outer_net(Z_reshaped)  # Shape: (batch, Q, 1)
        
        # Sumar las contribuciones de cada proyección.
        y = torch.sum(chi, dim=1) # Shape: (batch, 1)
        
        return y
    
    def _apply_trainable_lambdas(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calcula las proyecciones Z usando lambda_matrix entrenable.
        VERSIÓN VECTORIZADA Y CORREGIDA para alto rendimiento.
        """
        device = x.device
        x_np = x.detach().cpu().numpy()
        batch_size = x_np.shape[0]

        # Crear un array de shifts para todas las proyecciones (Q).
        shifts = np.arange(self.Q).reshape(self.Q, 1) * self.projector.epsilon

        # Expandir X y los shifts para broadcasting.
        X_shifted = x_np[:, np.newaxis, :] + shifts[np.newaxis, :, :]
        
        # Llamar a la función de interpolación UNA SOLA VEZ con el array completo.
        psi_values = self.projector.psi(X_shifted)
        
        # Convertir a Torch para la multiplicación con los lambdas.
        psi_t = torch.from_numpy(psi_values).float().to(device)
        
        # Multiplicación matricial usando einsum.
        Z = torch.einsum('bqp,qp->bq', psi_t, self.lambda_matrix)
        
        return Z

    def get_projection_contributions(self, x: torch.Tensor) -> torch.Tensor:
        """
        Retorna las contribuciones individuales de cada proyección.
        """
        with torch.no_grad():
            Z = self._apply_trainable_lambdas(x)
            Z_reshaped = Z.unsqueeze(-1)
            contributions = self.outer_net(Z_reshaped).squeeze(-1)
        
        return contributions
    
    def get_lambda_stats(self) -> dict:
        """
        Retorna estadísticas sobre los coeficientes λ actuales.
        """
        lambda_np = self.lambda_matrix.detach().cpu().numpy()
        return {
            'mean': float(lambda_np.mean()),
            'std': float(lambda_np.std()),
            'min': float(lambda_np.min()),
            'max': float(lambda_np.max()),
            'matrix': lambda_np,
            'sum_per_projection': lambda_np.sum(axis=1).tolist()
        }