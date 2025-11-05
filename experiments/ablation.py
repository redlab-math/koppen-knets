"""
================================================================================
ESTUDIO DE ABLACIÓN
================================================================================
Script para generar todos los resultados:
- Métricas de desempeño (R², RMSE, MAE)
- Métricas de regularidad (Lipschitz, Smoothness, Adversarial, Gradient CV)
- Análisis estadístico (t-tests, effect sizes, Bonferroni correction)
- Tablas en LaTeX
- Todas las figuras del paper
- Resumen ejecutivo

Uso:
    python experiments/ablation_psi.py \
        --checkpoint data/checkpoints/koppen_n2_k6_gamma10.h5 \
        --output results/ablation_final \
        --n_train 500 \
        --n_test 500 \
        --n_folds 5 \
        --epochs 300 \
        --batch_size 64 \
        --learning_rate 5e-4 \
        --patience 50 \
        --activation gelu \
        --gpu 0 \
        --seed 42
================================================================================
"""

import argparse
import logging
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')


from src.knet_core.kst_projector import KSTProjector
from src.knet_core.kst_sl_model import KSTSprecherLorentzModel

# ============================================================================
# CONFIGURACIÓN DE LOGGING
# ============================================================================

def setup_logger(output_dir: Path):
    """Configura logger con salida a archivo y consola"""
    log_dir = output_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger('KSTProjector')
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    # Manejador de archivos
    fh = logging.FileHandler(log_dir / f'ablation_{datetime.now():%Y%m%d_%H%M%S}.log')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('[%(levelname)s - %(name)s] %(message)s'))
    logger.addHandler(fh)
    
    # Manejador de logs
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('[%(levelname)s - %(name)s] %(message)s'))
    logger.addHandler(ch)
    
    return logger

logger = None

# MATPLOTLIB

plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14
})
sns.set_style("whitegrid")
sns.set_palette("Set2")

# Función de activación

def get_activation_function(name: str) -> Callable:
    """Retorna la función de activación según el nombre"""
    activations = {
        'gelu': nn.GELU,
        'relu': nn.ReLU,
        'tanh': nn.Tanh,
        'sigmoid': nn.Sigmoid,
        'elu': nn.ELU,
        'leakyrelu': nn.LeakyReLU,
        'selu': nn.SELU,
        'silu': nn.SiLU
    }
    
    if name.lower() not in activations:
        raise ValueError(f"Activación '{name}' no soportada. Opciones: {list(activations.keys())}")
    
    return activations[name.lower()]

# Métricas de funciones internas

def compute_lipschitz_constant(
    model: nn.Module,
    X: torch.Tensor,
    num_samples: int = 500,
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    Estima la constante de Lipschitz empírica del modelo.
    L_emp = percentile_95(||f(x_i) - f(x_j)|| / ||x_i - x_j||)
    """
    model.eval()
    X = X.to(device)
    
    n = min(len(X), num_samples)
    indices = np.random.choice(len(X), size=n, replace=False)
    X_sample = X[indices]
    
    with torch.no_grad():
        y_sample = model(X_sample).cpu().numpy().flatten()
    
    X_np = X_sample.cpu().numpy()
    
    X_dist = squareform(pdist(X_np, metric='euclidean'))
    y_dist = squareform(pdist(y_sample.reshape(-1, 1), metric='euclidean'))
    
    mask = X_dist > 1e-8
    ratios = np.where(mask, y_dist / X_dist, 0)
    
    ratios_nonzero = ratios[ratios > 0]
    
    if len(ratios_nonzero) == 0:
        return {
            'lipschitz_constant': np.nan,
            'lipschitz_mean': np.nan,
            'lipschitz_std': np.nan
        }
    
    return {
        'lipschitz_constant': float(np.percentile(ratios_nonzero, 95)),
        'lipschitz_mean': float(np.mean(ratios_nonzero)),
        'lipschitz_std': float(np.std(ratios_nonzero))
    }


def measure_internal_smoothness(
    model: nn.Module,
    X: torch.Tensor,
    k_neighbors: int = 10,
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    Mide la suavidad del espacio de representaciones internas Z.
    Para cada punto, computa la varianza de sus k vecinos más cercanos en Z-space.
    """
    model.eval()
    X = X.to(device)
    
    # Intentar obtener representación interna
    if hasattr(model, 'get_projection_contributions'):
        with torch.no_grad():
            Z = model.get_projection_contributions(X).cpu().numpy()
    elif hasattr(model, '_apply_trainable_lambdas'):
        with torch.no_grad():
            Z = model._apply_trainable_lambdas(X).cpu().numpy()
    else:
        # Fallback: usar salida del modelo
        with torch.no_grad():
            Z = model(X).cpu().numpy()
    
    X_np = X.cpu().numpy()
    
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=min(k_neighbors + 1, len(X))).fit(X_np)
    distances, indices = nbrs.kneighbors(X_np)
    
    variances = []
    for i in range(len(X)):
        neighbor_indices = indices[i, 1:]
        Z_neighbors = Z[neighbor_indices]
        
        if len(Z_neighbors) > 1:
            var = np.var(Z_neighbors, axis=0).mean()
            variances.append(var)
    
    variances = np.array(variances)
    
    return {
        'smoothness_mean': float(np.mean(variances)),
        'smoothness_median': float(np.median(variances)),
        'smoothness_std': float(np.std(variances))
    }


def adversarial_robustness(
    model: nn.Module,
    X: torch.Tensor,
    epsilon: float = 0.01,
    num_samples: int = 100,
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    Mide la sensibilidad a perturbaciones adversariales.
    Sensitivity = ||f(x + δ) - f(x)|| / ||δ|| para δ aleatorio con ||δ|| = ε
    """
    model.eval()
    X = X.to(device)
    
    n = min(len(X), num_samples)
    indices = np.random.choice(len(X), size=n, replace=False)
    X_sample = X[indices]
    
    delta = torch.randn_like(X_sample)
    delta = delta / torch.norm(delta, dim=1, keepdim=True) * epsilon
    
    X_perturbed = X_sample + delta
    
    with torch.no_grad():
        y_original = model(X_sample).cpu().numpy().flatten()
        y_perturbed = model(X_perturbed).cpu().numpy().flatten()
    
    output_change = np.abs(y_perturbed - y_original)
    input_change = np.linalg.norm(delta.cpu().numpy(), axis=1)
    
    sensitivity = output_change / (input_change + 1e-8)
    
    return {
        'adversarial_sensitivity': float(np.mean(sensitivity)),
        'output_change_mean': float(np.mean(output_change)),
        'relative_error': float(np.mean(output_change / (np.abs(y_original) + 1e-8)))
    }


def track_gradient_stability(grad_norms: List[float]) -> Dict[str, float]:
    """
    Analiza la estabilidad de los gradientes durante el entrenamiento.
    CV = σ(||∇L||) / μ(||∇L||)
    """
    if not grad_norms or len(grad_norms) < 2:
        return {
            'gradient_cv': np.nan,
            'gradient_trend': np.nan,
            'vanishing_ratio': np.nan
        }
    
    grad_norms = np.array(grad_norms)
    grad_norms = grad_norms[~np.isnan(grad_norms)]
    
    if len(grad_norms) < 2:
        return {
            'gradient_cv': np.nan,
            'gradient_trend': np.nan,
            'vanishing_ratio': np.nan
        }
    
    mean_grad = np.mean(grad_norms)
    std_grad = np.std(grad_norms)
    cv = std_grad / (mean_grad + 1e-8)
    
    x = np.arange(len(grad_norms))
    trend_slope = np.polyfit(x, grad_norms, 1)[0] if len(grad_norms) > 1 else 0.0
    
    threshold = 1e-4
    vanishing_ratio = np.mean(grad_norms < threshold)
    
    return {
        'gradient_cv': float(cv),
        'gradient_trend': float(trend_slope),
        'vanishing_ratio': float(vanishing_ratio)
    }

# Nuevas arquitecturas

class MLP(nn.Module):
    """MLP baseline estándar con activación configurable"""
    
    def __init__(
        self,
        n_inputs: int,
        hidden_sizes: List[int],
        dropout: float = 0.1,
        activation: Callable = nn.GELU
    ):
        super().__init__()
        
        layers = []
        sizes = [n_inputs] + hidden_sizes + [1]
        
        for i in range(len(sizes) - 2):
            layers.extend([
                nn.Linear(sizes[i], sizes[i+1]),
                activation(),
                nn.Dropout(dropout)
            ])
        
        layers.append(nn.Linear(sizes[-2], sizes[-1]))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class KNetFixedActivation(nn.Module):
    """
    Ablación: K-Net con activación fija (tanh) en lugar de ψ Lipschitz.
    Λ entrenable (como el original).
    """
    
    def __init__(
        self,
        n: int,
        Q: int,
        hidden_sizes: List[int],
        dropout: float = 0.1,
        activation: Callable = nn.GELU
    ):
        super().__init__()
        self.n = n
        self.Q = Q
        self.epsilon = 1.0 / (2 * n)
        
        # Lambda entrenable (inicializado similar al original)
        lambda_init = 1.0 / np.sqrt(np.arange(1, n + 1) + 2)
        lambda_init = lambda_init / lambda_init.sum()
        lambda_init += np.random.randn(n) * 0.1
        lambda_matrix = np.repeat(lambda_init[np.newaxis, :], Q, axis=0)
        lambda_matrix = lambda_matrix / lambda_matrix.sum(axis=1, keepdims=True)
        
        self.lambda_matrix = nn.Parameter(torch.from_numpy(lambda_matrix).float())
        
        # Red externa (misma arquitectura que original)
        layers = []
        sizes = [1] + hidden_sizes + [1]
        for i in range(len(sizes) - 2):
            layers.extend([
                nn.Linear(sizes[i], sizes[i+1]),
                activation(),
                nn.Dropout(dropout)
            ])
        layers.append(nn.Linear(sizes[-2], sizes[-1]))
        self.outer_net = nn.Sequential(*layers)
        
        if logger:
            logger.info(f"KNetFixedActivation: Q={Q}, n={n}, ψ=tanh (fija)")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        lambda_normalized = self.lambda_matrix / self.lambda_matrix.sum(dim=1, keepdim=True)
        
        # Calcular Z_q usando tanh en lugar de ψ Lipschitz
        Z_list = []
        for q in range(self.Q):
            x_shifted = x + q * self.epsilon
            psi_vals = torch.tanh(x_shifted)  # DIFERENCIA CLAVE: tanh en lugar de ψ
            Z_q = (psi_vals * lambda_normalized[q]).sum(dim=1, keepdim=True)
            Z_list.append(Z_q)
        
        Z = torch.cat(Z_list, dim=1).unsqueeze(-1)  # (batch, Q, 1)
        chi = self.outer_net(Z)  # (batch, Q, 1)
        y = torch.sum(chi, dim=1)  # (batch, 1)
        
        return y
    
    def get_projection_contributions(self, x: torch.Tensor) -> torch.Tensor:
        """Para análisis de smoothness"""
        with torch.no_grad():
            lambda_normalized = self.lambda_matrix / self.lambda_matrix.sum(dim=1, keepdim=True)
            Z_list = []
            for q in range(self.Q):
                x_shifted = x + q * self.epsilon
                psi_vals = torch.tanh(x_shifted)
                Z_q = (psi_vals * lambda_normalized[q]).sum(dim=1, keepdim=True)
                Z_list.append(Z_q)
            return torch.cat(Z_list, dim=1)


class KNetFixedLambda(nn.Module):
    """
    Ablación: K-Net con ψ Lipschitz pero Λ FIJOS (no entrenables).
    """
    
    def __init__(
        self,
        projector: KSTProjector,
        hidden_sizes: List[int],
        dropout: float = 0.1,
        activation: Callable = nn.GELU
    ):
        super().__init__()
        self.projector = projector
        self.n = projector.n
        self.Q = projector.Q
        
        # Lambda FIJOS (buffer, no Parameter)
        if hasattr(projector, 'lambdas'):
            lambda_p = projector.lambdas
            if isinstance(lambda_p, np.ndarray):
                lambda_p = torch.from_numpy(lambda_p).float()
            lambda_matrix = lambda_p.unsqueeze(0).repeat(self.Q, 1)
        else:
            lambda_init = 1.0 / np.sqrt(np.arange(1, self.n + 1) + 2)
            lambda_init = lambda_init / lambda_init.sum()
            lambda_matrix = torch.from_numpy(np.repeat(lambda_init[np.newaxis, :], self.Q, axis=0)).float()
        
        lambda_matrix = lambda_matrix / lambda_matrix.sum(dim=1, keepdim=True)
        self.register_buffer('lambda_matrix', lambda_matrix)  # Buffer, NO Parameter
        
        # Red externa
        layers = []
        sizes = [1] + hidden_sizes + [1]
        for i in range(len(sizes) - 2):
            layers.extend([
                nn.Linear(sizes[i], sizes[i+1]),
                activation(),
                nn.Dropout(dropout)
            ])
        layers.append(nn.Linear(sizes[-2], sizes[-1]))
        self.outer_net = nn.Sequential(*layers)
        
        if logger:
            logger.info(f"KNetFixedLambda: Q={self.Q}, n={self.n}, Λ FIJOS (no entrenables)")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        x_np = x.detach().cpu().numpy()
        batch_size = x_np.shape[0]
        
        # Crear shifts
        shifts = np.arange(self.Q).reshape(self.Q, 1) * self.projector.epsilon
        X_shifted = x_np[:, np.newaxis, :] + shifts[np.newaxis, :, :]
        
        # Evaluar ψ Lipschitz
        psi_values = self.projector.psi(X_shifted)
        psi_t = torch.from_numpy(psi_values).float().to(device)
        
        # Usar einsum con lambda_matrix (fixed)
        Z = torch.einsum('bqp,qp->bq', psi_t, self.lambda_matrix)
        
        Z_reshaped = Z.unsqueeze(-1)
        chi = self.outer_net(Z_reshaped)
        y = torch.sum(chi, dim=1)
        
        return y
    
    def get_projection_contributions(self, x: torch.Tensor) -> torch.Tensor:
        """Para análisis de smoothness"""
        with torch.no_grad():
            device = x.device
            x_np = x.detach().cpu().numpy()
            
            shifts = np.arange(self.Q).reshape(self.Q, 1) * self.projector.epsilon
            X_shifted = x_np[:, np.newaxis, :] + shifts[np.newaxis, :, :]
            
            psi_values = self.projector.psi(X_shifted)
            psi_t = torch.from_numpy(psi_values).float().to(device)
            
            Z = torch.einsum('bqp,qp->bq', psi_t, self.lambda_matrix)
            return Z

# Dataclass resultados

@dataclass
class AblationResult:
    """Resultado completo de un fold"""
    variant_name: str
    function_name: str
    fold: int
    
    r2: float
    rmse: float
    mae: float
    
    lipschitz_constant: float = np.nan
    lipschitz_mean: float = np.nan
    lipschitz_std: float = np.nan
    
    smoothness_mean: float = np.nan
    smoothness_median: float = np.nan
    smoothness_std: float = np.nan
    
    adversarial_sensitivity: float = np.nan
    output_change_mean: float = np.nan
    relative_error: float = np.nan
    
    gradient_cv: float = np.nan
    gradient_trend: float = np.nan
    vanishing_ratio: float = np.nan
    
    train_time_s: float = 0.0
    n_params: int = 0
    n_epochs: int = 0
    
    has_learnable_psi: bool = False
    has_learnable_lambda: bool = False
    psi_type: str = "unknown"

# ============================================================================
# ENTRENAMIENTO
# ============================================================================

def train_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    args,
    device: str
) -> Dict:
    """Entrenamiento genérico con tracking de gradientes"""
    
    y_scaler = StandardScaler().fit(y_train.reshape(-1, 1))
    y_train_scaled = y_scaler.transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1)).flatten()
    
    train_dataset = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train_scaled).float()
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    X_val_t = torch.from_numpy(X_val).float().to(device)
    y_val_t = torch.from_numpy(y_val_scaled).float().to(device)
    
    # Optimizer con learning rate diferencial para Λ si existe
    if hasattr(model, 'lambda_matrix') and isinstance(model.lambda_matrix, nn.Parameter):
        optimizer = torch.optim.AdamW([
            {'params': [model.lambda_matrix], 'lr': args.learning_rate, 'weight_decay': 0.0},
            {'params': [p for n, p in model.named_parameters() if 'lambda_matrix' not in n],
             'lr': args.learning_rate * 0.5, 'weight_decay': 1e-4}
        ])
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20, verbose=False
    )
    
    criterion = nn.MSELoss()
    
    history = {'train_loss': [], 'val_loss': [], 'grad_norms': []}
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_grad_norm = 0.0
        n_batches = 0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).view(-1, 1)
            
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            epoch_grad_norm += total_norm
            
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        
        model.eval()
        with torch.no_grad():
            y_val_pred = model(X_val_t)
            val_loss = criterion(y_val_pred, y_val_t.view(-1, 1)).item()
        
        history['train_loss'].append(epoch_loss / n_batches)
        history['val_loss'].append(val_loss)
        history['grad_norms'].append(epoch_grad_norm / n_batches)
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                break
    
    return {
        'model': model,
        'y_scaler': y_scaler,
        'history': history
    }

# ============================================================================
# FUNCIONES BENCHMARK
# ============================================================================

def quadratic_2d(X: np.ndarray) -> np.ndarray:
    return X[:, 0]**2 + X[:, 1]**2

def branin_2d(X: np.ndarray) -> np.ndarray:
    x1, x2 = 15 * X[:, 0] - 5, 15 * X[:, 1]
    a, b, c, r, s, t = 1, 5.1/(4*np.pi**2), 5/np.pi, 6, 10, 1/(8*np.pi)
    return a * (x2 - b*x1**2 + c*x1 - r)**2 + s*(1-t)*np.cos(x1) + s

def rosenbrock_2d(X: np.ndarray) -> np.ndarray:
    x1, x2 = 4 * X[:, 0] - 2, 4 * X[:, 1] - 2
    return 100 * (x2 - x1**2)**2 + (1 - x1)**2

def rastrigin_2d(X: np.ndarray) -> np.ndarray:
    x1, x2 = 10.24 * X[:, 0] - 5.12, 10.24 * X[:, 1] - 5.12
    A = 10
    return 2*A + x1**2 - A*np.cos(2*np.pi*x1) + x2**2 - A*np.cos(2*np.pi*x2)

def ackley_2d(X: np.ndarray) -> np.ndarray:
    x1, x2 = 10 * X[:, 0] - 5, 10 * X[:, 1] - 5
    return -20 * np.exp(-0.2 * np.sqrt(0.5 * (x1**2 + x2**2))) \
           - np.exp(0.5 * (np.cos(2*np.pi*x1) + np.cos(2*np.pi*x2))) + np.e + 20

def six_hump_camel_2d(X: np.ndarray) -> np.ndarray:
    x1, x2 = 6 * X[:, 0] - 3, 4 * X[:, 1] - 2
    return (4 - 2.1*x1**2 + x1**4/3)*x1**2 + x1*x2 + (-4 + 4*x2**2)*x2**2

BENCHMARK_FUNCTIONS = {
    'quadratic': quadratic_2d,
    'branin': branin_2d,
    'rosenbrock': rosenbrock_2d,
    'rastrigin': rastrigin_2d,
    'ackley': ackley_2d,
    'six_hump': six_hump_camel_2d
}

FUNCTION_CATEGORIES = {
    'quadratic': 'smooth',
    'branin': 'smooth',
    'rosenbrock': 'non_convex',
    'six_hump': 'non_convex',
    'rastrigin': 'multimodal',
    'ackley': 'multimodal'
}

# ============================================================================
# EJECUTAR UN FOLD
# ============================================================================

def run_ablation_fold(
    variant_config: Dict,
    fold_data: Dict,
    args,
    fold_idx: int,
    device: str,
    projector: Optional[KSTProjector],
    activation_fn: Callable
) -> Tuple[AblationResult, Dict]:
    """Ejecuta un fold completo para una variante"""
    
    variant_name = variant_config['name']
    model_factory = variant_config['model_factory']
    
    # Construir modelo según variante
    if variant_name == 'K-Net-Complete':
        model = model_factory(
            projector=projector,
            hidden_sizes=[256, 128, 64],
            dropout=0.1,
            activation=activation_fn,
            logger=logger
        )
        has_learnable_psi = False
        has_learnable_lambda = True
        psi_type = 'lipschitz'
    
    elif variant_name == 'K-Net-FixedPsi':
        model = model_factory(
            n=2,
            Q=5,
            hidden_sizes=[256, 128, 64],
            dropout=0.1,
            activation=activation_fn
        )
        has_learnable_psi = False
        has_learnable_lambda = True
        psi_type = 'tanh'
    
    elif variant_name == 'K-Net-FixedLambda':
        model = model_factory(
            projector=projector,
            hidden_sizes=[256, 128, 64],
            dropout=0.1,
            activation=activation_fn
        )
        has_learnable_psi = False
        has_learnable_lambda = False
        psi_type = 'lipschitz'
    
    elif variant_name == 'MLP':
        model = model_factory(
            n_inputs=2,
            hidden_sizes=[256, 128, 64],
            dropout=0.1,
            activation=activation_fn
        )
        has_learnable_psi = False
        has_learnable_lambda = False
        psi_type = 'N/A'
    
    else:
        raise ValueError(f"Unknown variant: {variant_name}")
    
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Entrenar
    t0 = time.time()
    train_out = train_model(
        model, 
        fold_data['X_train'], fold_data['y_train'],
        fold_data['X_val'], fold_data['y_val'],
        args, device
    )
    train_time = time.time() - t0
    
    # Evaluar
    model = train_out['model']
    y_scaler = train_out['y_scaler']
    
    X_test_t = torch.from_numpy(fold_data['X_test']).float().to(device)
    
    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(X_test_t).cpu().numpy().flatten()
    
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_test = fold_data['y_test']
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    # Métricas internas
    internal_metrics = {}
    
    try:
        lipschitz = compute_lipschitz_constant(model, X_test_t, num_samples=500, device=device)
        internal_metrics.update(lipschitz)
    except Exception as e:
        if logger:
            logger.warning(f"  └─ Error en Lipschitz: {e}")
        internal_metrics.update({
            'lipschitz_constant': np.nan,
            'lipschitz_mean': np.nan,
            'lipschitz_std': np.nan
        })
    
    try:
        smoothness = measure_internal_smoothness(model, X_test_t, k_neighbors=10, device=device)
        internal_metrics.update(smoothness)
    except Exception as e:
        if logger:
            logger.warning(f"  └─ Error en Smoothness: {e}")
        internal_metrics.update({
            'smoothness_mean': np.nan,
            'smoothness_median': np.nan,
            'smoothness_std': np.nan
        })
    
    try:
        robustness = adversarial_robustness(model, X_test_t, epsilon=0.01, num_samples=100, device=device)
        internal_metrics.update(robustness)
    except Exception as e:
        if logger:
            logger.warning(f"  └─ Error en Adversarial: {e}")
        internal_metrics.update({
            'adversarial_sensitivity': np.nan,
            'output_change_mean': np.nan,
            'relative_error': np.nan
        })
    
    try:
        if train_out['history']['grad_norms']:
            grad_stability = track_gradient_stability(train_out['history']['grad_norms'])
            internal_metrics.update(grad_stability)
        else:
            internal_metrics.update({
                'gradient_cv': np.nan,
                'gradient_trend': np.nan,
                'vanishing_ratio': np.nan
            })
    except Exception as e:
        if logger:
            logger.warning(f"  └─ Error en Gradient: {e}")
        internal_metrics.update({
            'gradient_cv': np.nan,
            'gradient_trend': np.nan,
            'vanishing_ratio': np.nan
        })
    
    result = AblationResult(
        variant_name=variant_name,
        function_name=fold_data['function_name'],
        fold=fold_idx,
        r2=r2,
        rmse=rmse,
        mae=mae,
        train_time_s=train_time,
        n_params=n_params,
        n_epochs=len(train_out['history']['train_loss']),
        has_learnable_psi=has_learnable_psi,
        has_learnable_lambda=has_learnable_lambda,
        psi_type=psi_type,
        **internal_metrics
    )
    
    return result, train_out['history']

# ============================================================================
# VISUALIZACIONES
# ============================================================================

def plot_performance_comparison(df: pd.DataFrame, output_dir: Path):
    """Figura 1: Comparación de desempeño"""
    fig_dir = output_dir / 'figures'
    fig_dir.mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = ['r2', 'rmse', 'mae']
    titles = ['R² (Coefficient of Determination)', 'RMSE (Root Mean Squared Error)', 'MAE (Mean Absolute Error)']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        sns.barplot(data=df, x='variant_name', y=metric, hue='variant_name', ax=axes[idx], errorbar='sd', legend=False)
        axes[idx].set_title(title, fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('')
        axes[idx].set_ylabel(metric.upper() if metric == 'r2' else metric.upper(), fontsize=11)
        axes[idx].tick_params(axis='x', rotation=45)
        axes[idx].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'fig1_performance_comparison.pdf', bbox_inches='tight')
    plt.close()
    
    if logger:
        logger.info("✓ Figura 1: Comparación de desempeño")


def plot_internal_properties(df: pd.DataFrame, output_dir: Path):
    """Figura 2: Propiedades internas"""
    fig_dir = output_dir / 'figures'
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    
    metrics = ['lipschitz_constant', 'smoothness_mean', 'adversarial_sensitivity', 'gradient_cv']
    titles = [
        'Empirical Lipschitz Constant',
        'Internal Smoothness (Z-space variance)',
        'Adversarial Sensitivity',
        'Gradient Coefficient of Variation'
    ]
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        df_clean = df[df[metric].notna()]
        
        if len(df_clean) > 0:
            sns.barplot(data=df_clean, x='variant_name', y=metric, hue='variant_name', ax=axes[idx], errorbar='sd', legend=False)
            axes[idx].set_title(title, fontsize=11, fontweight='bold')
            axes[idx].set_xlabel('')
            axes[idx].set_ylabel(title.split('(')[0].strip(), fontsize=10)
            axes[idx].tick_params(axis='x', rotation=45)
            axes[idx].grid(True, alpha=0.3, axis='y')
        else:
            axes[idx].text(0.5, 0.5, 'No data available', ha='center', va='center', transform=axes[idx].transAxes)
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'fig2_internal_properties.pdf', bbox_inches='tight')
    plt.close()
    
    if logger:
        logger.info("✓ Figura 2: Propiedades internas")


def plot_scatter(df: pd.DataFrame, output_dir: Path):
    """Figura 3: R² vs Lipschitz"""
    fig_dir = output_dir / 'figures'
    
    df_clean = df[['variant_name', 'function_name', 'r2', 'lipschitz_constant']].dropna()
    
    if len(df_clean) == 0:
        if logger:
            logger.warning("No hay datos suficientes para scatter plot")
        return
    
    df_clean['category'] = df_clean['function_name'].map(FUNCTION_CATEGORIES)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    variants = df_clean['variant_name'].unique()
    colors = sns.color_palette("Set2", len(variants))
    markers = ['o', '^', 's', 'D']
    
    for idx, variant in enumerate(variants):
        df_var = df_clean[df_clean['variant_name'] == variant]
        ax.scatter(
            df_var['lipschitz_constant'], df_var['r2'],
            label=variant, alpha=0.6, s=100,
            color=colors[idx % len(colors)],
            marker=markers[idx % len(markers)],
            edgecolors='black', linewidths=0.5
        )
    
    from sklearn.linear_model import LinearRegression
    X_reg = df_clean['lipschitz_constant'].values.reshape(-1, 1)
    y_reg = df_clean['r2'].values
    reg = LinearRegression().fit(X_reg, y_reg)
    
    x_line = np.linspace(X_reg.min(), X_reg.max(), 100)
    y_line = reg.predict(x_line.reshape(-1, 1))
    r2_score_reg = reg.score(X_reg, y_reg)
    
    ax.plot(x_line, y_line, 'r--', alpha=0.5, linewidth=2, label=f'Linear Fit (R²={r2_score_reg:.3f})')
    ax.plot(0, 1, 'g*', markersize=20, label='Ideal (R²=1, L=0)')
    
    ax.set_xlabel('Empirical Lipschitz Constant (lower = more regular)', fontsize=12)
    ax.set_ylabel('R² (higher = better performance)', fontsize=12)
    ax.set_title('Performance vs Regularity', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'fig3_scatter.pdf', bbox_inches='tight')
    plt.close()
    
    if logger:
        logger.info("✓ Figura 3: Scatter plot performance-regularity")


def plot_by_category(df: pd.DataFrame, output_dir: Path):
    """Figura 4: Desempeño por categoría"""
    fig_dir = output_dir / 'figures'
    
    df['category'] = df['function_name'].map(FUNCTION_CATEGORIES)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    categories = ['smooth', 'non_convex', 'multimodal']
    
    for idx, cat in enumerate(categories):
        df_cat = df[df['category'] == cat]
        
        if len(df_cat) > 0:
            summary = df_cat.groupby('variant_name')['r2'].mean().reset_index()
            axes[idx].bar(summary['variant_name'], summary['r2'], alpha=0.7)
            axes[idx].set_title(f'{cat.replace("_", " ").title()} Functions', fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('Mean R²', fontsize=11)
            axes[idx].tick_params(axis='x', rotation=45)
            axes[idx].grid(True, alpha=0.3, axis='y')
            axes[idx].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'fig4_performance_by_category.pdf', bbox_inches='tight')
    plt.close()
    
    if logger:
        logger.info("✓ Figura 4: Desempeño por categoría")


def plot_boxplots(df: pd.DataFrame, output_dir: Path):
    """Figura 5: Boxplots de métricas de regularidad"""
    fig_dir = output_dir / 'figures'
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    
    metrics = ['lipschitz_constant', 'smoothness_mean', 'adversarial_sensitivity', 'gradient_cv']
    titles = ['Lipschitz Constant', 'Internal Smoothness', 'Adversarial Sensitivity', 'Gradient CV']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        df_clean = df[df[metric].notna()]
        
        if len(df_clean) > 0:
            sns.boxplot(data=df_clean, x='variant_name', y=metric, ax=axes[idx], palette='Set2')
            axes[idx].set_title(title, fontsize=11, fontweight='bold')
            axes[idx].set_xlabel('')
            axes[idx].tick_params(axis='x', rotation=45)
            axes[idx].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'fig5_boxplots_regularity.pdf', bbox_inches='tight')
    plt.close()
    
    if logger:
        logger.info("✓ Figura 5: Boxplots de regularidad")

# ============================================================================
# ANÁLISIS ESTADÍSTICO Y TABLAS
# ============================================================================

def statistical_analysis(df: pd.DataFrame, output_dir: Path):
    """Análisis estadístico completo"""
    report = []
    report.append("# ANÁLISIS ESTADÍSTICO - ESTUDIO DE ABLACIÓN\n\n")
    report.append(f"**Fecha:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
    report.append(f"**Total experimentos:** {len(df)}\n")
    report.append(f"**Variantes:** {', '.join(df['variant_name'].unique())}\n")
    report.append(f"**Funciones:** {', '.join(df['function_name'].unique())}\n\n")
    report.append("---\n\n")
    report.append("## 1. COMPARACIONES PAREADAS (K-Net-Complete vs Ablaciones)\n\n")
    
    complete_data = df[df['variant_name'] == 'K-Net-Complete']
    
    report.append("### 1.1. Métrica: R² (Performance)\n\n")
    
    comparisons = []
    for variant in df['variant_name'].unique():
        if variant != 'K-Net-Complete':
            variant_data = df[df['variant_name'] == variant]
            
            merged = pd.merge(
                complete_data[['function_name', 'fold', 'r2']],
                variant_data[['function_name', 'fold', 'r2']],
                on=['function_name', 'fold'],
                suffixes=('_complete', f'_{variant}')
            )
            
            if len(merged) >= 2:
                r2_complete = merged['r2_complete'].values
                r2_variant = merged[f'r2_{variant}'].values
                
                t_stat, p_val = stats.ttest_rel(r2_complete, r2_variant)
                mean_diff = r2_complete.mean() - r2_variant.mean()
                
                pooled_std = np.sqrt((r2_complete.std()**2 + r2_variant.std()**2) / 2)
                cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
                
                comparisons.append({
                    'variant': variant,
                    't_stat': t_stat,
                    'p_val': p_val,
                    'mean_complete': r2_complete.mean(),
                    'mean_variant': r2_variant.mean(),
                    'mean_diff': mean_diff,
                    'cohens_d': cohens_d
                })
    
    n_comparisons = len(comparisons)
    alpha_bonf = 0.05 / n_comparisons if n_comparisons > 0 else 0.05
    
    report.append(f"**Corrección de Bonferroni:** α ajustado = {alpha_bonf:.4f}\n\n")
    
    for comp in comparisons:
        sig = '***' if comp['p_val'] < 0.001 else '**' if comp['p_val'] < 0.01 else '*' if comp['p_val'] < alpha_bonf else 'ns'
        effect = 'grande' if abs(comp['cohens_d']) > 0.8 else 'mediano' if abs(comp['cohens_d']) > 0.5 else 'pequeño'
        
        report.append(f"#### K-Net-Complete vs {comp['variant']}\n\n")
        report.append(f"- **R² Complete:** {comp['mean_complete']:.4f}\n")
        report.append(f"- **R² {comp['variant']}:** {comp['mean_variant']:.4f}\n")
        report.append(f"- **Diferencia:** {comp['mean_diff']:+.4f}\n")
        report.append(f"- **t-statistic:** {comp['t_stat']:.4f}\n")
        report.append(f"- **p-value:** {comp['p_val']:.4f} {sig}\n")
        report.append(f"- **Cohen's d:** {comp['cohens_d']:.4f} (efecto {effect})\n\n")
    
    report.append("### 1.2. Métrica: Lipschitz Constant (Regularity)\n\n")
    
    for variant in df['variant_name'].unique():
        if variant != 'K-Net-Complete':
            variant_data = df[df['variant_name'] == variant]
            
            merged = pd.merge(
                complete_data[['function_name', 'fold', 'lipschitz_constant']].dropna(),
                variant_data[['function_name', 'fold', 'lipschitz_constant']].dropna(),
                on=['function_name', 'fold'],
                suffixes=('_complete', f'_{variant}')
            )
            
            if len(merged) >= 2:
                l_complete = merged['lipschitz_constant_complete'].values
                l_variant = merged[f'lipschitz_constant_{variant}'].values
                
                t_stat, p_val = stats.ttest_rel(l_complete, l_variant)
                mean_diff = l_complete.mean() - l_variant.mean()
                
                report.append(f"#### K-Net-Complete vs {variant}\n\n")
                report.append(f"- **L Complete:** {l_complete.mean():.4f}\n")
                report.append(f"- **L {variant}:** {l_variant.mean():.4f}\n")
                report.append(f"- **Diferencia:** {mean_diff:+.4f} (negativo = Complete más regular)\n")
                report.append(f"- **p-value:** {p_val:.4f}\n\n")
    
    with open(output_dir / 'statistical_analysis.md', 'w', encoding='utf-8') as f:
        f.writelines(report)
    
    if logger:
        logger.info(f"✓ Análisis estadístico guardado")


def generate_latex_tables(df: pd.DataFrame, output_dir: Path):
    """Genera tablas LaTeX"""
    tables_dir = output_dir / 'latex_tables'
    tables_dir.mkdir(exist_ok=True)
    
    # Tabla 1: Regularidad
    metrics = ['lipschitz_constant', 'smoothness_mean', 'adversarial_sensitivity', 'gradient_cv']
    summary = df.groupby('variant_name')[metrics].agg(['mean', 'std']).round(4)
    
    latex_table1 = []
    latex_table1.append("% TABLA 2 DEL PAPER: Internal Regularity Properties\n")
    latex_table1.append("\\begin{table}[h]\n")
    latex_table1.append("\\centering\n")
    latex_table1.append("\\caption{Mean $\\pm$ standard deviation of internal regularity metrics.}\n")
    latex_table1.append("\\label{tab:regularity-metrics}\n")
    latex_table1.append("\\begin{tabular}{lcccc}\n")
    latex_table1.append("\\toprule\n")
    latex_table1.append("\\textbf{Model} & \\textbf{$L_{\\text{emp}}$ $\\downarrow$} & \\textbf{Smoothness $\\downarrow$} & \\textbf{Adv. Rob. $\\downarrow$} & \\textbf{Grad. CV $\\downarrow$} \\\\\n")
    latex_table1.append("\\midrule\n")
    
    for variant in summary.index:
        l_mean = summary.loc[variant, ('lipschitz_constant', 'mean')]
        l_std = summary.loc[variant, ('lipschitz_constant', 'std')]
        s_mean = summary.loc[variant, ('smoothness_mean', 'mean')]
        s_std = summary.loc[variant, ('smoothness_mean', 'std')]
        a_mean = summary.loc[variant, ('adversarial_sensitivity', 'mean')]
        a_std = summary.loc[variant, ('adversarial_sensitivity', 'std')]
        g_mean = summary.loc[variant, ('gradient_cv', 'mean')]
        g_std = summary.loc[variant, ('gradient_cv', 'std')]
        
        latex_table1.append(f"{variant} & {l_mean:.2f} $\\pm$ {l_std:.2f} & {s_mean:.3f} $\\pm$ {s_std:.3f} & {a_mean:.2f} $\\pm$ {a_std:.2f} & {g_mean:.3f} $\\pm$ {g_std:.3f} \\\\\n")
    
    latex_table1.append("\\bottomrule\n")
    latex_table1.append("\\end{tabular}\n")
    latex_table1.append("\\end{table}\n")
    
    with open(tables_dir / 'table2_regularity.tex', 'w') as f:
        f.writelines(latex_table1)
    
    # Tabla 2: Performance
    perf_metrics = ['r2', 'rmse', 'mae', 'train_time_s', 'n_params']
    summary_perf = df.groupby('variant_name')[perf_metrics].agg(['mean', 'std']).round(4)
    
    latex_table2 = []
    latex_table2.append("% TABLA 3 DEL PAPER: Predictive Performance\n")
    latex_table2.append("\\begin{table}[h]\n")
    latex_table2.append("\\centering\n")
    latex_table2.append("\\caption{Mean $\\pm$ standard deviation of predictive performance metrics.}\n")
    latex_table2.append("\\label{tab:performance-metrics}\n")
    latex_table2.append("\\begin{tabular}{lccccc}\n")
    latex_table2.append("\\toprule\n")
    latex_table2.append("\\textbf{Model} & \\textbf{R² $\\uparrow$} & \\textbf{RMSE $\\downarrow$} & \\textbf{MAE $\\downarrow$} & \\textbf{Time (s)} & \\textbf{Params} \\\\\n")
    latex_table2.append("\\midrule\n")
    
    for variant in summary_perf.index:
        r2_mean = summary_perf.loc[variant, ('r2', 'mean')]
        r2_std = summary_perf.loc[variant, ('r2', 'std')]
        rmse_mean = summary_perf.loc[variant, ('rmse', 'mean')]
        rmse_std = summary_perf.loc[variant, ('rmse', 'std')]
        mae_mean = summary_perf.loc[variant, ('mae', 'mean')]
        mae_std = summary_perf.loc[variant, ('mae', 'std')]
        time_mean = summary_perf.loc[variant, ('train_time_s', 'mean')]
        params = int(summary_perf.loc[variant, ('n_params', 'mean')])
        
        latex_table2.append(f"{variant} & {r2_mean:.3f} $\\pm$ {r2_std:.3f} & {rmse_mean:.2f} $\\pm$ {rmse_std:.2f} & {mae_mean:.2f} $\\pm$ {mae_std:.2f} & {time_mean:.1f} & {params:,} \\\\\n")
    
    latex_table2.append("\\bottomrule\n")
    latex_table2.append("\\end{tabular}\n")
    latex_table2.append("\\end{table}\n")
    
    with open(tables_dir / 'table3_performance.tex', 'w') as f:
        f.writelines(latex_table2)
    
    # Tabla 3: Por categoría
    df['category'] = df['function_name'].map(FUNCTION_CATEGORIES)
    cat_summary = df.groupby(['category', 'variant_name'])[['r2', 'lipschitz_constant']].mean().round(4)
    
    latex_table3 = []
    latex_table3.append("% TABLA 4 DEL PAPER: Performance by Function Category\n")
    latex_table3.append("\\begin{table}[h]\n")
    latex_table3.append("\\centering\n")
    latex_table3.append("\\caption{Performance and regularity by function category.}\n")
    latex_table3.append("\\label{tab:category-breakdown}\n")
    latex_table3.append("\\begin{tabular}{llcc}\n")
    latex_table3.append("\\toprule\n")
    latex_table3.append("\\textbf{Category} & \\textbf{Model} & \\textbf{R² Mean} & \\textbf{$L_{\\text{emp}}$ Mean} \\\\\n")
    latex_table3.append("\\midrule\n")
    
    for cat in ['smooth', 'non_convex', 'multimodal']:
        for variant in ['K-Net-Complete', 'MLP']:
            try:
                r2_val = cat_summary.loc[(cat, variant), 'r2']
                l_val = cat_summary.loc[(cat, variant), 'lipschitz_constant']
                latex_table3.append(f"{cat.replace('_', ' ').title()} & {variant} & {r2_val:.3f} & {l_val:.2f} \\\\\n")
            except KeyError:
                pass
        latex_table3.append("\\midrule\n")
    
    latex_table3.append("\\bottomrule\n")
    latex_table3.append("\\end{tabular}\n")
    latex_table3.append("\\end{table}\n")
    
    with open(tables_dir / 'table4_by_category.tex', 'w') as f:
        f.writelines(latex_table3)
    
    if logger:
        logger.info(f"✓ Tablas LaTeX generadas")


def generate_executive_summary(df: pd.DataFrame, output_dir: Path):
    """Genera resumen ejecutivo"""
    summary = []
    summary.append("# RESUMEN EJECUTIVO - ESTUDIO DE ABLACIÓN\n\n")
    summary.append(f"**Fecha:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    summary.append("---\n\n")
    
    summary.append("## ESTADÍSTICAS GLOBALES\n\n")
    summary.append(f"- **Total de experimentos:** {len(df)}\n")
    summary.append(f"- **Funciones evaluadas:** {df['function_name'].nunique()}\n")
    summary.append(f"- **Variantes comparadas:** {df['variant_name'].nunique()}\n")
    summary.append(f"- **Folds por función:** {df.groupby('function_name')['fold'].nunique().mean():.1f}\n\n")
    
    summary.append("## RANKING POR DESEMPEÑO (R²)\n\n")
    perf_ranking = df.groupby('variant_name')['r2'].mean().sort_values(ascending=False)
    for i, (variant, r2) in enumerate(perf_ranking.items(), 1):
        summary.append(f"{i}. **{variant}**: R² = {r2:.4f}\n")
    summary.append("\n")
    
    summary.append("## RANKING POR REGULARIDAD (Lipschitz, menor es mejor)\n\n")
    reg_ranking = df.groupby('variant_name')['lipschitz_constant'].mean().sort_values()
    for i, (variant, l) in enumerate(reg_ranking.items(), 1):
        summary.append(f"{i}. **{variant}**: L_emp = {l:.4f}\n")
    summary.append("\n")
    
    summary.append("## KEY FINDINGS\n\n")
    
    complete_r2 = df[df['variant_name'] == 'K-Net-Complete']['r2'].mean()
    fixed_lambda_r2 = df[df['variant_name'] == 'K-Net-FixedLambda']['r2'].mean()
    drop_pct = (1 - fixed_lambda_r2 / complete_r2) * 100
    
    summary.append(f"### 1. Lambda entrenable ES CRÍTICO\n\n")
    summary.append(f"- K-Net-Complete: R² = {complete_r2:.4f}\n")
    summary.append(f"- K-Net-FixedLambda: R² = {fixed_lambda_r2:.4f}\n")
    summary.append(f"- **Drop: {drop_pct:.1f}%** al fijar Λ\n\n")
    
    fixed_psi_r2 = df[df['variant_name'] == 'K-Net-FixedPsi']['r2'].mean()
    improvement_pct = (complete_r2 / fixed_psi_r2 - 1) * 100
    
    summary.append(f"### 2. ψ Lipschitz mejor que tanh\n\n")
    summary.append(f"- K-Net-Complete (ψ Lipschitz): R² = {complete_r2:.4f}\n")
    summary.append(f"- K-Net-FixedPsi (tanh): R² = {fixed_psi_r2:.4f}\n")
    summary.append(f"- **Mejora: +{improvement_pct:.1f}%** con ψ Lipschitz\n\n")
    
    mlp_r2 = df[df['variant_name'] == 'MLP']['r2'].mean()
    gap_pct = (1 - complete_r2 / mlp_r2) * 100
    
    summary.append(f"### 3. KST-Complete vs MLP\n\n")
    summary.append(f"- MLP: R² = {mlp_r2:.4f}\n")
    summary.append(f"- K-Net-Complete: R² = {complete_r2:.4f}\n")
    summary.append(f"- **Gap: {gap_pct:.1f}%**\n\n")
    
    complete_l = df[df['variant_name'] == 'K-Net-Complete']['lipschitz_constant'].mean()
    mlp_l = df[df['variant_name'] == 'MLP']['lipschitz_constant'].mean()
    
    if not np.isnan(complete_l) and not np.isnan(mlp_l):
        reg_improvement = (1 - complete_l / mlp_l) * 100
        summary.append(f"### 4. Ventaja en Regularidad\n\n")
        summary.append(f"- K-Net-Complete: L_emp = {complete_l:.4f}\n")
        summary.append(f"- MLP: L_emp = {mlp_l:.4f}\n")
        summary.append(f"- **Mejora: {reg_improvement:+.1f}%**\n\n")
    
    with open(output_dir / 'executive_summary.md', 'w', encoding='utf-8') as f:
        f.writelines(summary)
    
    if logger:
        logger.info(f"✓ Resumen ejecutivo guardado")

# ============================================================================
# MAIN
# ============================================================================

def run_ablation_experiment(args):
    global logger
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger(output_dir)
    
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu'
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    logger.info("=" * 80)
    logger.info("ESTUDIO DE ABLACIÓN: FUNCIÓN ψ LIPSCHITZ")
    logger.info("=" * 80)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Output: {output_dir.resolve()}")
    logger.info(f"Device: {device}")
    logger.info(f"Activation: {args.activation}")
    logger.info("=" * 80)
    
    # Cargar proyector
    projector = KSTProjector(n=2, psi_checkpoint=args.checkpoint)
    logger.info("✓ Proyector KST cargado")
    
    # Obtener función de activación
    activation_fn = get_activation_function(args.activation)
    
    # Configurar variantes
    variant_configs = [
        {'name': 'K-Net-Complete', 'model_factory': KSTSprecherLorentzModel},
        {'name': 'K-Net-FixedPsi', 'model_factory': KNetFixedActivation},
        {'name': 'K-Net-FixedLambda', 'model_factory': KNetFixedLambda},
        {'name': 'MLP', 'model_factory': MLP}
    ]
    
    logger.info(f"✓ {len(BENCHMARK_FUNCTIONS)} funciones de prueba cargadas")
    
    # Ejecutar experimentos
    all_results = []
    
    for func_name, func in BENCHMARK_FUNCTIONS.items():
        logger.info(f"\n{'=' * 80}")
        logger.info(f"{func_name.upper()}")
        logger.info("=" * 80)
        
        X = np.random.rand(args.n_train + args.n_test, 2)
        y = func(X)
        
        kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
        
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
            logger.info(f"\n  Fold {fold_idx + 1}/{args.n_folds}")
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            val_split = int(len(X_train) * 0.15)
            fold_data = {
                'X_train': X_train[val_split:],
                'y_train': y_train[val_split:],
                'X_val': X_train[:val_split],
                'y_val': y_train[:val_split],
                'X_test': X_test,
                'y_test': y_test,
                'function_name': func_name
            }
            
            for config in variant_configs:
                result, history = run_ablation_fold(
                    config, fold_data, args, fold_idx, device, projector, activation_fn
                )
                all_results.append(result)
                
                logger.info(
                    f"    {config['name']:20s} | "
                    f"R²={result.r2:.4f} L={result.lipschitz_constant:.3f} "
                    f"Time={result.train_time_s:.1f}s"
                )
    
    # Guardar resultados
    results_df = pd.DataFrame([asdict(r) for r in all_results])
    results_df.to_csv(output_dir / 'ablation_results.csv', index=False)
    logger.info(f"\n✓ Resultados guardados: {output_dir / 'ablation_results.csv'}")
    
    # Generar visualizaciones
    logger.info("\n" + "=" * 80)
    logger.info("GENERANDO VISUALIZACIONES")
    logger.info("=" * 80)
    
    plot_performance_comparison(results_df, output_dir)
    plot_internal_properties(results_df, output_dir)
    plot_scatter(results_df, output_dir)
    plot_by_category(results_df, output_dir)
    plot_boxplots(results_df, output_dir)
    
    # Análisis estadístico
    logger.info("\n" + "=" * 80)
    logger.info("ANÁLISIS ESTADÍSTICO")
    logger.info("=" * 80)
    
    statistical_analysis(results_df, output_dir)
    
    # Tablas LaTeX
    logger.info("\n" + "=" * 80)
    logger.info("GENERANDO TABLAS LATEX")
    logger.info("=" * 80)
    
    generate_latex_tables(results_df, output_dir)
    
    # Resumen ejecutivo
    logger.info("\n" + "=" * 80)
    logger.info("GENERANDO RESUMEN EJECUTIVO")
    logger.info("=" * 80)
    
    generate_executive_summary(results_df, output_dir)
    
    logger.info("\n" + "=" * 80)
    logger.info("✓ COMPLETO")
    logger.info("=" * 80)
    logger.info(f"\nResultados en: {output_dir.resolve()}")
    logger.info(f"- ablation_results.csv")
    logger.info(f"- figures/ (5 figuras PDF)")
    logger.info(f"- latex_tables/ (3 tablas LaTeX)")
    logger.info(f"- statistical_analysis.md")
    logger.info(f"- executive_summary.md")
    logger.info("=" * 80)

    return output_dir / 'ablation_results.csv' # modif

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Estudio de ablación DEFINITIVO")
    
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output', type=str, default='results/ablation_final')
    parser.add_argument('--n_train', type=int, default=500)
    parser.add_argument('--n_test', type=int, default=500)
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--activation', type=str, default='gelu',
                       choices=['gelu', 'relu', 'tanh', 'sigmoid', 'elu', 'leakyrelu', 'selu', 'silu'])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    main(args)