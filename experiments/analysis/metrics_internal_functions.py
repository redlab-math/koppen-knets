"""
Módulo de métricas de funciones internas para K-nets.
Extraído de ablation.py para benchmark.py
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors

def compute_lipschitz_constant(
    model: nn.Module,
    X: torch.Tensor,
    num_samples: int = 500,
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    Estima la constante de Lipschitz empírica del modelo.
    L_emp = percentile_95(||f(x_i) - f(x_j)|| / ||x_i - x_j||)
    
    (Función extraída de ablation.py)
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
    
    (Función extraída de ablation.py)
    """
    model.eval()
    X = X.to(device)
    
    # Intentar obtener representación interna
    Z = None
    if hasattr(model, 'get_projection_contributions'): # KNetFixedLambda
        with torch.no_grad():
            Z = model.get_projection_contributions(X).cpu().numpy()
    elif hasattr(model, '_apply_trainable_lambdas'): # KSTSprecherLorentzModel
        with torch.no_grad():
            Z = model._apply_trainable_lambdas(X).cpu().numpy()
    
    if Z is None:
        # Fallback para MLP: usar la salida de la penúltima capa si es posible
        # O simplemente usar la salida final si no
        try:
            # Asumiendo que 'model.network' es el Sequential
            penultimate_layer_output = [None]
            def hook(module, input, output):
                penultimate_layer_output[0] = output.cpu().numpy()
            
            # Asumir que la penúltima capa es la antepenúltima del Sequential (antes de Linear y Dropout)
            hook_handle = model.network[-2].register_forward_hook(hook)
            with torch.no_grad():
                model(X)
            hook_handle.remove()
            Z = penultimate_layer_output[0]
            if Z is None:
                 raise Exception("Hook falló")
        except Exception:
             # Fallback final: usar salida del modelo (menos ideal)
            with torch.no_grad():
                Z = model(X).cpu().numpy()

    X_np = X.cpu().numpy()
    
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
    
    (Función extraída de ablation.py)
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
    
    (Función extraída de ablation.py)
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
        'cv': float(cv),
        'trend_slope': float(trend_slope),
        'vanishing_ratio': float(vanishing_ratio)
    }