#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Benchmark: k-net vs Baselines (MLP, GP)
================================================================================
Con métricas de funciones internas:
- Constante de Lipschitz empírica
- Suavidad de representaciones internas
- Robustez adversarial
- Estabilidad de gradientes

Además de las métricas tradicionales de desempeño (R², RMSE, MAE)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import argparse
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from matplotlib.gridspec import GridSpec
import seaborn as sns

from src.knet_core.kst_projector import KSTProjector
from src.knet_core.kst_sl_model import KSTSprecherLorentzModel

# ============================================================================
# IMPORTAR MÓDULO DE MÉTRICAS INTERNAS
# ============================================================================
sys.path.insert(0, str(Path(__file__).resolve().parent / "analysis"))
from metrics_internal_functions import (
    compute_lipschitz_constant,
    measure_internal_smoothness,
    adversarial_robustness,
    track_gradient_stability
)

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

mpl.rcParams.update({
    'figure.dpi': 300, 'savefig.dpi': 300,
    'font.family': 'serif', 'font.size': 10
})
sns.set_style("whitegrid")
sns.set_palette("colorblind")

def setup_logging(output_dir: Path) -> logging.Logger:
    log_dir = output_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('Benchmark')
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    fh = logging.FileHandler(log_dir / f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ch = logging.StreamHandler(sys.stdout)
    for h in [fh, ch]:
        h.setFormatter(logging.Formatter('%(levelname)s | %(message)s'))
        logger.addHandler(h)
    return logger

# ============================================================================
# MAPEO DE FUNCIONES DE ACTIVACIÓN
# ============================================================================

def get_activation_function(name: str) -> Callable:
    """Mapea nombres de activación a clases/funciones de PyTorch."""
    activation_map = {
        'gelu': nn.GELU,
        'relu': nn.ReLU,
        'tanh': nn.Tanh,
        'sigmoid': nn.Sigmoid,
        'elu': nn.ELU,
        'leakyrelu': nn.LeakyReLU,
        'selu': nn.SELU,
        'silu': nn.SiLU,
    }
    
    name_lower = name.lower()
    if name_lower not in activation_map:
        raise ValueError(
            f"Activación '{name}' no reconocida. "
            f"Opciones disponibles: {list(activation_map.keys())}"
        )
    
    return activation_map[name_lower]

# ============================================================================
# FUNCIONES DE BENCHMARK (6 FUNCIONES)
# ============================================================================

@dataclass
class BenchmarkFunction:
    name: str
    func: Callable
    dim: int = 2
    category: str = 'smooth'  # 'smooth', 'multimodal', 'non_convex'

def branin_2d(X: np.ndarray) -> np.ndarray:
    x1, x2 = X[:, 0], X[:, 1]
    x1_mapped, x2_mapped = 15 * x1 - 5, 15 * x2
    a, b, c, r, s, t = 1, 5.1/(4*np.pi**2), 5/np.pi, 6, 10, 1/(8*np.pi)
    return a * (x2_mapped - b * x1_mapped**2 + c * x1_mapped - r)**2 + s * (1 - t) * np.cos(x1_mapped) + s

def rosenbrock_2d(X: np.ndarray) -> np.ndarray:
    x1, x2 = X[:, 0], X[:, 1]
    x1_mapped, x2_mapped = 4 * x1 - 2, 4 * x2 - 2
    return 100 * (x2_mapped - x1_mapped**2)**2 + (1 - x1_mapped)**2

def rastrigin_2d(X: np.ndarray) -> np.ndarray:
    x1, x2 = X[:, 0], X[:, 1]
    x1_mapped, x2_mapped = 10.24 * x1 - 5.12, 10.24 * x2 - 5.12
    A = 10
    return 2*A + x1_mapped**2 - A*np.cos(2*np.pi*x1_mapped) + x2_mapped**2 - A*np.cos(2*np.pi*x2_mapped)

def ackley_2d(X: np.ndarray) -> np.ndarray:
    x1, x2 = X[:, 0], X[:, 1]
    x1_mapped, x2_mapped = 10 * x1 - 5, 10 * x2 - 5
    return (-20*np.exp(-0.2*np.sqrt(0.5*(x1_mapped**2 + x2_mapped**2))) 
            - np.exp(0.5*(np.cos(2*np.pi*x1_mapped) + np.cos(2*np.pi*x2_mapped))) + np.e + 20)

def quadratic_2d(X: np.ndarray) -> np.ndarray:
    """Nueva: Función cuadrática simple (convexa, suave)"""
    x1, x2 = X[:, 0] - 0.5, X[:, 1] - 0.5
    return x1**2 + x2**2

def six_hump_camel(X: np.ndarray) -> np.ndarray:
    """Nueva: Six-Hump Camel (6 mínimos locales, regiones planas)"""
    x1, x2 = X[:, 0]*6-3, X[:, 1]*4-2
    return (4 - 2.1*x1**2 + x1**4/3)*x1**2 + x1*x2 + (-4 + 4*x2**2)*x2**2

def get_benchmark_functions() -> Dict[str, BenchmarkFunction]:
    return {
        'quadratic': BenchmarkFunction('Quadratic', quadratic_2d, category='smooth'),
        'branin': BenchmarkFunction('Branin', branin_2d, category='smooth'),
        'rosenbrock': BenchmarkFunction('Rosenbrock', rosenbrock_2d, category='non_convex'),
        'rastrigin': BenchmarkFunction('Rastrigin', rastrigin_2d, category='multimodal'),
        'ackley': BenchmarkFunction('Ackley', ackley_2d, category='multimodal'),
        'six_hump': BenchmarkFunction('SixHumpCamel', six_hump_camel, category='non_convex'),
    }

# ============================================================================
# MODELOS
# ============================================================================

class MLP(nn.Module):
    def __init__(self, n_inputs: int, hidden_sizes: List[int], dropout: float, activation: Callable = nn.GELU):
        super().__init__()
        layers = []
        sizes = [n_inputs] + hidden_sizes + [1]
        for i in range(len(sizes) - 2):
            layers.extend([nn.Linear(sizes[i], sizes[i+1]), activation(), nn.Dropout(dropout)])
        layers.append(nn.Linear(sizes[-2], sizes[-1]))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

# ============================================================================
# ESTRUCTURA DE RESULTADOS EXTENDIDA
# ============================================================================

@dataclass
class FoldResult:
    """Resultados extendidos con métricas de funciones internas"""
    # Identificación
    model_name: str
    function_name: str
    fold: int
    activation: str
    
    # Métricas de desempeño tradicionales
    r2: float
    rmse: float
    mae: float
    train_time_s: float
    n_params: int
    final_train_loss: float
    best_val_loss: float
    n_epochs: int
    
    # NUEVAS: Métricas de funciones internas
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

# ============================================================================
# ENTRENAMIENTO CON TRACKING DE GRADIENTES
# ============================================================================

def train_pytorch_model(model: nn.Module, X_train, y_train, X_val, y_val, args, device):
    """Entrenamiento con registro de normas de gradiente"""
    y_scaler = StandardScaler().fit(y_train.reshape(-1, 1))
    y_train_scaled = y_scaler.transform(y_train.reshape(-1, 1))
    y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1))
    
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train_scaled).float()),
        batch_size=args.batch_size, shuffle=True
    )
    
    X_val_t = torch.from_numpy(X_val).float().to(device)
    y_val_t = torch.from_numpy(y_val_scaled).float().to(device)
    
    # Learning rate diferencial para KST-SL
    if hasattr(model, 'lambda_matrix'):
        optimizer = torch.optim.AdamW([
            {'params': [model.lambda_matrix], 'lr': args.learning_rate * 0.5, 'weight_decay': 0.0},
            {'params': model.outer_net.parameters(), 'lr': args.learning_rate, 'weight_decay': 1e-5}
        ])
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=args.patience // 2, factor=0.5, verbose=False)
    loss_fn = nn.MSELoss()
    
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'lr': [], 'grad_norms': []}  # ← NUEVO
    
    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0
        epoch_grad_norms = []
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            
            # Gradient clipping
            if hasattr(model, 'lambda_matrix'):
                torch.nn.utils.clip_grad_norm_(model.lambda_matrix, max_norm=0.5)
                torch.nn.utils.clip_grad_norm_(model.outer_net.parameters(), max_norm=1.0)
                
                # Registrar norma de gradiente de λ
                if model.lambda_matrix.grad is not None:
                    grad_norm = torch.norm(model.lambda_matrix.grad).item()
                    epoch_grad_norms.append(grad_norm)
            else:
                # Para MLP, registrar norma total de gradientes
                total_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
                epoch_grad_norms.append(total_norm)
            
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        history['grad_norms'].append(np.mean(epoch_grad_norms) if epoch_grad_norms else 0.0)
        
        # Validation
        model.eval()
        with torch.no_grad():
            y_val_pred = model(X_val_t)
            val_loss = loss_fn(y_val_pred, y_val_t).item()
        
        history['val_loss'].append(val_loss)
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= args.patience:
            break
    
    return {'scaler': y_scaler, 'history': history}

# ============================================================================
# EJECUTAR UN FOLD CON MÉTRICAS EXTENDIDAS
# ============================================================================

def run_fold(model_config, data, args, fold_idx, device, projector=None, logger=None):
    """Ejecuta un fold y calcula TODAS las métricas (tradicionales + internas)"""
    model_name = model_config['name']
    X_train, y_train = data['X_train'], data['y_train']
    X_test, y_test = data['X_test'], data['y_test']
    
    start_time = time.time()
    
    # ========================================================================
    # ENTRENAR MODELO
    # ========================================================================
    if model_name == 'KST-SL':
        activation_fn = get_activation_function(args.activation)
        model = KSTSprecherLorentzModel(
            projector=projector,
            hidden_sizes=model_config['hidden_sizes'],
            dropout=model_config['dropout'],
            activation=activation_fn
        ).to(device)
        train_out = train_pytorch_model(model, X_train, y_train, data['X_val'], data['y_val'], args, device)
        
        model.eval()
        with torch.no_grad():
            y_pred = train_out['scaler'].inverse_transform(
                model(torch.from_numpy(X_test).float().to(device)).cpu().numpy()
            ).ravel()
    
    elif model_name == 'MLP':
        activation_fn = get_activation_function(args.activation)
        model = MLP(X_train.shape[1], model_config['hidden_sizes'], model_config['dropout'], activation_fn).to(device)
        train_out = train_pytorch_model(model, X_train, y_train, data['X_val'], data['y_val'], args, device)
        
        model.eval()
        with torch.no_grad():
            y_pred = train_out['scaler'].inverse_transform(
                model(torch.from_numpy(X_test).float().to(device)).cpu().numpy()
            ).ravel()
    
    else:  # GP
        model = GaussianProcessRegressor(
            kernel=ConstantKernel(1.0) * Matern(nu=2.5),
            n_restarts_optimizer=10, alpha=1e-6, random_state=args.seed
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        train_out = {'history': {'train_loss': [], 'val_loss': [], 'grad_norms': []}}
    
    train_time = time.time() - start_time
    n_params = sum(p.numel() for p in model.parameters()) if isinstance(model, nn.Module) else 0
    
    # ========================================================================
    # CALCULAR MÉTRICAS TRADICIONALES
    # ========================================================================
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    # ========================================================================
    # CALCULAR MÉTRICAS DE FUNCIONES INTERNAS
    # ========================================================================
    internal_metrics = {}
    
    if isinstance(model, nn.Module):  # Solo para modelos neuronales
        if logger:
            logger.info(f"  └─ Calculando métricas internas para {model_name}...")
        
        X_test_tensor = torch.from_numpy(X_test).float()
        
        try:
            # Lipschitz
            lipschitz = compute_lipschitz_constant(
                model, X_test_tensor, num_samples=min(500, len(X_test)), device=device
            )
            internal_metrics.update({
                'lipschitz_constant': lipschitz['lipschitz_constant'],
                'lipschitz_mean': lipschitz['lipschitz_mean'],
                'lipschitz_std': lipschitz['lipschitz_std']
            })
        except Exception as e:
            if logger:
                logger.warning(f"  └─ Error en Lipschitz: {e}")
        
        try:
            # Smoothness
            smoothness = measure_internal_smoothness(
                model, X_test_tensor, k_neighbors=min(10, len(X_test)//2), device=device
            )
            internal_metrics.update({
                'smoothness_mean': smoothness['smoothness_mean'],
                'smoothness_median': smoothness['smoothness_median'],
                'smoothness_std': smoothness['smoothness_std']
            })
        except Exception as e:
            if logger:
                logger.warning(f"  └─ Error en Smoothness: {e}")
        
        try:
            # Robustez adversarial
            robustness = adversarial_robustness(
                model, X_test_tensor, epsilon=0.01, num_samples=min(100, len(X_test)), device=device
            )
            internal_metrics.update({
                'adversarial_sensitivity': robustness['adversarial_sensitivity'],
                'output_change_mean': robustness['output_change_mean'],
                'relative_error': robustness['relative_error']
            })
        except Exception as e:
            if logger:
                logger.warning(f"  └─ Error en Robustness: {e}")
        
        try:
            # Estabilidad de gradientes
            if train_out['history']['grad_norms']:
                grad_stability = track_gradient_stability(train_out['history']['grad_norms'])
                internal_metrics.update({
                    'gradient_cv': grad_stability['cv'],
                    'gradient_trend': grad_stability['trend_slope'],
                    'vanishing_ratio': grad_stability['vanishing_ratio']
                })
        except Exception as e:
            if logger:
                logger.warning(f"  └─ Error en Gradient Stability: {e}")
    
    # ========================================================================
    # CREAR RESULTADO
    # ========================================================================
    result = FoldResult(
        model_name=model_name,
        function_name=data['function_name'],
        fold=fold_idx,
        activation=args.activation if model_name != 'GP' else 'N/A',
        r2=r2,
        rmse=rmse,
        mae=mae,
        train_time_s=train_time,
        n_params=n_params,
        final_train_loss=train_out['history']['train_loss'][-1] if train_out['history']['train_loss'] else 0,
        best_val_loss=min(train_out['history']['val_loss']) if train_out['history']['val_loss'] else 0,
        n_epochs=len(train_out['history']['train_loss']),
        **internal_metrics  # Agregar métricas internas (con defaults NaN si no se calcularon)
    )
    
    return result, train_out

# ============================================================================
# VISUALIZACIONES
# ============================================================================

def plot_all_results(results_df, histories, funcs_dict, output_dir, logger):
    """Genera figuras tradicionales + 3 nuevas figuras de métricas internas"""
    fig_dir = output_dir / 'figures'
    fig_dir.mkdir(exist_ok=True)
    
    # ========================================================================
    # FIGURAS (5)
    # ========================================================================
    
    # FIGURA 1: Performance Comparativo (R² y RMSE)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for i, metric in enumerate(['r2', 'rmse']):
        sns.barplot(data=results_df, x='function_name', y=metric, hue='model_name', ax=axes[i], errorbar='sd')
        axes[i].set_title(f'{metric.upper()} por Función')
        axes[i].set_xlabel('')
        axes[i].set_xticklabels([t.get_text().capitalize() for t in axes[i].get_xticklabels()], rotation=45)
        if i > 0: axes[i].get_legend().remove()
    plt.tight_layout()
    plt.savefig(fig_dir / 'fig1_performance_comparison.pdf')
    plt.close()
    logger.info("✓ Figura 1: Comparación de performance")
    
    # FIGURA 2: Learning Curves
    n_funcs = len(histories)
    fig, axes = plt.subplots(2, (n_funcs+1)//2, figsize=(18, 10))
    axes = axes.ravel()
    for idx, (func_name, hist) in enumerate(histories.items()):
        for model_name in ['KST-SL', 'MLP']:
            if model_name in hist and hist[model_name]['train_loss']:
                epochs = range(1, len(hist[model_name]['train_loss']) + 1)
                axes[idx].plot(epochs, hist[model_name]['train_loss'], label=f'{model_name} (train)', alpha=0.7)
                if hist[model_name]['val_loss']:
                    axes[idx].plot(epochs, hist[model_name]['val_loss'], 
                                 label=f'{model_name} (val)', linestyle='--', alpha=0.7)
        axes[idx].set_xlabel('Época')
        axes[idx].set_ylabel('MSE Loss')
        axes[idx].set_title(func_name.capitalize())
        axes[idx].set_yscale('log')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / 'fig2_learning_curves.pdf')
    plt.close()
    logger.info("✓ Figura 2: Curvas de aprendizaje")
    
    # FIGURA 3: Box plots de errores
    fig, axes = plt.subplots(2, (n_funcs+1)//2, figsize=(18, 10))
    axes = axes.ravel()
    for idx, func in enumerate(results_df['function_name'].unique()):
        df_func = results_df[results_df['function_name'] == func]
        sns.boxplot(data=df_func, x='model_name', y='rmse', ax=axes[idx])
        axes[idx].set_title(f'{func.capitalize()} - Distribución RMSE')
        axes[idx].set_xlabel('')
        axes[idx].set_ylabel('RMSE')
    plt.tight_layout()
    plt.savefig(fig_dir / 'fig3_error_distributions.pdf')
    plt.close()
    logger.info("✓ Figura 3: Distribuciones de error")
    
    # FIGURA 4: Tiempo y parámetros
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    df_agg = results_df.groupby(['function_name', 'model_name']).agg({
        'train_time_s': 'mean', 'n_params': 'first'
    }).reset_index()
    sns.barplot(data=df_agg, x='function_name', y='train_time_s', hue='model_name', ax=axes[0])
    axes[0].set_title('Tiempo de Entrenamiento (s)')
    axes[0].set_xlabel('')
    axes[0].set_xticklabels([t.get_text().capitalize() for t in axes[0].get_xticklabels()], rotation=45)
    
    df_params = df_agg[df_agg['n_params'] > 0].drop_duplicates('model_name')
    axes[1].bar(df_params['model_name'], df_params['n_params'])
    axes[1].set_title('Número de Parámetros')
    axes[1].set_ylabel('Parámetros')
    plt.tight_layout()
    plt.savefig(fig_dir / 'fig4_computational_cost.pdf')
    plt.close()
    logger.info("✓ Figura 4: Costo computacional")
    
    # FIGURA 5: Convergencia
    fig, axes = plt.subplots(2, (n_funcs+1)//2, figsize=(18, 10))
    axes = axes.ravel()
    for idx, func in enumerate(results_df['function_name'].unique()):
        df_func = results_df[results_df['function_name'] == func]
        df_pivot = df_func.pivot_table(values='n_epochs', index='fold', columns='model_name')
        df_pivot.plot(kind='bar', ax=axes[idx], alpha=0.7)
        axes[idx].set_title(f'{func.capitalize()} - Épocas hasta convergencia')
        axes[idx].set_xlabel('Fold')
        axes[idx].set_ylabel('Épocas')
        axes[idx].legend(title='Modelo')
    plt.tight_layout()
    plt.savefig(fig_dir / 'fig5_convergence_analysis.pdf')
    plt.close()
    logger.info("✓ Figura 5: Análisis de convergencia")
    
    # ========================================================================
    # NUEVAS FIGURAS: MÉTRICAS DE FUNCIONES INTERNAS (3)
    # ========================================================================
    
    # Filtrar solo modelos neuronales (excluir GP)
    df_neural = results_df[results_df['model_name'] != 'GP'].copy()
    
    # FIGURA 6: Constante de Lipschitz
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 6a: Por función
    sns.barplot(data=df_neural, x='function_name', y='lipschitz_constant', 
                hue='model_name', ax=axes[0], errorbar='sd')
    axes[0].set_title('Constante de Lipschitz Empírica')
    axes[0].set_xlabel('')
    axes[0].set_ylabel('L (Percentil 95)')
    axes[0].set_xticklabels([t.get_text().capitalize() for t in axes[0].get_xticklabels()], rotation=45)
    
    # 6b: Comparación global
    df_global = df_neural.groupby('model_name').agg({
        'lipschitz_constant': ['mean', 'std']
    }).reset_index()
    df_global.columns = ['model_name', 'mean', 'std']
    axes[1].bar(df_global['model_name'], df_global['mean'], yerr=df_global['std'], 
                alpha=0.7, capsize=5)
    axes[1].set_title('Lipschitz Promedio (Todas las Funciones)')
    axes[1].set_ylabel('L')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()

    plt.savefig(fig_dir / 'fig6_lipschitz_analysis.pdf')
    plt.close()
    logger.info("✓ Figura 6: Análisis de constante de Lipschitz")
    
    # FIGURA 7: Suavidad de Representaciones Internas
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 7a: Suavidad por función
    sns.barplot(data=df_neural, x='function_name', y='smoothness_mean', 
                hue='model_name', ax=axes[0], errorbar='sd')
    axes[0].set_title('Suavidad de Representaciones Internas')
    axes[0].set_xlabel('')
    axes[0].set_ylabel('Varianza Local Promedio')
    axes[0].set_xticklabels([t.get_text().capitalize() for t in axes[0].get_xticklabels()], rotation=45)
    axes[0].set_yscale('log')
    
    # 7b: Comparación con mediana
    df_smooth = df_neural.groupby('model_name').agg({
        'smoothness_mean': 'mean',
        'smoothness_median': 'mean'
    }).reset_index()
    
    x = np.arange(len(df_smooth))
    width = 0.35
    axes[1].bar(x - width/2, df_smooth['smoothness_mean'], width, label='Media', alpha=0.7)
    axes[1].bar(x + width/2, df_smooth['smoothness_median'], width, label='Mediana', alpha=0.7)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df_smooth['model_name'])
    axes[1].set_ylabel('Varianza Local')
    axes[1].set_title('Suavidad: Media vs Mediana')
    axes[1].legend()
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'fig7_smoothness_analysis.pdf')
    plt.close()
    logger.info("✓ Figura 7: Análisis de suavidad")
    
    # FIGURA 8: Robustez Adversarial y Estabilidad de Gradientes
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 8a: Sensibilidad adversarial por función
    ax1 = fig.add_subplot(gs[0, 0])
    sns.barplot(data=df_neural, x='function_name', y='adversarial_sensitivity', 
                hue='model_name', ax=ax1, errorbar='sd')
    ax1.set_title('Sensibilidad Adversarial')
    ax1.set_xlabel('')
    ax1.set_ylabel('||Δf|| / ||Δx||')
    ax1.set_xticklabels([t.get_text().capitalize() for t in ax1.get_xticklabels()], rotation=45)
    ax1.legend().remove()
    
    # 8b: Error relativo
    ax2 = fig.add_subplot(gs[0, 1])
    sns.barplot(data=df_neural, x='function_name', y='relative_error', 
                hue='model_name', ax=ax2, errorbar='sd')
    ax2.set_title('Error Relativo ante Perturbaciones')
    ax2.set_xlabel('')
    ax2.set_ylabel('||Δf|| / ||f||')
    ax2.set_xticklabels([t.get_text().capitalize() for t in ax2.get_xticklabels()], rotation=45)
    ax2.legend().remove()
    
    # 8c: Comparación global de robustez
    ax3 = fig.add_subplot(gs[0, 2])
    df_robust = df_neural.groupby('model_name').agg({
        'adversarial_sensitivity': ['mean', 'std']
    }).reset_index()
    df_robust.columns = ['model_name', 'mean', 'std']
    ax3.bar(df_robust['model_name'], df_robust['mean'], yerr=df_robust['std'], 
            alpha=0.7, capsize=5, color=['#3498db', '#e74c3c'])
    ax3.set_title('Robustez Global')
    ax3.set_ylabel('Sensibilidad Promedio')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 8d: Coeficiente de variación de gradientes
    ax4 = fig.add_subplot(gs[1, 0])
    sns.barplot(data=df_neural, x='function_name', y='gradient_cv', 
                hue='model_name', ax=ax4, errorbar='sd')
    ax4.set_title('Estabilidad de Gradientes (CV)')
    ax4.set_xlabel('')
    ax4.set_ylabel('Coeficiente de Variación')
    ax4.set_xticklabels([t.get_text().capitalize() for t in ax4.get_xticklabels()], rotation=45)
    ax4.legend().remove()
    
    # 8e: Tendencia temporal de gradientes
    ax5 = fig.add_subplot(gs[1, 1])
    sns.barplot(data=df_neural, x='function_name', y='gradient_trend', 
                hue='model_name', ax=ax5, errorbar='sd')
    ax5.set_title('Tendencia de Gradientes')
    ax5.set_xlabel('')
    ax5.set_ylabel('Pendiente')
    ax5.set_xticklabels([t.get_text().capitalize() for t in ax5.get_xticklabels()], rotation=45)
    ax5.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # 8f: Ratio de gradientes desvanecientes
    ax6 = fig.add_subplot(gs[1, 2])
    sns.barplot(data=df_neural, x='function_name', y='vanishing_ratio', 
                hue='model_name', ax=ax6, errorbar='sd')
    ax6.set_title('Gradientes Desvanecientes')
    ax6.set_xlabel('')
    ax6.set_ylabel('Proporción (||∇|| < 1e-4)')
    ax6.set_xticklabels([t.get_text().capitalize() for t in ax6.get_xticklabels()], rotation=45)
    ax6.set_ylim([0, 1])
    
    # Leyenda compartida
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 0.98))
    
    plt.savefig(fig_dir / 'fig8_robustness_gradient_stability.pdf')
    plt.close()
    logger.info("✓ Figura 8: Robustez y estabilidad de gradientes")
    
    # ========================================================================
    # FIGURA 9: RADAR CHART COMPARATIVO (SÍNTESIS)
    # ========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), subplot_kw=dict(projection='polar'))
    
    # Preparar datos para radar chart
    metrics_normalized = {}
    metrics_to_plot = ['r2', 'lipschitz_constant', 'smoothness_mean', 
                       'adversarial_sensitivity', 'gradient_cv']
    
    for model in df_neural['model_name'].unique():
        df_model = df_neural[df_neural['model_name'] == model]
        values = []
        for metric in metrics_to_plot:
            if metric == 'r2':
                # R² más alto es mejor -> mantener
                values.append(df_model[metric].mean())
            else:
                # Otras métricas: más bajo es mejor -> invertir
                values.append(1.0 / (df_model[metric].mean() + 1e-6))
        metrics_normalized[model] = values
    
    # Normalizar a [0, 1]
    all_values = np.array(list(metrics_normalized.values()))
    min_vals = all_values.min(axis=0)
    max_vals = all_values.max(axis=0)
    
    for model in metrics_normalized:
        metrics_normalized[model] = (np.array(metrics_normalized[model]) - min_vals) / (max_vals - min_vals + 1e-6)
    
    # Plot para funciones suaves
    smooth_funcs = ['quadratic', 'branin', 'rosenbrock']
    df_smooth = df_neural[df_neural['function_name'].isin(smooth_funcs)]
    
    angles = np.linspace(0, 2 * np.pi, len(metrics_to_plot), endpoint=False).tolist()
    angles += angles[:1]
    
    for model in df_smooth['model_name'].unique():
        df_model = df_smooth[df_smooth['model_name'] == model]
        values = []
        for metric in metrics_to_plot:
            if metric == 'r2':
                values.append(df_model[metric].mean())
            else:
                values.append(1.0 / (df_model[metric].mean() + 1e-6))
        
        # Normalizar
        values = (np.array(values) - min_vals) / (max_vals - min_vals + 1e-6)
        values = values.tolist()
        values += values[:1]
        
        axes[0].plot(angles, values, 'o-', linewidth=2, label=model)
        axes[0].fill(angles, values, alpha=0.15)
    
    axes[0].set_xticks(angles[:-1])
    axes[0].set_xticklabels(['R²', 'Lipschitz⁻¹', 'Smoothness⁻¹', 'Robustness⁻¹', 'Grad Stability⁻¹'], 
                            fontsize=9)
    axes[0].set_ylim(0, 1)
    axes[0].set_title('Funciones Suaves', fontsize=12, pad=20)
    axes[0].legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    axes[0].grid(True)
    
    # Plot para funciones multimodales
    multimodal_funcs = ['rastrigin', 'ackley', 'six_hump']
    df_multi = df_neural[df_neural['function_name'].isin(multimodal_funcs)]
    
    for model in df_multi['model_name'].unique():
        df_model = df_multi[df_multi['model_name'] == model]
        values = []
        for metric in metrics_to_plot:
            if metric == 'r2':
                values.append(df_model[metric].mean())
            else:
                values.append(1.0 / (df_model[metric].mean() + 1e-6))
        
        # Normalizar
        values = (np.array(values) - min_vals) / (max_vals - min_vals + 1e-6)
        values = values.tolist()
        values += values[:1]
        
        axes[1].plot(angles, values, 'o-', linewidth=2, label=model)
        axes[1].fill(angles, values, alpha=0.15)
    
    axes[1].set_xticks(angles[:-1])
    axes[1].set_xticklabels(['R²', 'Lipschitz⁻¹', 'Smoothness⁻¹', 'Robustness⁻¹', 'Grad Stability⁻¹'], 
                            fontsize=9)
    axes[1].set_ylim(0, 1)
    axes[1].set_title('Funciones Multimodales', fontsize=12, pad=20)
    axes[1].legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'fig9_radar_comparison.pdf')
    plt.close()
    logger.info("✓ Figura 9: Radar chart comparativo")

# ============================================================================
# ORQUESTADOR PRINCIPAL
# ============================================================================

def run_benchmark_experiment(args):
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)
    
    logger.info("="*80)
    logger.info("BENCHMARK CIENTÍFICO - KST-SL vs BASELINES")
    logger.info(f"Output: {output_dir.resolve()}")
    logger.info(f"Activación KST-SL: {args.activation.upper()}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info("="*80)
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    # Cargar proyector
    projector = KSTProjector(n=2, psi_checkpoint=args.checkpoint)
    logger.info("✓ Proyector KST cargado")
    
    # Funciones de benchmark
    benchmark_functions = get_benchmark_functions()
    logger.info(f"✓ {len(benchmark_functions)} funciones de benchmark cargadas")
    
    all_results = []
    all_histories = {}
    
    # Configuraciones de modelos
    model_configs = [
        {'name': 'KST-SL', 'hidden_sizes': [256, 128, 64], 'dropout': 0.1},
        {'name': 'MLP', 'hidden_sizes': [256, 128, 64], 'dropout': 0.1},
        {'name': 'GP'},
    ]
    
    # ========================================================================
    # LOOP PRINCIPAL: FUNCIONES
    # ========================================================================
    for func_name, b_func in benchmark_functions.items():
        logger.info(f"\n{'='*80}\n{b_func.name} (Categoría: {b_func.category})\n{'='*80}")
        
        # Generar datos
        sampler = qmc.LatinHypercube(d=2, seed=args.seed)
        X = sampler.random(n=args.n_train + args.n_test)
        y = b_func.func(X)
        
        logger.info(f"Datos generados: {len(X)} muestras")
        logger.info(f"  └─ Rango Y: [{y.min():.3f}, {y.max():.3f}]")
        
        kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
        
        # ====================================================================
        # Entrenar modelos finales para curvas de aprendizaje
        # ====================================================================
        fold_histories = {}
        for config in model_configs:
            if config['name'] != 'GP':
                logger.info(f"Entrenando {config['name']} final para visualización...")
                X_train, X_val = X[:args.n_train], X[:int(args.n_train*0.15)]
                y_train, y_val = y[:args.n_train], y[:int(args.n_train*0.15)]
                
                if config['name'] == 'KST-SL':
                    activation_fn = get_activation_function(args.activation)
                    model = KSTSprecherLorentzModel(
                        projector, 
                        config['hidden_sizes'], 
                        config['dropout'],
                        activation=activation_fn
                    ).to(device)
                else:
                    activation_fn = get_activation_function(args.activation)
                    model = MLP(2, config['hidden_sizes'], config['dropout'], activation_fn).to(device)
                
                train_out = train_pytorch_model(model, X_train, y_train, X_val, y_val, args, device)
                fold_histories[config['name']] = train_out['history']
        
        all_histories[func_name] = fold_histories
        
        # ====================================================================
        # Cross-validation con métricas extendidas
        # ====================================================================
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
            logger.info(f"\n  Fold {fold_idx+1}/{args.n_folds}")
            
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
            
            for config in model_configs:
                result, _ = run_fold(config, fold_data, args, fold_idx, device, projector, logger)
                all_results.append(result)
                
                # Log resumido
                logger.info(
                    f"  {config['name']:10s} | "
                    f"R²={result.r2:.4f} RMSE={result.rmse:.4f} | "
                    f"L={result.lipschitz_constant:.3f} "
                    f"Smooth={result.smoothness_mean:.3e} "
                    f"Robust={result.adversarial_sensitivity:.3f}"
                )
    
    # ========================================================================
    # GUARDAR RESULTADOS
    # ========================================================================
    results_df = pd.DataFrame([asdict(r) for r in all_results])
    results_df.to_csv(output_dir / 'results_extended.csv', index=False)
    logger.info(f"\n✓ CSV extendido guardado: {output_dir / 'results_extended.csv'}")
    
    # ========================================================================
    # ESTADÍSTICAS TRADICIONALES
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("RESUMEN: MÉTRICAS TRADICIONALES")
    logger.info("="*80)
    summary_trad = results_df.groupby(['function_name', 'model_name']).agg(
        R2=('r2', 'mean'), R2_std=('r2', 'std'),
        RMSE=('rmse', 'mean'), RMSE_std=('rmse', 'std'),
        Time=('train_time_s', 'mean'), Epochs=('n_epochs', 'mean')
    ).round(4)
    print("\n" + summary_trad.to_string())
    
    # ========================================================================
    # ESTADÍSTICAS DE FUNCIONES INTERNAS
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("RESUMEN: MÉTRICAS DE FUNCIONES INTERNAS")
    logger.info("="*80)
    
    df_neural = results_df[results_df['model_name'] != 'GP']
    
    summary_internal = df_neural.groupby('model_name').agg(
        Lipschitz=('lipschitz_constant', 'mean'),
        Lip_std=('lipschitz_constant', 'std'),
        Smoothness=('smoothness_mean', 'mean'),
        Smooth_std=('smoothness_mean', 'std'),
        Adversarial=('adversarial_sensitivity', 'mean'),
        Adv_std=('adversarial_sensitivity', 'std'),
        GradCV=('gradient_cv', 'mean'),
        GradCV_std=('gradient_cv', 'std')
    ).round(4)
    print("\n" + summary_internal.to_string())
    
    # ========================================================================
    # TESTS ESTADÍSTICOS
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("TESTS ESTADÍSTICOS (t-test pareado)")
    logger.info("="*80)
    
    for func in results_df['function_name'].unique():
        df_func = results_df[results_df['function_name'] == func]
        
        logger.info(f"\n{func.capitalize()}:")
        
        # Test en R²
        kst_r2 = df_func[df_func['model_name'] == 'KST-SL']['r2'].values
        mlp_r2 = df_func[df_func['model_name'] == 'MLP']['r2'].values
        gp_r2 = df_func[df_func['model_name'] == 'GP']['r2'].values
        
        if len(kst_r2) > 0 and len(mlp_r2) > 0:
            t_mlp, p_mlp = stats.ttest_rel(kst_r2, mlp_r2)
            sig_mlp = '***' if p_mlp < 0.001 else '**' if p_mlp < 0.01 else '*' if p_mlp < 0.05 else 'ns'
            logger.info(f"  R² | KST-SL vs MLP: t={t_mlp:.3f}, p={p_mlp:.4f} {sig_mlp}")
        
        if len(kst_r2) > 0 and len(gp_r2) > 0:
            t_gp, p_gp = stats.ttest_rel(kst_r2, gp_r2)
            sig_gp = '***' if p_gp < 0.001 else '**' if p_gp < 0.01 else '*' if p_gp < 0.05 else 'ns'
            logger.info(f"  R² | KST-SL vs GP:  t={t_gp:.3f}, p={p_gp:.4f} {sig_gp}")
        
        # Test en Lipschitz (solo modelos neuronales)
        df_neural_func = df_func[df_func['model_name'] != 'GP']
        kst_lip = df_neural_func[df_neural_func['model_name'] == 'KST-SL']['lipschitz_constant'].dropna().values
        mlp_lip = df_neural_func[df_neural_func['model_name'] == 'MLP']['lipschitz_constant'].dropna().values
        
        if len(kst_lip) > 0 and len(mlp_lip) > 0 and len(kst_lip) == len(mlp_lip):
            t_lip, p_lip = stats.ttest_rel(kst_lip, mlp_lip)
            sig_lip = '***' if p_lip < 0.001 else '**' if p_lip < 0.01 else '*' if p_lip < 0.05 else 'ns'
            logger.info(f"  Lipschitz | KST-SL vs MLP: t={t_lip:.3f}, p={p_lip:.4f} {sig_lip}")
    
    # ========================================================================
    # VISUALIZACIONES
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("GENERANDO FIGURAS (9 TOTALES)")
    logger.info("="*80)
    plot_all_results(results_df, all_histories, benchmark_functions, output_dir, logger)
    
    # ========================================================================
    # ANÁLISIS POR CATEGORÍA
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("ANÁLISIS POR CATEGORÍA DE FUNCIÓN")
    logger.info("="*80)
    
    for category in ['smooth', 'non_convex', 'multimodal']:
        funcs_in_cat = [k for k, v in benchmark_functions.items() if v.category == category]
        df_cat = results_df[results_df['function_name'].isin(funcs_in_cat)]
        
        logger.info(f"\nCategoría: {category.upper()}")
        logger.info(f"Funciones: {', '.join(funcs_in_cat)}")
        
        summary_cat = df_cat.groupby('model_name').agg(
            R2=('r2', 'mean'),
            Lipschitz=('lipschitz_constant', 'mean'),
            Smoothness=('smoothness_mean', 'mean'),
            Robustness=('adversarial_sensitivity', 'mean')
        ).round(4)
        
        print(summary_cat.to_string())
    
    # ========================================================================
    # FINALIZAR
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info(f"✓ COMPLETO - Resultados en: {output_dir.resolve()}")
    logger.info("="*80)
    
    # Resumen final en pantalla
    print("\n" + "="*80)
    print("RESUMEN EJECUTIVO")
    print("="*80)
    print(f"\nTotal de experimentos: {len(all_results)}")
    print(f"Funciones evaluadas: {len(benchmark_functions)}")
    print(f"Modelos comparados: {len(model_configs)}")
    print(f"Folds por función: {args.n_folds}")
    print(f"\nArchivos generados:")
    print(f"  - results_extended.csv (con {len(results_df.columns)} columnas)")
    print(f"  - 9 figuras en {output_dir / 'figures'}")
    print(f"  - Log completo en {output_dir / 'logs'}")

    return output_dir / 'results_extended.csv'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Benchmark Científico Completo con Métricas Internas")
    parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint HDF5 del proyector KST')
    parser.add_argument('--output', type=str, default=f"results/full_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument('--n_train', type=int, default=500, help='Número de muestras de entrenamiento')
    parser.add_argument('--n_test', type=int, default=1000, help='Número de muestras de test')
    parser.add_argument('--n_folds', type=int, default=5, help='Número de folds para cross-validation')
    parser.add_argument('--epochs', type=int, default=500, help='Épocas máximas de entrenamiento')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--patience', type=int, default=50, help='Paciencia para early stopping')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--activation',
        type=str,
        default='gelu',
        choices=['gelu', 'relu', 'tanh', 'sigmoid', 'elu', 'leakyrelu', 'selu', 'silu'],
        help='Función de activación para redes neuronales (KST-SL y MLP)'
    )
    
    args = parser.parse_args()
    main(args)