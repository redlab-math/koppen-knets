"""
Generate figures matching Actor's thesis (Figures 5.2-5.7).
This version includes all functions required by __main__.py.
"""
import matplotlib.pyplot as plt
import numpy as np
import h5py
from pathlib import Path
from typing import List

from .decimation import decimate_from_hdf5, adaptive_decimation_for_k

def _get_path(base_name: str) -> str:
    """Helper to construct the full HDF5 path."""
    if base_name in ['x_holder', 'y_holder']:
        return f'holder_function/{base_name}'
    if base_name in ['s_lipschitz', 'psi_lipschitz']:
        return f'lipschitz_function/{base_name}'
    if base_name == 'sigma':
        return 'reparametrization/sigma'
    return base_name

# --- FUNCIÓN AÑADIDA ---
def plot_figure_5_2_holder_functions(checkpoints: List[Path], output_path: Path):
    """Figure 5.2: ψ̃_k for multiple k values."""
    fig, ax = plt.subplots(figsize=(8, 8))
    for cp in sorted(checkpoints):
        with h5py.File(cp, 'r') as h5f:
            k = h5f['metadata'].attrs['k']
            n_points = adaptive_decimation_for_k(k)
            x, y = decimate_from_hdf5(h5f, _get_path('x_holder'), _get_path('y_holder'), n_points)
            ax.plot(x, y, label=f'k={k}', alpha=0.8, linewidth=1)
    
    ax.set_title("Hölder Functions ψ̃_k for Various k (Actor Fig 5.2)")
    ax.set_xlabel("x")
    ax.set_ylabel("ψ̃(x)")
    ax.legend()
    ax.grid(True, linestyle=':')
    ax.set_aspect('equal', 'box')
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  ✓ Saved Figure 5.2 style plot to {output_path}")

# --- FUNCIÓN AÑADIDA ---
def plot_figure_5_3_lipschitz_functions(checkpoints: List[Path], output_path: Path):
    """Figure 5.3: ψ_k for multiple k values."""
    fig, ax = plt.subplots(figsize=(8, 8))
    for cp in sorted(checkpoints):
        with h5py.File(cp, 'r') as h5f:
            k = h5f['metadata'].attrs['k']
            s, psi = decimate_from_hdf5(h5f, _get_path('s_lipschitz'), _get_path('psi_lipschitz'), 10000)
            ax.plot(s, psi, label=f'k={k}', alpha=0.8, linewidth=1)
    
    ax.set_title("Lipschitz Functions ψ_k for Various k (Actor Fig 5.3)")
    ax.set_xlabel("s (arc length)")
    ax.set_ylabel("ψ(s)")
    ax.legend()
    ax.grid(True, linestyle=':')
    ax.set_aspect('equal', 'box')
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  ✓ Saved Figure 5.3 style plot to {output_path}")

def plot_figure_5_4_comparison(checkpoint_path: Path, output_path: Path):
    """Figure 5.4: Side-by-side comparison ψ̃_k vs ψ_k."""
    with h5py.File(checkpoint_path, 'r') as h5f:
        k = int(h5f['metadata'].attrs['k'])
        n_points = adaptive_decimation_for_k(k, max_points=20000)
        x_h, y_h = decimate_from_hdf5(h5f, _get_path('x_holder'), _get_path('y_holder'), n_points)
        s_l, psi_l = decimate_from_hdf5(h5f, _get_path('s_lipschitz'), _get_path('psi_lipschitz'), 10000)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f"Hölder vs. Lipschitz Comparison (k={k})", fontsize=16)

        ax1.plot(x_h, y_h, 'b-', linewidth=1)
        ax1.set_title('Hölder Function ψ̃(x)', weight='bold')
        ax1.set_xlabel('x'); ax1.set_ylabel('ψ̃(x)'); ax1.grid(True, alpha=0.3); ax1.set_aspect('equal')

        ax2.plot(s_l, psi_l, 'r-', linewidth=1)
        ax2.set_title('Lipschitz Function ψ(s)', weight='bold')
        ax2.set_xlabel('s (arc length)'); ax2.set_ylabel('ψ(s)'); ax2.grid(True, alpha=0.3); ax2.set_aspect('equal')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95]); plt.savefig(output_path, dpi=150); plt.close()
    print(f"  ✓ Saved Figure 5.4 style plot to {output_path}")

def plot_figure_5_5_refinement_rate(checkpoints: List[Path], output_path: Path):
    """Figure 5.5: Semilog plot of max interval sizes vs k."""
    k_values, max_intervals, theoretical = [], [], []
    for cp in checkpoints:
        with h5py.File(cp, 'r') as h5f:
            k = int(h5f['metadata'].attrs['k']); gamma = int(h5f['metadata'].attrs['gamma'])
            x_ds = h5f[_get_path('x_holder')]
            indices = np.linspace(0, len(x_ds) - 2, min(10000, len(x_ds) - 1), dtype=int)
            k_values.append(k); max_intervals.append(float(np.max(np.diff(x_ds[indices])))); theoretical.append(1.0 / (gamma ** k))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(k_values, max_intervals, 'o-', label='Measured max interval')
    ax.semilogy(k_values, theoretical, 's--', label='Theoretical γ^(-k)', alpha=0.7)
    ax.set_title('Refinement Condition Verification (Actor Fig 5.5)'); ax.set_xlabel('Refinement Level k'); ax.set_ylabel('Max Interval Size |T_k| (log)')
    ax.grid(True, which='both', linestyle=':'); ax.legend()
    plt.tight_layout(); plt.savefig(output_path, dpi=150); plt.close()
    print(f"  ✓ Saved Figure 5.5 style plot to {output_path}")

def create_comprehensive_report(checkpoint_path: Path, output_path: Path):
    """Master diagnostic plot combining all key visualizations."""
    with h5py.File(checkpoint_path, 'r') as h5f:
        k = int(h5f['metadata'].attrs['k']); n = int(h5f['metadata'].attrs['n']); gamma = int(h5f['metadata'].attrs['gamma'])
        n_plot = adaptive_decimation_for_k(k, max_points=15000)
        x_h, y_h = decimate_from_hdf5(h5f, _get_path('x_holder'), _get_path('y_holder'), n_plot)
        s_l, psi_l = decimate_from_hdf5(h5f, _get_path('s_lipschitz'), _get_path('psi_lipschitz'), 10000)
        L = float(h5f['reparametrization'].attrs['arc_length'])

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Köppen-Actor Analysis | n={n}, k={k}, γ={gamma}', fontsize=16, weight='bold')

        axes[0, 0].plot(x_h, y_h, 'b-', lw=0.8); axes[0, 0].set_title('a) Hölder ψ̃(x)'); axes[0, 0].grid(True, alpha=0.3)
        axes[0, 1].plot(s_l, psi_l, 'r-', lw=0.8); axes[0, 1].set_title('b) Lipschitz ψ(s)'); axes[0, 1].grid(True, alpha=0.3)
        dy_h = np.abs(np.diff(y_h) / np.diff(x_h)); axes[1, 0].semilogy((x_h[:-1]+x_h[1:])/2, dy_h, 'b-', lw=0.5); axes[1, 0].set_title('c) |dψ̃/dx| (Unbounded)'); axes[1, 0].grid(True, alpha=0.3)
        dy_l = np.abs(np.diff(psi_l) / np.diff(s_l)); axes[1, 1].plot((s_l[:-1]+s_l[1:])/2, dy_l, 'r-', lw=0.5); axes[1, 1].axhline(L, c='k', ls='--', label=f'L={L:.3f}'); axes[1, 1].set_title('d) |dψ/ds| (Bounded)'); axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.savefig(output_path, dpi=150); plt.close()
    print(f"✓ Comprehensive report saved: {output_path}")

