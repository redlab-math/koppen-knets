"""
Entry point for the Köppen-Actor KST project.
This script initializes the configuration, orchestrates the pipeline execution,
and generates the final plots, tying together all the modular components.
"""
import argparse
import sys
from pathlib import Path
import jax

# --- Solución a los problemas de importación ---
# Añade el directorio 'src' al path de Python.
# Esto permite que el script encuentre el paquete 'koppen_knets'
# sin importar si está instalado o no.
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / 'src'
sys.path.insert(0, str(SRC_DIR))
# ------------------------------------------------

# --- Importaciones Corregidas ---
# Ahora se importa desde 'koppen_knets', que coincide con tu estructura de carpetas.
from koppen_knets.config.params import create_default_config
from koppen_knets.pipeline.orchestrator import run_complete_pipeline
from koppen_knets.viz.figures import (
    plot_figure_5_2_holder_functions,
    plot_figure_5_3_lipschitz_functions,
    plot_figure_5_4_comparison,
    plot_figure_5_5_refinement_rate,
    create_comprehensive_report
)
from koppen_knets.core.recursion import test_koppen_properties


def setup_argparse():
    """Configure command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Köppen-Actor KST (Reengineered)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test of recursion logic (k=3)
  python __main__.py --test
  
  # Standard run (k=6)
  python __main__.py --n 2 --k 6
  
  # Generate all figures (k=1..6)
  python __main__.py --n 2 --multi_k --k_min 1 --k_max 6
  
  # Large k for multiparallel systems (skip verification and plots)
  python __main__.py --n 2 --k 12 --no_verify --no_plots --output cluster_k12
        """
    )
    
    parser.add_argument('--n', type=int, default=2, help='Spatial dimension (default: 2)')
    parser.add_argument('--k', type=int, default=6, help='Refinement level for single run (default: 6)')
    parser.add_argument('--gamma', type=int, default=None, help='Base for expansion (default: 2n+2)')
    parser.add_argument('--epsilon', type=float, default=None, help='Shift parameter (default: 1/(2n))')
    parser.add_argument('--multi_k', action='store_true', help='Run for multiple k values (k_min to k_max)')
    parser.add_argument('--k_min', type=int, default=1, help='Minimum k for multi-k mode (default: 1)')
    parser.add_argument('--k_max', type=int, default=6, help='Maximum k for multi-k mode (default: 6)')
    parser.add_argument('--output', type=str, default='koppen_results', help='Output directory (default: koppen_results)')
    parser.add_argument('--no_verify', action='store_true', help='Skip verification (for very large k)')
    parser.add_argument('--no_plots', action='store_true', help='Skip plotting (cluster mode)')
    parser.add_argument('--test', action='store_true', help='Run unit tests on recursion')
    parser.add_argument('--quiet', action='store_true', help='Minimal output')
    
    return parser


def run_single_k(args):
    """Execute pipeline for a single k value"""
    config = create_default_config(n=args.n, k=args.k, gamma=args.gamma, epsilon=args.epsilon)
    output_dir = Path(args.output)
    verbose = not args.quiet
    
    results = run_complete_pipeline(config, output_dir, skip_verification=args.no_verify, verbose=verbose)
    
    if not args.no_plots:
        print("\n" + "="*80 + "\nGENERATING PLOTS\n" + "="*80)
        checkpoint = Path(results['checkpoint_path'])
        create_comprehensive_report(checkpoint, output_dir / f'comprehensive_report_k{args.k}.png')
    
    return results


def run_multi_k(args):
    """Execute pipeline for multiple k values and generate aggregate plots"""
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    verbose = not args.quiet
    checkpoints = []

    for k in range(args.k_min, args.k_max + 1):
        print(f"\n{'='*80}\nPROCESSING k = {k} / {args.k_max}\n{'='*80}")
        config = create_default_config(n=args.n, k=k, gamma=args.gamma, epsilon=args.epsilon)
        k_output_dir = output_dir / f'k{k}'
        results = run_complete_pipeline(config, k_output_dir, skip_verification=args.no_verify, verbose=verbose)
        checkpoints.append(Path(results['checkpoint_path']))
    
    if not args.no_plots:
        print(f"\n{'='*80}\nGENERATING AGGREGATE FIGURES\n{'='*80}\n")
        plot_figure_5_2_holder_functions(checkpoints, output_dir / 'figure_5_2_holder_functions.png')
        plot_figure_5_3_lipschitz_functions(checkpoints, output_dir / 'figure_5_3_lipschitz_functions.png')
        plot_figure_5_5_refinement_rate(checkpoints, output_dir / 'figure_5_5_refinement_rate.png')
        print(f"\nAll figures saved to: {output_dir}")


def main():
    """Main entry point"""
    parser = setup_argparse()
    args = parser.parse_args()
    
    if not args.quiet:
        print("="*80 + "\nJAX CONFIGURATION\n" + "="*80)
        try:
            print(f"Backend: {jax.default_backend()}")
            print(f"Devices: {jax.devices()}")
            jax.config.update("jax_enable_x64", True)
            print(f"x64 enabled: {jax.config.x64_enabled}")
        except Exception as e:
            print(f"Could not configure JAX: {e}")
        print("="*80)
    
    if args.test:
        print("\nRunning internal tests on recursion logic...")
        test_koppen_properties(n=args.n, gamma=args.gamma or 2*args.n+2, k=3)
        return
    
    try:
        if args.multi_k:
            run_multi_k(args)
        else:
            run_single_k(args)
        print("\n" + "="*80 + "\nSUCCESS\n" + "="*80)
    except Exception as e:
        print("\n" + "="*80 + f"\nERROR: {type(e).__name__}: {e}\n" + "="*80)
        if not args.quiet:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()