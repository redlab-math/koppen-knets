# main.py
#!/usr/bin/env python3
"""
Main pipeline for Kolmogorov-Arnold Networks experiments.

Executes:
    --checkpoint: Generate internal function ψ (Actor algorithm)
    --benchmark: Evaluate model performance on test functions
    --ablation: Analyze component contributions
    --all: Execute complete pipeline
"""
import argparse
import sys
from pathlib import Path
from datetime import datetime

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / 'src'
sys.path.insert(0, str(SRC_DIR))

from config.params import KSTConfig
from koppen_pipeline.orchestrator import run_complete_pipeline
from utils.logging_utils import setup_experiment_logging
from utils.reproducibility import set_global_seeds


def parse_args():
    parser = argparse.ArgumentParser(
        description="Kolmogorov-Arnold Networks: Complete Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--checkpoint', action='store_true', help='Generate ψ checkpoint')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark evaluation')
    parser.add_argument('--ablation', action='store_true', help='Run ablation study')
    parser.add_argument('--all', action='store_true', help='Execute complete pipeline')
    
    parser.add_argument('--n', type=int, default=2, help='Input dimension (default: 2)')
    parser.add_argument('--k', type=int, default=6, help='Refinement parameter (default: 6)')
    parser.add_argument('--gamma', type=float, default=None, help='Base expansion (default: 2n+2)')
    parser.add_argument('--epsilon', type=float, default=None, help='Shift parameter (default: 1/(2n))')
    
    parser.add_argument('--epochs', type=int, default=500, help='Training epochs (default: 500)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size (default: 64)')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Learning rate (default: 5e-4)')
    parser.add_argument('--patience', type=int, default=50, help='Early stopping patience (default: 50)')
    parser.add_argument('--activation', type=str, default='gelu', 
                       choices=['gelu', 'relu', 'tanh', 'sigmoid', 'elu', 'leakyrelu', 'selu', 'silu'],
                       help='Activation function (default: gelu)')
    
    parser.add_argument('--n_train', type=int, default=500, help='Training samples (default: 500)')
    parser.add_argument('--n_test', type=int, default=1000, help='Test samples (default: 1000)')
    parser.add_argument('--n_folds', type=int, default=5, help='Cross-validation folds (default: 5)')
    
    parser.add_argument('--checkpoint_file', type=str, default=None, 
                       help='Path to existing checkpoint (for benchmark/ablation)')
    parser.add_argument('--output', type=str, default='results', help='Output directory (default: results)')
    
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID (default: 0)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    
    parser.add_argument('--no_verify', action='store_true', help='Skip verification step')
    parser.add_argument('--no_plots', action='store_true', help='Skip plot generation')
    parser.add_argument('--quiet', action='store_true', help='Suppress console output')
    
    return parser.parse_args()


def run_checkpoint_phase(args, logger):
    logger.info("=" * 80)
    logger.info("PHASE: CHECKPOINT GENERATION")
    logger.info("=" * 80)
    
    output_dir = Path(args.output) / 'checkpoints'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    epsilon_val = args.epsilon if args.epsilon is not None else 1.0 / (2.0 * args.n)
    gamma_val = args.gamma if args.gamma is not None else 2 * args.n + 2
    
    config = KSTConfig(n=args.n, k=args.k, gamma=int(gamma_val), epsilon=epsilon_val)
    
    checkpoint_name = f'koppen_n{args.n}_k{args.k}_gamma{int(gamma_val)}.h5'
    
    results = run_complete_pipeline(
        config=config,
        output_dir=output_dir,
        checkpoint_name=checkpoint_name,
        skip_verification=args.no_verify,
        verbose=not args.quiet
    )
    
    checkpoint_path = Path(results['checkpoint_path'])
    logger.info(f"Checkpoint saved: {checkpoint_path}")
    logger.info(f"Lipschitz constant: {results['metrics'].get('lipschitz_constant', 'N/A'):.6f}")
    
    return str(checkpoint_path)


def run_benchmark_phase(args, checkpoint_path, logger):
    logger.info("=" * 80)
    logger.info("PHASE: BENCHMARK EVALUATION")
    logger.info("=" * 80)
    
    sys.path.insert(0, str(ROOT_DIR))
    from experiments.benchmark import run_benchmark_experiment
    
    output_dir = Path(args.output) / 'benchmark'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    benchmark_args = argparse.Namespace(
        checkpoint=checkpoint_path,
        output=str(output_dir),
        n_train=args.n_train,
        n_test=args.n_test,
        n_folds=args.n_folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        activation=args.activation,
        gpu=args.gpu,
        seed=args.seed
    )
    
    results_csv = run_benchmark_experiment(benchmark_args)
    logger.info(f"Benchmark results: {results_csv}")
    
    return str(results_csv)


def run_ablation_phase(args, checkpoint_path, logger):
    logger.info("=" * 80)
    logger.info("PHASE: ABLATION STUDY")
    logger.info("=" * 80)
    
    sys.path.insert(0, str(ROOT_DIR))
    from experiments.ablation import run_ablation_experiment
    
    output_dir = Path(args.output) / 'ablation'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    ablation_args = argparse.Namespace(
        checkpoint=checkpoint_path,
        output=str(output_dir),
        n_train=args.n_train,
        n_test=args.n_test,
        n_folds=args.n_folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        activation=args.activation,
        gpu=args.gpu,
        seed=args.seed
    )
    
    results_csv = run_ablation_experiment(ablation_args)
    logger.info(f"Ablation results: {results_csv}")
    
    return str(results_csv)


def main():
    args = parse_args()
    
    if not any([args.checkpoint, args.benchmark, args.ablation, args.all]):
        print("ERROR: Specify at least one phase: --checkpoint, --benchmark, --ablation, or --all")
        return 1
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_experiment_logging(
        output_dir=output_dir,
        experiment_name='pipeline',
        level='INFO',
        quiet=args.quiet
    )
    
    set_global_seeds(args.seed)
    
    logger.info("=" * 80)
    logger.info("KOLMOGOROV-ARNOLD NETWORKS PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Output: {output_dir.resolve()}")
    logger.info(f"Seed: {args.seed}")
    
    try:
        checkpoint_path = None
        
        if args.checkpoint or args.all:
            checkpoint_path = run_checkpoint_phase(args, logger)
        elif args.checkpoint_file:
            checkpoint_path = args.checkpoint_file
            logger.info(f"Using existing checkpoint: {checkpoint_path}")
        
        if (args.benchmark or args.ablation or args.all) and not checkpoint_path:
            raise ValueError("Benchmark/ablation require --checkpoint_file or --checkpoint/--all")
        
        if args.benchmark or args.all:
            run_benchmark_phase(args, checkpoint_path, logger)
        
        if args.ablation or args.all:
            run_ablation_phase(args, checkpoint_path, logger)
        
        logger.info("=" * 80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline failed: {type(e).__name__}: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())