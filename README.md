# Empirical Characterization of Lipschitz Continuity in k-nets with Precomputed Inner Functions

Code repository accompanying:

> Huerta-Mendoza, U. B. (2025). *Empirical Characterization of Lipschitz Continuity in K-nets with Precomputed Inner Functions.* Informaticae Abstracta, 3(2), 4-24. https://informaticae.uaemex.mx/article/view/27775

A single command reproduces the artifacts reported in the paper: the precomputed Lipschitz-continuous inner function $\psi$, the internal-function figures, and the benchmark and ablation tables and plots.

## Architecture

The architecture, referred to as `K-Net-Complete` in the paper, is the `KSTSprecherLorentzModel` defined in `src/knet_core/kst_sl_model.py`. It realizes the Sprecher-Lorentz form of the Kolmogorov-Arnold representation theorem,

$$ f(x) = \sum_{q=0}^{2n} \Phi\\left( \sum_{p=1}^{n} \lambda_{qp}\, \psi(x_p + q\varepsilon) \right), $$

with three components. The inner function $\psi$ is fixed and precomputed: it is the Lipschitz-continuous function obtained by arc-length reparametrization of the Köppen function following Actor (2018), stored as an HDF5 checkpoint and evaluated by `KSTProjector` (`src/knet_core/kst_projector.py`) through a left-step lookup with periodic extension. The scale coefficients $\Lambda \in \mathbb{R}^{Q \times n}$, with $Q = 2n+1$, are trainable. The outer network $\Phi$ is a shared univariate MLP applied independently to each projection $Z_q$.

## Requirements and installation

Python 3.12, PyTorch 2.4.0 (CUDA 12.1) and JAX. JAX drives the inner-function construction, PyTorch drives model training, scikit-learn provides the Gaussian Process baseline, and h5py handles checkpoint storage.

```bash
conda env create -f environment.yml
conda activate koppen-knets
```

Alternatively, inside a Python 3.12 environment, `pip install -r requirements.txt`. A CUDA-compatible GPU is optional; it accelerates training but is not required, and the code falls back to CPU automatically.

## Reproduction

`main.py` is the entry point. A single command runs the full flow, in order: checkpoint generation, internal-function figures, benchmark, ablation.

```bash
python main.py --all \
    --n 2 --k 6 \
    --output results/run_k6 \
    --n_train 500 --n_test 1000 --n_folds 5 \
    --epochs 300 --batch_size 64 \
    --learning_rate 5e-4 --patience 50 \
    --activation gelu \
    --gpu 0 --seed 42
```

The base $\gamma$ defaults to $2n+2$ and the shift $\varepsilon$ to $1/(2n)$; set `--gamma` and `--epsilon` to override. Phases can also be run individually with `--checkpoint`, `--benchmark`, `--ablation`. The benchmark and ablation phases require a checkpoint, produced in the same call or supplied via `--checkpoint_file`:

```bash
python main.py --benchmark --ablation \
    --checkpoint_file results/run_k6/checkpoints/koppen_n2_k6_gamma6.h5 \
    --output results/run_k6
```

`--no_plots` skips figure generation; `--no_verify` skips the KST-condition verification. The resource check before checkpoint generation does not block in non-interactive sessions.

## Outputs

A full run under `--output <dir>` writes:

```
<dir>/
├── checkpoints/koppen_n<n>_k<k>_gamma<gamma>.h5   precomputed psi and intermediate datasets
├── figures/
│   ├── comprehensive_report.png                   Hölder, Lipschitz and derivative panels
│   └── holder_vs_lipschitz.png                    side-by-side comparison
├── benchmark/                                     metrics CSV, LaTeX tables, comparison plots
└── ablation/                                      metrics CSV, LaTeX tables, regularity plots, statistical summary
```

## Pipeline

The checkpoint phase, in `src/koppen_pipeline/orchestrator.py`, generates the Hölder function $\tilde\psi_k$ by streaming the Köppen recursion to HDF5, computes the cumulative arc length $\sigma$, reparametrizes to the Lipschitz function $\psi(s)$, and optionally verifies the KST conditions.

`experiments/benchmark.py` compares `K-Net-Complete` against an MLP and a Gaussian Process (`Constant × Matern` kernel) over six 2-D functions (quadratic, Branin, Rosenbrock, Rastrigin, Ackley, six-hump camel), reporting MAE, RMSE, $R^2$, the empirical Lipschitz constant, and paired t-tests.

`experiments/ablation.py` evaluates four variants, `K-Net-Complete`, `K-Net-FixedPsi` ($\psi$ fixed to $\tanh$), `K-Net-FixedLambda` ($\Lambda$ not trainable) and `MLP`, over the same functions, with regularity metrics: empirical Lipschitz constant, internal smoothness, adversarial robustness, and gradient-norm stability.

## Repository structure

```
.
├── main.py                      Entry point: checkpoint, figures, benchmark, ablation
├── environment.yml              Conda environment (koppen-knets)
├── requirements.txt             Pip requirements
├── LICENSE
│
├── src/
│   ├── config/
│   │   ├── params.py            KSTConfig and configuration handling
│   │   └── validation.py        Verification of the KST conditions
│   ├── koppen_core/             Inner-function construction
│   │   ├── recursion.py         Köppen recursion (Hölder function)
│   │   ├── math.py              Base, shift and weight helpers
│   │   ├── arclength.py         Streaming arc-length computation
│   │   └── reparam.py           Reparametrization to the Lipschitz function
│   ├── koppen_pipeline/         Checkpoint generation
│   │   ├── orchestrator.py      Pipeline driver
│   │   ├── storage.py           HDF5 I/O and streaming Hölder generation
│   │   └── memory.py            Memory monitoring and resource checks
│   ├── koppen_viz/              Internal-function figures
│   │   ├── figures.py
│   │   └── decimation.py
│   ├── knet_core/               The k-net
│   │   ├── kst_projector.py     Loads and evaluates the psi checkpoint
│   │   └── kst_sl_model.py      KSTSprecherLorentzModel (K-Net-Complete)
│   └── utils/
│       ├── logging_utils.py
│       └── reproducibility.py
│
├── experiments/
│   ├── benchmark.py             K-Net vs MLP vs GP
│   ├── ablation.py              Four variants and regularity metrics
│   └── analysis/
│       └── metrics_internal_functions.py   Regularity metrics shared by benchmark.py
│
└── deprecated/                  Earlier entry point and tests, kept for reference
```
