# Empirical Characterization of Lipschitz Continuity in k-nets with Precomputed Inner Functions

Code repository accompanying:

> Huerta-Mendoza, U. B. (2025). *Empirical Characterization of Lipschitz Continuity in K-nets with Precomputed Inner Functions.* Informaticae Abstracta, 3(2), 4-24. https://informaticae.uaemex.mx/article/view/27775

<<<<<<< HEAD
The repository reproduces the results reported in the paper: generation of the precomputed Lipschitz-continuous inner function $\psi$, the benchmark evaluation against MLP and Gaussian Process baselines, and the ablation study over the architecture components.
=======
A single command reproduces the artifacts reported in the paper: the precomputed Lipschitz-continuous inner function $\psi$, the internal-function figures, and the benchmark and ablation tables and plots.
>>>>>>> 85ab959 (Reparación del flujo de ejecución y portabilidad del entorno)

## Architecture

The architecture, referred to as `K-Net-Complete` in the paper, is the `KSTSprecherLorentzModel` defined in `src/knet_core/kst_sl_model.py`. It realizes the Sprecher-Lorentz form of the Kolmogorov-Arnold representation theorem,

$$ f(x) = \sum_{q=0}^{2n} \Phi\!\left( \sum_{p=1}^{n} \lambda_{qp}\, \psi(x_p + q\varepsilon) \right), $$

<<<<<<< HEAD
with three components:

The inner function $\psi$ is fixed and precomputed. It is the Lipschitz-continuous function obtained by arc-length reparametrization of the Köppen function following Actor (2018), stored as an HDF5 checkpoint and evaluated by the `KSTProjector` class in `src/knet_core/kst_projector.py` through a left-step lookup with periodic extension.

The scale coefficients $\Lambda \in \mathbb{R}^{Q \times n}$, with $Q = 2n+1$, are trainable (`lambda_matrix`).

The outer network $\Phi$ is a shared univariate MLP (`outer_net`) applied independently to each projection $Z_q$. Its width and activation are configurable.

## Requirements and installation

The environment is specified for Python 3.12, PyTorch 2.4.0 (CUDA 12.1) and JAX. JAX drives the inner-function construction; PyTorch drives model training; scikit-learn provides the Gaussian Process baseline; h5py handles checkpoint storage.

With conda:

```bash
conda env create -f environment.yml
conda activate koppen-knets
```

With pip, inside a Python 3.12 environment:

```bash
pip install -r requirements.txt
```

A CUDA-compatible GPU is optional. It accelerates training but is not required; the problem sizes in the paper run on CPU.

## Usage

`main.py` is the entry point. It exposes three phases, executable individually or together with `--all`. The order under `--all` is checkpoint generation, benchmark, ablation. The benchmark and ablation phases require a checkpoint, supplied either by running `--checkpoint`/`--all` in the same call or by pointing `--checkpoint_file` to an existing one.

Full reproduction with default base $\gamma = 2n+2$ and shift $\varepsilon = 1/(2n)$:

=======
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

>>>>>>> 85ab959 (Reparación del flujo de ejecución y portabilidad del entorno)
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

<<<<<<< HEAD
Generate only the $\psi$ checkpoint:

```bash
python main.py --checkpoint --n 2 --k 6 --output results/run_k6
```

Run benchmark and ablation reusing an existing checkpoint:
=======
The base $\gamma$ defaults to $2n+2$ and the shift $\varepsilon$ to $1/(2n)$; set `--gamma` and `--epsilon` to override. Phases can also be run individually with `--checkpoint`, `--benchmark`, `--ablation`. The benchmark and ablation phases require a checkpoint, produced in the same call or supplied via `--checkpoint_file`:
>>>>>>> 85ab959 (Reparación del flujo de ejecución y portabilidad del entorno)

```bash
python main.py --benchmark --ablation \
    --checkpoint_file results/run_k6/checkpoints/koppen_n2_k6_gamma6.h5 \
    --output results/run_k6
```

<<<<<<< HEAD
### Command-line arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--checkpoint` | off | Generate the $\psi$ checkpoint |
| `--benchmark` | off | Run the benchmark evaluation |
| `--ablation` | off | Run the ablation study |
| `--all` | off | Run the three phases in sequence |
| `--n` | 2 | Input dimension |
| `--k` | 6 | Refinement level of the Köppen construction |
| `--gamma` | $2n+2$ | Base of the expansion |
| `--epsilon` | $1/(2n)$ | Shift parameter |
| `--epochs` | 500 | Maximum training epochs |
| `--batch_size` | 64 | Batch size |
| `--learning_rate` | 5e-4 | Learning rate |
| `--patience` | 50 | Early-stopping patience |
| `--activation` | gelu | Outer-network activation (gelu, relu, tanh, sigmoid, elu, leakyrelu, selu, silu) |
| `--n_train` | 500 | Training samples |
| `--n_test` | 1000 | Test samples |
| `--n_folds` | 5 | Cross-validation folds |
| `--checkpoint_file` | none | Existing checkpoint for benchmark/ablation |
| `--output` | results | Output directory |
| `--gpu` | 0 | GPU device id |
| `--seed` | 42 | Random seed |
| `--no_verify` | off | Skip KST-condition verification |
| `--quiet` | off | Suppress console output |

## Checkpoint pipeline

The checkpoint phase, orchestrated in `src/koppen_pipeline/orchestrator.py`, proceeds in four steps. It generates the Hölder function $\tilde\psi_k$ by streaming the Köppen recursion to HDF5, computes the cumulative arc length $\sigma$, reparametrizes to the Lipschitz function $\psi(s)$, and verifies the KST conditions. The result is written to `results/<output>/checkpoints/koppen_n<n>_k<k>_gamma<gamma>.h5`.

## Evaluation

`experiments/benchmark.py` compares `K-Net-Complete` against an MLP and a Gaussian Process (`GaussianProcessRegressor`, $\text{Constant} \times \text{Matern}$ kernel) over six 2-D functions (quadratic, Branin, Rosenbrock, Rastrigin, Ackley, six-hump camel), reporting MAE, RMSE, $R^2$, the empirical Lipschitz constant, and paired t-tests.

`experiments/ablation.py` evaluates four variants, `K-Net-Complete` (`KSTSprecherLorentzModel`), `K-Net-FixedPsi` (`KNetFixedActivation`, inner function fixed to $\tanh$), `K-Net-FixedLambda` ($\Lambda$ not trainable) and `MLP`, over the same functions, with regularity metrics: empirical Lipschitz constant, internal smoothness, adversarial robustness, and gradient-norm stability.
=======
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
>>>>>>> 85ab959 (Reparación del flujo de ejecución y portabilidad del entorno)

## Repository structure

```
.
<<<<<<< HEAD
├── main.py                      Entry point: orchestrates the three phases
=======
├── main.py                      Entry point: checkpoint, figures, benchmark, ablation
>>>>>>> 85ab959 (Reparación del flujo de ejecución y portabilidad del entorno)
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
<<<<<<< HEAD
│   │   ├── orchestrator.py      Four-step pipeline driver
│   │   ├── storage.py           HDF5 I/O and streaming Hölder generation
│   │   └── memory.py            Memory monitoring and resource checks
=======
│   │   ├── orchestrator.py      Pipeline driver
│   │   ├── storage.py           HDF5 I/O and streaming Hölder generation
│   │   └── memory.py            Memory monitoring and resource checks
│   ├── koppen_viz/              Internal-function figures
│   │   ├── figures.py
│   │   └── decimation.py
>>>>>>> 85ab959 (Reparación del flujo de ejecución y portabilidad del entorno)
│   ├── knet_core/               The k-net
│   │   ├── kst_projector.py     Loads and evaluates the psi checkpoint
│   │   └── kst_sl_model.py      KSTSprecherLorentzModel (K-Net-Complete)
│   └── utils/
│       ├── logging_utils.py
<<<<<<< HEAD
│       └── reproducibility.py   Global seeding
=======
│       └── reproducibility.py
>>>>>>> 85ab959 (Reparación del flujo de ejecución y portabilidad del entorno)
│
├── experiments/
│   ├── benchmark.py             K-Net vs MLP vs GP
│   ├── ablation.py              Four variants and regularity metrics
│   └── analysis/
<<<<<<< HEAD
│       └── metrics_internal_functions.py   Auxiliary analysis, not in the main pipeline
│
├── checkpoint.py                Legacy entry point (previous package layout), superseded by main.py
├── pyproject.toml               Empty placeholder
├── deprecated/                  Earlier tests and reference implementation
└── tests/                       Placeholder
```

Reproduction goes entirely through `main.py`. `checkpoint.py` targets a previous `koppen_knets.*` package layout that is no longer present and is kept only for reference.
=======
│       └── metrics_internal_functions.py   Regularity metrics shared by benchmark.py
│
└── deprecated/                  Earlier entry point and tests, kept for reference
```
>>>>>>> 85ab959 (Reparación del flujo de ejecución y portabilidad del entorno)
