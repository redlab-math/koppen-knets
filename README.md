Code Repository for: "Empirical Characterization of Lipschitz Continuity in k-nets with Precomputed Inner Functions"


This repository contains the complete implementation, experimentation scripts, and analysis pipeline for "Empirical Characterization of Lipschitz Continuity in k-nets with Precomputed Inner Functions"

The code provided allows for the full replication of the paper's key results, including:

The generation of the Lipschitz-continuous inner function ($\psi$) checkpoint.

The ablation study that validates the model's components (Section 5.1, 5.2).

The benchmark evaluation that characterizes the network's performance and regularity against baselines.

Architecture

The central architecture, referred to as "k-net-Complete" in the paper, is defined in src/knet_core/kst_sl_model.py as KSTSprecherLorentzModel. It implements the Sprecher-Lorentz reformulation of the Kolmogorov-Arnold Representation Theorem (KRT) with three primary components:

Fixed Inner Function ($\psi$): A precomputed, Lipschitz-continuous $\psi$ function, as derived from Actor (2018), loaded from an HDF5 checkpoint by the KSTProjector class (src/knet_core/kst_projector.py).

Trainable Coefficients ($\Lambda$): A learnable coefficient matrix lambda_matrix ($\Lambda \in \mathbb{R}^{Q \times n}$) that scales the projections.

Shared Outer Network ($\Phi$): A shared, univariate MLP (outer_net) that processes each projection $Z_q$ independently.

Repository Structure

.
├── koppen_knets_f_3.pdf    # The paper manuscript [Reference]
├── main.py                 # ENTRY POINT: Main pipeline orchestrator
│
├── experiments/
│   ├── ablation.py         # Ablation study script (Section 4.2, 5.1, 5.2)
│   │                     (Defines KNetFixedPsi, KNetFixedLambda, MLP)
│   │                     (Implements regularity metrics: L_emp, Smoothness, etc.)
│   └── benchmark.py        # Benchmark evaluation script (vs. MLP, GP)
│
└── src/
    ├── knet_core/
    │   ├── kst_projector.py    # KSTProjector class: Loads and evaluates the ψ.h5 checkpoint
    │   └── kst_sl_model.py     # KSTSprecherLorentzModel class: The "k-net-Complete" architecture
    │
    └── koppen_pipeline/
        ├── orchestrator.py   # Logic for generating the ψ checkpoint (Phase 1)
        └── ...               # Other helper modules for checkpoint generation


A CUDA-compatible GPU is required for fast training (specified via the --gpu flag).

The main.py script is the orchestrator designed to reproduce all results from the paper with a single command.

This command executes the entire three-phase experimental pipeline. It will first generate the $\psi$ checkpoint, then run the benchmark, and finally run the complete ablation study.

python main.py --all \
    --n 2 --k 6 --gamma 10 \
    --output results/ablation_complete_k6_r \
    --n_train 500 --n_test 500 --n_folds 5 \
    --epochs 300 --batch_size 64 \
    --learning_rate 5e-4 --patience 50 \
    --activation gelu \
    --gpu 0 --seed 42
