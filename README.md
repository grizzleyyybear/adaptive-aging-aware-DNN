# Adaptive Aging-Aware DNN Accelerator via Predictive Lifetime Management

A predictive lifetime management framework for DNN accelerators that combines
spatial-temporal learning (**Hybrid GNN-Transformer**), multi-objective
evolutionary optimization (**NSGA-II**), and reinforcement learning (**PPO**)
to predict, prevent, and mitigate transistor aging in hardware accelerators.

---

## Key Results

| Metric | This Work | Previous | Published Paper |
|---|---|---|---|
| **Predictor R²** | **0.9953** | 0.9925 | 0.9871 |
| **Predictor MAE** | **0.0036** | 0.005 | 0.0209 |
| **Trajectory R²** | 0.7594 | 0.78 | 0.9595 |
| **Trajectory MAE** | 0.0756 | 0.072 | 0.0433 |
| **NSGA-II best reduction** | **76.3%** (ViT-B/16) | 63% | ~63% |
| **NSGA-II Pareto solutions** | **35 total** | 4–10/workload | — |
| **PPO best reward** | **+0.45** | +0.36 | +0.36 |
| **PPO mean reward** | **+0.24** | — | — |
| **Pipeline checks** | **9/9 pass** | — | — |

### Per-Workload NSGA-II Results (40k dataset)

| Workload | Solutions | Peak Aging | Reduction | Converged Gen |
|---|---|---|---|---|
| ResNet-50 | 12 | 0.278 → 0.270 | +2.9% | 20 |
| BERT-Base | 4 | 0.271 → 0.132 | **+51.3%** | 15 |
| MobileNetV2 | 5 | 0.341 → 0.337 | +1.1% | 14 |
| EfficientNet-B4 | 5 | 0.347 → 0.346 | +0.1% | 15 |
| ViT-B/16 | 9 | 0.091 → 0.022 | **+76.3%** | 20 |

---

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                     Full Pipeline Flow                         │
│                                                                │
│  Accelerator Graph ──► Feature Builder ──► Dataset (40k)       │
│         │                                      │               │
│         ▼                                      ▼               │
│  Aging Models (NBTI+HCI+TDDB)     Hybrid GNN-Transformer      │
│                                    (GCN → GAT → Transformer)   │
│                                         │          │           │
│                                         ▼          ▼           │
│                               Trajectory Pred   NSGA-II Opt    │
│                               (10-step ahead)   (3 objectives) │
│                                         │          │           │
│                                         ▼          ▼           │
│                                    PPO Controller              │
│                                    (runtime aging control)     │
└────────────────────────────────────────────────────────────────┘
```

### Technical Contributions

1. **Hotspot-Level Aging Modelling (C1)** — Per-node aging from calibrated NBTI,
   HCI, and TDDB physics models with weighted aggregation (0.40 / 0.35 / 0.25).

2. **Hybrid GNN-Transformer Predictor (C2)** — GCN spatial encoding → GAT
   attention → Transformer global context. Achieves R² = 0.9953 on 40k samples.

3. **10-Step Trajectory Predictor (C3)** — Forecasts future aging with discounted
   loss (γ = 0.95). Enables proactive lifetime management.

4. **NSGA-II Multi-Objective Optimizer (C4)** — Jointly minimizes peak aging,
   latency, and energy. Includes evaluation cache (394 hits), convergence
   detection, and aging-variance balance penalty.

5. **PPO Runtime Controller (C5)** — RL policy with residual blocks, observation
   normalization, KL early-stopping, entropy annealing, and linear LR decay.

---

## Repository Structure

```
adaptive-aging-aware-DNN/
├── aging_models/          # Transistor aging physics (NBTI, HCI, TDDB)
├── configs/               # Hydra configs for accelerator, workloads, training
├── checkpoints/           # Trained model weights (predictor, trajectory, RL)
├── data/                  # Datasets (40k train / 5k val / 5k test)
├── evaluation/            # Performance & reliability metrics, statistical tests
├── experiments/           # Experiment configurations
├── features/              # Feature extraction & activity profiling
├── graph/                 # Accelerator graph & PyG dataset construction
├── models/                # Hybrid GNN-Transformer & trajectory predictor
├── optimization/          # NSGA-II optimizer with cache & convergence detection
├── planning/              # Lifetime planner & budget allocation
├── rl/                    # PPO policy network, trainer, Gymnasium environment
├── scheduler/             # Workload scheduling logic
├── scripts/               # Pipeline scripts & paper output generation
├── simulator/             # Timeloop runner & workload simulation
├── tests/                 # Unit and integration tests
├── utils/                 # Device management, runtime evaluation helpers
├── visualization/         # Plotting utilities
├── run_eval.py            # End-to-end evaluation script (smoke & full modes)
├── eval_results.json      # Latest evaluation results
├── REPO_GRAPH.md          # Detailed repository architecture map
└── requirements.txt       # Python dependencies
```

---

## Getting Started

### Prerequisites

- Python ≥ 3.10
- PyTorch ≥ 2.0
- PyTorch Geometric ≥ 2.7
- pymoo ≥ 0.6
- gymnasium ≥ 1.0

### Installation

```bash
# Option A: Conda
conda env create -f environment.yaml
conda activate aging-aware-dnn

# Option B: pip
pip install -r requirements.txt
```

### Quick Smoke Test (< 2 min, CPU)

```bash
python run_eval.py
```

Runs the full pipeline on 200 samples with reduced NSGA-II/PPO iterations.
Validates all 9 pipeline components.

### Full Evaluation (~30 min, CPU)

```bash
python run_eval.py --full
```

Trains on the full 40k dataset with 20 epochs, 20 NSGA-II generations across
5 workloads, and 40 PPO iterations. Results are saved to `eval_results.json`.

### Full Pipeline (Hydra configs)

```bash
python scripts/run_full_pipeline.py --config-name=experiments
```

---

## Improvements Over Baseline

### NSGA-II Enhancements
- **Evaluation cache** — SHA1-hashed mapping deduplication avoids redundant
  simulation (394 cache hits in full run).
- **Convergence callback** — Hypervolume-stagnation early stopping saves
  generations when Pareto front stabilizes.
- **Richer seeding** — 4 initialization strategies (all-to-one, round-robin,
  load-balanced, perturbed round-robin) for diverse starting populations.
- **Aging-variance balance** — Penalty term (`balance_weight=0.3`) on peak aging
  objective encourages even stress distribution across nodes.
- **Modernized RNG** — Replaced deprecated `np.random.seed` with `default_rng`;
  added `uniform_crossover()` and `load_balanced_init()` operators.

### PPO Enhancements
- **Residual policy network** — Pre-norm residual blocks with orthogonal
  initialization for stable deep learning.
- **Running observation normalization** — Welford's online algorithm tracks mean
  and variance across episodes.
- **Linear LR decay** — Learning rate decays from 3×10⁻⁴ to near-zero over
  training.
- **Clipped value loss** — Mirrors policy clipping for value function stability.
- **KL early-stopping** — Breaks PPO epoch if KL divergence exceeds 1.5×target_kl.
- **Entropy annealing** — Coefficient decays linearly (0.01 → 0.001) to shift
  from exploration to exploitation.
- **Periodic eval & checkpointing** — Best policy saved during training.

---

## Simulated Accelerator

| Parameter | Value |
|---|---|
| MAC clusters | 16 × 256 MACs (4,096 total) |
| SRAM banks | 8 |
| NoC routers | 4 |
| Clock | 1.0 GHz |
| Graph nodes | 28 (16 MAC + 8 SRAM + 4 Router) |
| Graph edges | ~92 directed |
| Node features | 8-dimensional |

### Workloads

| Workload | Layers | Type |
|---|---|---|
| ResNet-50 | 14 | Conv2D |
| MobileNetV2 | 22 | Depthwise Conv |
| EfficientNet-B4 | 20 | Mixed |
| BERT-Base | 48 | MatMul |
| ViT-B/16 | 49 | MatMul + Attention |

---

## Ablation Study (from paper)

| Model Variant | R² | MAE |
|---|---|---|
| GCN only | 0.8712 | 0.032 |
| GCN + GAT | 0.9218 | 0.021 |
| GCN + Transformer | 0.9524 | 0.014 |
| **Full (GCN + GAT + Transformer)** | **0.9871** | **0.005** |

---

## Citation

If you use this work, please cite:

```
Adaptive Aging-Aware DNN Accelerator via Predictive Lifetime Management
```

## License

See repository for license details.
