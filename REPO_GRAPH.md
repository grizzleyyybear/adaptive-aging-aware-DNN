# Repository Graph — Adaptive Aging-Aware DNN Accelerator

> **Generated:** 2025-04-05 | **Last updated:** 2025-06-13 | **Purpose:** Codebase reference map for audits & onboarding

---

## 1. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         CONFIGURATION (Hydra/OmegaConf)                │
│           configs/ ─ accelerator · workloads · training · experiments  │
└──────┬──────────────────────────────┬───────────────────────┬──────────┘
       │                              │                       │
       ▼                              ▼                       ▼
┌──────────────┐  ┌──────────────────────────┐  ┌────────────────────────┐
│ AGING MODELS │  │      SIMULATOR           │  │  GRAPH                 │
│  NBTI · HCI  │  │  TimeloopRunner          │  │  AcceleratorGraph      │
│  TDDB        │  │  WorkloadRunner           │  │  AgingDataset (PyG)    │
│  AgingLabel  │  │  (analytical roofline)    │  │  (N,8) features        │
│  Generator   │  └────────────┬─────────────┘  └──────────┬─────────────┘
└──────┬───────┘               │                            │
       │            ┌──────────┘                            │
       ▼            ▼                                       ▼
┌──────────────────────────┐                   ┌────────────────────────┐
│  FEATURES                │                   │  MODELS (ML)           │
│  ActivityExtractor       │──────────────────►│  HybridGNNTransformer  │
│  FeatureBuilder          │                   │  TrajectoryPredictor   │
│  [N,8] node features     │                   │  TrainingPipeline      │
└──────────────────────────┘                   └──────────┬─────────────┘
                                                          │
       ┌──────────────────────────────────────────────────┤
       ▼                              ▼                   ▼
┌──────────────────┐  ┌───────────────────────┐  ┌───────────────────┐
│  RL              │  │  OPTIMIZATION         │  │  PLANNING         │
│  AgingControlEnv │  │  NSGA2Optimizer       │  │  LifetimePlanner  │
│  ActorCritic     │  │  MappingProblem       │  │  Budget alloc.    │
│  PPOTrainer      │  │  MappingChromosome    │  │  Violation detect │
└────────┬─────────┘  └──────────┬────────────┘  └─────────┬─────────┘
         │                       │                          │
         ▼                       ▼                          │
┌─────────────────┐  ┌───────────────────────┐              │
│  SCHEDULER      │  │  EVALUATION           │◄─────────────┘
│  RuntimeMapper  │  │  PerformanceMetrics   │
│  (dispatch)     │  │  ReliabilityMetrics   │
└─────────────────┘  │  StatisticalTests     │
                     └──────────┬────────────┘
                                │
                                ▼
                     ┌───────────────────────┐
                     │  VISUALIZATION        │
                     │  Heatmaps · Pareto    │
                     │  Trajectories · Arch. │
                     └───────────────────────┘
```

---

## 2. Module Dependency Graph

```
aging_models ◄──────────────────────────────────────────┐
  │                                                      │
  ▼                                                      │
features ◄── simulator                                   │
  │              │                                       │
  ▼              ▼                                       │
graph ─────► models ──► utils.device                     │
  │              │                                       │
  ├──────────────┤                                       │
  ▼              ▼                                       │
utils.runtime_eval ◄────────────────────────────┐        │
  │                                              │        │
  ├───────────► planning                         │        │
  │                                              │        │
  ├───────────► rl ──────────────────────────────┤        │
  │               ├─► simulator                  │        │
  │               ├─► features                   │        │
  │               ├─► aging_models ──────────────┼────────┘
  │               ├─► graph                      │
  │               └─► planning                   │
  │                                              │
  ├───────────► optimization                     │
  │               ├─► simulator                  │
  │               ├─► features                   │
  │               └─► graph                      │
  │                                              │
  └───────────► scheduler                        │
                  └─► optimization               │
                                                 │
visualization                                    │
  ├─► graph                                      │
  └─► optimization                               │
                                                 │
evaluation (standalone: numpy, scipy, pandas) ───┘
```

---

## 3. Directory & File Map

### `aging_models/` — Physics-based Transistor Aging

| File | Key Exports | Description |
|------|------------|-------------|
| `nbti_model.py` | `NBTIModel` | Negative Bias Temperature Instability: ΔVth = A × (activity × stress)ⁿ |
| `hci_model.py` | `HCIModel` | Hot Carrier Injection: ΔId/Id = B × current_densityᵐ × √stress |
| `tddb_model.py` | `TDDBModel` | Time-Dependent Dielectric Breakdown: Weibull failure probability |
| `aging_label_generator.py` | `AgingLabelGenerator` | Combines NBTI+HCI+TDDB → unified aging score [0,1] |

### `models/` — Neural Network Architectures

| File | Key Exports | Description |
|------|------------|-------------|
| `hybrid_gnn_transformer.py` | `HybridGNNTransformer`, `PositionalEncoding` | GCN→GAT→Transformer encoder, sigmoid output [0,1] |
| `trajectory_predictor.py` | `TrajectoryPredictor` | Multi-step future aging prediction with variance-aware loss |
| `training_pipeline.py` | `TrainingPipeline` | Train/eval loop; early stopping monitors R² for trajectory / loss for predictor; optional `checkpoint_dir`; WandB logging |

**Deps:** torch, torch_geometric (GATConv, GCNConv), sklearn, wandb, omegaconf

### `graph/` — Hardware Graph Representation

| File | Key Exports | Description |
|------|------------|-------------|
| `accelerator_graph.py` | `AcceleratorGraph` | NetworkX topology builder (MAC→SRAM→Router mesh), PyG conversion |
| `graph_dataset.py` | `AgingDataset` | PyG InMemoryDataset; synthetic [N,8] features, trajectory labels |

**Dataset sample schema:**
- `x`: [N, 8] — switching_act, compute_util, mem_rate, duty_cycle, temp_proxy, node_type, workload_type, stress_time
- `edge_index`: [2, E], `edge_attr`: [E, 2]
- `y`: [N, 1] current aging, `y_trajectory`: [N, 10] future aging
- `workload_emb`: [5], `mapping`: [64], `stress_time`: [1], `latency`, `energy`

### `rl/` — Reinforcement Learning

| File | Key Exports | Description |
|------|------------|-------------|
| `environment.py` | `AgingControlEnv` | Gymnasium env; Discrete(5) actions; composite reward |
| `policy_network.py` | `ActorCritic` | Shared-trunk PPO policy + value heads |
| `trainer.py` | `PPOTrainer` | GAE advantage estimation, policy clipping, entropy reg. |

**Action space:** 0=No-op, 1=Balance-load, 2=Rotate-all, 3=Half-rotate, 4=Planner-recommend
**Reward:** 0.5×aging_improve + 0.3×balance + 0.2×latency_improve

### `simulator/` — Hardware Performance Simulation

| File | Key Exports | Description |
|------|------------|-------------|
| `timeloop_runner.py` | `AnalyticalSimulator`, `AcceleratorConfig`, `LayerSpec`, `SimResult` | Roofline-based latency/energy/utilization model |
| `workload_runner.py` | `WorkloadRunner` | Generates layer specs for 5 DNN architectures |

**Supported workloads:** ResNet-50, BERT-Base, MobileNetV2, EfficientNet-B4, ViT-B/16

### `features/` — Feature Extraction

| File | Key Exports | Description |
|------|------------|-------------|
| `feature_builder.py` | `FeatureBuilder` | Constructs [N,8] node feature matrices |
| `activity_extractor.py` | `ActivityExtractor` | Normalizes SimResult → activity dicts with temp proxies |

### `optimization/` — Multi-Objective Optimizer

| File | Key Exports | Description |
|------|------------|-------------|
| `chromosome_representation.py` | `MappingChromosome` | Layer→Cluster encoding, crossover, mutation, repair |
| `nsga2_optimizer.py` | `NSGA2Optimizer`, `MappingProblem`, `ParetoSolution` | pymoo NSGA-II; 3 objectives: peak_aging, latency, energy |

### `planning/` — Lifetime Budget Allocation

| File | Key Exports | Description |
|------|------------|-------------|
| `lifetime_planner.py` | `LifetimePlanner` | Budget strategies: equalized / type-weighted / capacity-weighted |

### `evaluation/` — Metrics & Statistics

| File | Key Exports | Description |
|------|------------|-------------|
| `performance_metrics.py` | `PerformanceMetrics` | Speedup, energy efficiency, lifetime extension, throughput |
| `reliability_metrics.py` | `ReliabilityMetrics` | Peak aging, variance, hotspot count, TTF, lifetime improvement |
| `statistical_tests.py` | `StatisticalTests` | Paired t-test, Cohen's d, confidence intervals, pandas→LaTeX |

### `scheduler/` — Runtime Dispatch

| File | Key Exports | Description |
|------|------------|-------------|
| `runtime_mapper.py` | `RuntimeMapper` | Converts NSGA-II results → executable layer→cluster trace |

### `visualization/` — Plotting

| File | Key Exports | Description |
|------|------------|-------------|
| `aging_heatmap.py` | `plot_aging_heatmap` | Spatial graph colored by aging scores |
| `trajectory_plots.py` | `plot_aging_trajectories`, `plot_lifetime_comparison_bar` | Temporal trends, bar charts |
| `pareto_plots.py` | `plot_pareto_2d`, `plot_pareto_3d` | Pareto frontier scatter plots |
| `architecture_diagrams.py` | `render_architecture_block_diagram` | System-level block diagram |

### `utils/` — Shared Utilities

| File | Key Exports | Description |
|------|------------|-------------|
| `device.py` | `get_device_request`, `resolve_device`, `configure_torch_runtime` | PyTorch device management (CPU/CUDA) |
| `runtime_eval.py` | `simulate_mapping`, `compute_physics_ttf`, `compute_predictor_ttf`, `cfg_get` | Central orchestrator for simulation + TTF computation |

### `experiments/` — Experimental Frameworks

| File | Key Exports | Description |
|------|------------|-------------|
| `baseline_experiments.py` | `run_all_baselines` | No-opt, round-robin, random baselines |
| `ablation_studies.py` | `run_ablation_studies` | Component ablation (w/o GNN, w/o Transformer, etc.) |

### `scripts/` — Entry Points

| File | Description |
|------|-------------|
| `run_full_pipeline.py` | End-to-end Hydra pipeline: train GNN → RL → NSGA-II → paper outputs |
| `train_trajectory_v2.py` | TrajectoryPredictor training |
| `train_real.py` | Alternative training entry |
| `run_nsga_ppo.py` | Combined NSGA-II + PPO training |
| `generate_paper_outputs.py` | LaTeX tables & publication figures |

### `configs/` — YAML Configuration

| File | Description |
|------|-------------|
| `accelerator.yaml` | Hardware topology (clusters, SRAM, NoC) |
| `workloads.yaml` | Workload specifications |
| `training.yaml` | Model hyperparameters |
| `experiments.yaml` | Full experiment config |
| `smoke_test.yaml` | Minimal config for quick validation |

### `tests/` — Test Suite

| File | Validates |
|------|-----------|
| `conftest.py` | Pytest fixtures (base_config) |
| `test_aging_models.py` | NBTI/HCI/TDDB physics correctness |
| `test_accelerator_graph.py` | Graph construction & PyG conversion |
| `test_dataset.py` | AgingDataset generation & caching |
| `test_hybrid_model.py` | GNN-Transformer forward/backward |
| `test_trajectory_predictor.py` | Multi-step trajectory predictions |
| `test_simulator.py` | TimeloopRunner roofline model |
| `test_rl_env.py` | Gymnasium env compliance |
| `test_nsga2.py` | NSGA-II multi-objective search |
| `test_statistical_tests.py` | t-tests, confidence intervals |
| `test_full_pipeline.py` | End-to-end integration |

---

## 4. Data Flow Pipeline

```
1. CONFIGURE        configs/*.yaml → OmegaConf DictConfig
                         │
2. SIMULATE          WorkloadRunner.get_workload_layers()
                         │
                    AnalyticalSimulator.run_workload()
                         │
                    SimResult (latency, energy, utilization, switching_activity)
                         │
3. EXTRACT           ActivityExtractor.extract_activities()
                         │
                    FeatureBuilder.build_node_features() → [N, 8] tensor
                         │
4. BUILD GRAPH       AcceleratorGraph.build() → NetworkX graph
                         │
                    AcceleratorGraph.to_pyg() → PyG Data
                         │
5. COMPUTE AGING     AgingLabelGenerator.compute_aging_score() → [N, 1]
                         │
                    AgingLabelGenerator.generate_trajectory_labels() → [N, 10]
                         │
6. TRAIN MODEL       TrainingPipeline.train(AgingDataset)
                         │
                    HybridGNNTransformer → node aging predictions
                    TrajectoryPredictor → multi-step aging forecasts
                         │
7. OPTIMIZE          NSGA2Optimizer.optimize()
                         │
                    ParetoSolution[] (mapping, peak_aging, latency, energy)
                         │
8. PLAN              LifetimePlanner.allocate_budgets()
                         │
                    Budget thresholds per node
                         │
9. RL CONTROL        PPOTrainer.train(AgingControlEnv)
                         │
                    Runtime action policy (balance, rotate, rebalance)
                         │
10. DISPATCH          RuntimeMapper.dispatch() → execution trace
                         │
11. EVALUATE          PerformanceMetrics + ReliabilityMetrics + StatisticalTests
                         │
12. VISUALIZE         Heatmaps, trajectories, Pareto fronts, architecture diagrams
```

---

## 5. External Dependencies

| Library | Purpose | Used By |
|---------|---------|---------|
| **torch** | Deep learning | models, rl, graph, features, utils |
| **torch_geometric** | Graph neural networks | models, graph |
| **pymoo** | NSGA-II optimization | optimization |
| **gymnasium** | RL environment API | rl |
| **networkx** | Graph construction | graph, visualization |
| **numpy** | Numerics everywhere | all modules |
| **scipy** | Statistical tests | evaluation |
| **pandas** | DataFrames, LaTeX export | evaluation |
| **matplotlib** | Plotting | visualization |
| **omegaconf** | Config management | graph, models, optimization, planning, rl |
| **hydra** | Experiment launcher | scripts |
| **wandb** | Experiment tracking | models, rl |
| **sklearn** | MAE, RMSE, R² metrics | models |
| **tqdm** | Progress bars | graph |

---

## 6. Root-Level Test & Utility Files

| File | Purpose |
|------|---------|
| `smoke_step1.py` | Smoke: graph construction → PyG |
| `smoke_step2.py` | Smoke: GNN-Transformer forward pass |
| `test_ds.py` | Dataset feature/shape validation |
| `test_gym.py` | RL env Gymnasium compliance |
| `test_gym_current.py` | Simplified RL env test |
| `test_hybrid.py` | GNN-Transformer train loop |
| `test_metrics.py` | Performance metrics on dummy data |
| `test_sim.py` | Simulator latency/energy checks |
| `test_traj.py` | Trajectory predictor forward + loss |
| `audit.py` | Import/package diagnostic → `diagnostic_data.json` |
| `generate_report.py` | Audit + test results → `DIAGNOSTIC_REPORT.md` |
| `run_tests.py` | Pytest runner → `pytest_clean.log` |
| `run_test_ds.py` | Dataset smoke test wrapper |
| `run_eval.py` | **Primary eval script** — smoke (`--smoke`) + full (`--full`) modes; 6-stage pipeline: graph→dataset→predictor→trajectory→NSGA-II→PPO; accepts `--ckpt-dir` |
| `paper_comparison.py` | Paper vs implementation comparison — ASCII report + matplotlib 3-panel figure (`paper_comparison.png`); loads `eval_results.json` |
| `eval_results.json` | Latest evaluation results (Predictor R²=0.9982, Trajectory R²=0.7718, PPO best=+0.585) |

---

## 7. Key Design Patterns

| Pattern | Implementation |
|---------|----------------|
| **Config management** | OmegaConf + Hydra YAML; flexible `cfg_get()` accessor |
| **Hardware abstraction** | AcceleratorGraph (NetworkX) → PyG Data conversion |
| **Analytical simulation** | Roofline model replaces Timeloop for research speed |
| **Feature pipeline** | SimResult → ActivityExtractor → FeatureBuilder → [N,8] tensor |
| **Multi-objective opt** | pymoo NSGA-II with integer repair, diverse seeding |
| **RL env design** | Gymnasium-compliant, composite reward, 5 discrete actions |
| **Model architecture** | GCN (residual) → GAT (attention) → Transformer → sigmoid head |
| **Trajectory learning** | Encoder output + current prediction → multi-horizon forecast |
| **Budget planning** | Type/capacity-weighted threshold allocation |
| **Paper-ready output** | Matplotlib figures + pandas→LaTeX tables |

---

## 8. Quick Audit Checklist

- [ ] **Aging models:** Do NBTI/HCI/TDDB formulas match paper equations?
- [ ] **Graph construction:** Do node counts match accelerator config?
- [ ] **Dataset:** Are features in [0,1]? No NaN/Inf? Correct shapes?
- [ ] **Model:** Does GNN-Transformer produce [N,1] output in [0,1]?
- [ ] **Trajectory:** Does predictor produce [N, horizon] with valid values?
- [ ] **Optimizer:** Does NSGA-II converge? Are Pareto solutions diverse?
- [ ] **RL env:** Does it pass `gymnasium.utils.env_checker.check_env()`?
- [ ] **Planning:** Do budgets sum correctly? Are violations detected?
- [ ] **Evaluation:** Do metrics match expected ranges?
- [ ] **Configs:** Are all YAML fields consumed? No unused/missing keys?
- [ ] **Tests:** Do all `tests/` pass? Are root-level smoke tests green?
- [ ] **Imports:** Do all `__init__.py` exports resolve correctly?
