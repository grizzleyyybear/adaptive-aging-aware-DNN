# Training plan

Everything in the paper that can be measured without training has been measured
(`scripts/verify_results.py`, `scripts/measure_paper_figures.py`,
`scripts/extended_analysis.py`, `scripts/variation_scoping.py`). The runs below are what is
left. Each one closes a named limitation in Section VI of `paper/main.tex`. They are listed
in priority order.

Every command writes to its own `--ckpt-dir` and `--results-path`, so the released
`checkpoints/` and `eval_results.json` are never overwritten. All new flags default to the
released configuration. All commands were smoke-tested on tiny settings, so the code paths
run, but no results from them exist yet.

## Common settings

The released run (README "Reproduce") used:

```bash
BASE="--quality --device cuda --gpu-4gb --dataset-size 1500 \
      --pred-epochs 60 --pred-patience 16 --traj-epochs 75 --traj-patience 18 \
      --nsga-pop 20 --nsga-gen 20 --ppo-iters 30 --ppo-steps 32"
```

New flags in `run_eval.py`:

| Flag | Meaning |
|---|---|
| `--seed N` | Model initialisation, training and PPO seed (default 42). |
| `--data-seed N` | Synthetic dataset draw (default 42). Each value gets its own dataset cache file. |
| `--components gcn,gat,transformer` | Architecture ablation. `none` means input projection plus MLP head only. |
| `--predict-only` | Stop after the predictor and trajectory stages. Use for seed, ablation and scaling runs. |
| `--ppo-use-predictor` | The PPO environment uses the learned predictor and trajectory head instead of the physics model and a heuristic forecast. |
| `--ppo-hidden N` | PPO actor-critic width (default 128). |
| `--aging-variation S` | Per-node lognormal process variation in the physics labels. Existed before; it is now part of the cache key, so it no longer silently reuses the no-variation dataset. |
| `--pe-array R,C` | PE array of the simulated accelerator, shared by data generation and runtime (default `4,4`, the released data). Non-default values get their own dataset cache file. |

Measured cost on the RTX 3050 laptop (idle):

- **Training:** about 1.5 s per epoch per model at 1,500 samples, plus about 15 s start-up. A
  `--predict-only` run with the released epoch limits takes at most about 4 minutes.
- **PPO:** about 1.5 s per 100 environment steps.
- **Analysis:** `extended_analysis.py` takes about 35 minutes.

Everything below fits in an afternoon, except E3 at 15,000 samples (about 10× data generation
and epoch time).

## E0: Regenerate data and retrain on the intended accelerator (do this first)

The released dataset was simulated on a **4×4** PE array. `AgingDataset`'s defaults
`pe_array_rows`/`pe_array_cols = 4` overrode the 16×16 `pe_array`, while NSGA-II and the PPO
environment in the released run simulated **16×16**. The released NSGA-II therefore fed the
predictor out-of-distribution features: MAC latency 10–15× off. This is fixed: the PE array is
now one explicit setting (`--pe-array`) used everywhere and part of the cache key. The paper's
no-training results were re-run on the 4×4 configuration the released model was trained on.

To get the accelerator the paper originally intended, retrain everything on 16×16:

```bash
python run_eval.py $BASE --pe-array 16,16 \
  --ckpt-dir runs/pe16_s42 --results-path runs/pe16_s42/eval_results.json
python scripts/extended_analysis.py --ckpt-dir runs/pe16_s42 --pe-array 16,16 --tag pe16
python scripts/variation_scoping.py --pe-array 16,16
```

**Decide once:** either keep 4×4 as the paper's accelerator, which is consistent today, or
switch the paper to 16×16 after E0. In that case add `--pe-array 16,16` to every command
below, including `eval_ppo_policy.py` and `extended_analysis.py`.

## E1: Seed variance (closes "single seed")

```bash
for s in 1 2 3 4 5; do
  python run_eval.py $BASE --predict-only --seed $s \
    --ckpt-dir runs/seed_$s --results-path runs/seed_$s/eval_results.json
done
```

Optional data variance: repeat with `--data-seed $s` as well.

**Report:** mean ± s.d. over the five seeds for predictor and trajectory R², MAE, RMSE and ρ.
These replace the single-seed values in Table I and the abstract. Keep the bootstrap CIs in
`paper/tables/extended.json` as the within-run uncertainty.

## E2: Architecture ablation (closes "no ablation")

```bash
for c in none gcn gat transformer gcn,gat gcn,transformer gat,transformer gcn,gat,transformer; do
  for s in 1 2 3; do
    python run_eval.py $BASE --predict-only --components $c --seed $s \
      --ckpt-dir runs/abl_${c//,/+}_$s --results-path runs/abl_${c//,/+}_$s/eval_results.json
  done
done
```

**Report:** an ablation table (mean ± s.d. over 3 seeds) of R² and MAE for the predictor and
trajectory head.

The `none` row (MLP, no message passing) is the one reviewers will look at first. The labels
are per-node functions of per-node features, so an MLP may match the full model. If it does,
say so: the graph layers are then justified only if a harder label model (E4, E6) needs them.

## E3: Data scaling (closes "small dataset")

```bash
for n in 500 1500 5000 15000; do
  python run_eval.py $BASE --predict-only --dataset-size $n \
    --ckpt-dir runs/size_$n --results-path runs/size_$n/eval_results.json
done
```

The first run at each size also generates and caches the dataset (simulation only). On a 4 GB
GPU keep `--gpu-4gb`, because batch size is already 16.

**Report:** a learning curve of test R² and MAE against training-set size.

## E4: Variation-aware surrogate for NSGA-II (the main open research result)

**Why this matters.** Without process variation, every MAC cluster ages identically. After
round-robin removes the MAC hotspot, the physics peak sits on a NoC router at 0.3565 that no
mapping changes. So every spreading strategy ties: round-robin, random search, surrogate
NSGA-II and physics NSGA-II all reach 21.6–21.7% (`paper/tables/extended.json`,
`mapping_study`). The released surrogate's extra "predicted" gains (up to 38.6%) are model
error. `scripts/variation_scoping.py` shows that once nodes differ, NSGA-II on the physics
model beats round-robin:

| Variation σ | NSGA-II (physics) vs round-robin |
|---|---|
| 0.1 | +8.1% ± 5.9 |
| 0.2 | +15.3% ± 11.0 |

These are over 3 simulated chips × 5 workloads; the results are identical for 4×4 and 16×16 PE
arrays, and the gain is up to 25% on some chips. The open question is whether a *learned*
surrogate captures this.

```bash
for v in 0.1 0.2; do
  python run_eval.py $BASE --predict-only --aging-variation $v \
    --ckpt-dir runs/var${v}_s42 --results-path runs/var${v}_s42/eval_results.json
  python scripts/extended_analysis.py --ckpt-dir runs/var${v}_s42 --aging-variation $v --tag var$v
done
```

**Report:** the `mapping_study` block of `paper/tables/extended_var*.json`, which scores each
mapping with the physics model:

- NSGA-II with the new surrogate,
- round-robin,
- same-budget random search,
- NSGA-II on the physics model directly.

**Success criteria (decision quality, not node R²):**

1. Surrogate NSGA-II beats round-robin by a clear margin, approaching the physics NSGA-II gain
   above.
2. `surrogate_fidelity.spearman_peak` is high.
3. The predicted reduction is close to the physics reduction, with no exploitation gap like the
   released 38.6% vs 21.6%.

If (1) fails while node R² stays near 0.99, train with a peak-aware loss (for example, extra
weight on each graph's maximum node), or add structured mappings (concentrated, round-robin) to
the training mix. Today the dataset only contains uniform random mappings
(`graph/graph_dataset.py`, the "Random mapping" step).

**Limitation of this design.** Variation is fixed per simulated chip (`variation_seed`), so this
surrogate learns one chip's map through node positions. To generalise across chips, add the
per-node variation multiplier as a node feature: in a real system it would come from on-die
aging monitors such as ring oscillators. That is E6c.

## E5: PPO with a real budget (closes "PPO negative result")

Run E5 in the variation setting from E4. Without variation, the greedy one-step oracle scores
1.572, the same as always-rotate at 1.574 (`extended.json`, `runtime_policies_episode_sum`).
There is no headroom, so no learned controller can win there. PPO costs only about 1.5 s per 100
steps, so you can raise `--ppo-iters` to 1000 (64k steps) if 300 has not converged.

```bash
for s in 1 2 3; do
  # (a) physics-model environment
  python run_eval.py $BASE --ppo-only --aging-variation 0.1 --seed $s \
    --ppo-iters 300 --ppo-steps 64 \
    --ckpt-dir runs/ppo_var0.1_s$s --results-path runs/ppo_var0.1_s$s/eval_results.json
  python scripts/eval_ppo_policy.py --ckpt-dir runs/ppo_var0.1_s$s --aging-variation 0.1 \
    --out runs/ppo_var0.1_s$s/policy_eval.json
done
```

Step (b) uses the learned predictor and trajectory head in the environment. First copy the E4
checkpoints into each PPO checkpoint directory:

```bash
for s in 1 2 3; do
  mkdir -p runs/ppo_pred_var0.1_s$s
  cp runs/var0.1_s42/predictor_best.pt runs/var0.1_s42/trajectory_best.pt runs/ppo_pred_var0.1_s$s/
  python run_eval.py $BASE --ppo-only --ppo-use-predictor --aging-variation 0.1 --seed $s \
    --ppo-iters 300 --ppo-steps 64 \
    --ckpt-dir runs/ppo_pred_var0.1_s$s --results-path runs/ppo_pred_var0.1_s$s/eval_results.json
  python scripts/eval_ppo_policy.py --ckpt-dir runs/ppo_pred_var0.1_s$s --use-predictor --aging-variation 0.1 \
    --out runs/ppo_pred_var0.1_s$s/policy_eval.json
done
```

With `--ppo-only --ppo-use-predictor`, `run_eval.py` loads `predictor_best.pt` and
`trajectory_best.pt` from `--ckpt-dir`, so the environment uses the learned aging estimate and
the learned 10-step forecast.

The budget is 300 × 64 = 19,200 environment steps per seed, 20× the released run. Increase it
if the learning curve has not flattened.

**Report:** greedy-policy return over 20 seeded episodes against Random, Hold, Rotate and the
greedy one-step oracle (`eval_ppo_policy.py` prints all five), mean ± s.d. over 3 seeds.

**Success criterion:** PPO beats Rotate and closes part of the gap to the oracle.

## E6: Model and label upgrades (optional, require code changes first)

- **a. Use edge attributes.** Pass `edge_weight` to the GCN layers and set `edge_dim` on
  `GATConv` in `models/hybrid_gnn_transformer.py`. Today `edge_attr` is ignored.
- **b. Real duty-cycle feature.** Feature 3 in `features/feature_builder.py` duplicates
  feature 0. Replace it with busy-cycles / total-cycles per unit from the simulator, or drop it
  (`FEATURE_DIM` 8 → 7).
- **c. Per-node variation as an input feature** (see E4). Expose the multiplier from
  `AgingLabelGenerator._variation_scales` and append it in `graph/graph_dataset.py`. Then
  train with several `variation_seed` chips so the model generalises across dies.
- **d. Time-dependent TDDB.** `aging_models/tddb_model.py` has no time dependence and
  contributes < 4e-4. Use a Weibull form F = 1 − exp(−(t/η)^β) with field-dependent η.

- **e. Mapping-dependent NoC traffic.** In `simulator/timeloop_runner.py` router traffic does
  not depend on which cluster runs which layer, which pins peak aging at the router floor.
  Route each layer's traffic through the routers that serve its cluster. MAC cluster *m* →
  SRAM *m* mod 8 → router (*m* mod 8) mod 4, as in `graph/accelerator_graph.py`. Without
  this, mapping only matters when some MAC cluster's aging exceeds the router floor, which
  happens with process variation (E4) but never without it.

Each of these changes the dataset or the model, so re-run E1 and E2 afterwards.

## Not fixable by training

- **Silicon calibration.** The NBTI/HCI coefficients need fitting to measured degradation
  data (foundry reliability reports or ring-oscillator measurements).
- **Real workloads.** `run_eval.py` already accepts
  `--dataset-source imported_trace --workload-traces ... --activity-traces ...` for Timeloop
  or Accelergy traces. Real per-layer traces would replace the reduced 12–15-layer workloads.

## After the runs

1. Put the new numbers into `paper/main.tex` (Tables I–II, Section V, the abstract) and delete
   the limitation paragraphs each experiment closes.
2. Re-run `scripts/verify_results.py --ckpt-dir ... --results ...` for any checkpoint you
   promote to `checkpoints/`. Add the run's `--pe-array` / `--aging-variation`. It checks
   full-architecture runs only; it does not handle `--components` ablations.
3. Rebuild the PDF: `cd paper && tectonic main.tex`.
