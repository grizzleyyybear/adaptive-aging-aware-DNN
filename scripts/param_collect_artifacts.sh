#!/bin/bash
# Bundle the generated artifacts from a PARAM run into a single tarball you can
# copy back to your laptop with scp/rsync.
#
#   bash scripts/param_collect_artifacts.sh [output_name]
#
# Produces: param_artifacts_<timestamp>.tar.gz
set -euo pipefail

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${1:-param_artifacts_${STAMP}.tar.gz}"

FILES=()
for f in \
  eval_results_a100_full.json \
  eval_results_a100_enhanced.json \
  eval_results.json \
  figures \
  paper/tables \
  paper/assets \
  checkpoints_a100_full \
  checkpoints_a100_enhanced \
  runs ; do
  [[ -e "$f" ]] && FILES+=("$f")
done

if [[ ${#FILES[@]} -eq 0 ]]; then
  echo "No artifacts found. Did the job run?"
  exit 1
fi

echo "==> Packing: ${FILES[*]}"
tar -czf "$OUT" "${FILES[@]}"
echo "==> Wrote $OUT"
echo "Copy it back with:  scp <user>@<param-host>:$(pwd)/$OUT ."
