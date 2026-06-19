#!/bin/bash
# Discover the correct SLURM partition + account for this PARAM site.
# Run this on the login node BEFORE sbatch:
#   bash scripts/param_discover.sh
#
# It prints the GPU partitions you can use and the account(s) tied to your
# user, then shows the exact sbatch command to copy-paste.
set -uo pipefail

USER_NAME="${USER:-$(whoami)}"

echo "================================================================"
echo " PARAM SLURM discovery for user: $USER_NAME"
echo "================================================================"

echo
echo "==> All partitions (look for ones with gpu/A100 in the name):"
sinfo -o "%P %a %l %D %G %N" 2>/dev/null || echo "  (sinfo unavailable)"

echo
echo "==> Partitions that advertise a GPU gres:"
sinfo -o "%P %G" 2>/dev/null | grep -i gpu || echo "  (none found via gres column; check the full list above)"

echo
echo "==> Your SLURM accounts / associations:"
if command -v sacctmgr >/dev/null 2>&1; then
  sacctmgr -nP show assoc where user="$USER_NAME" format=Account,Partition,QOS 2>/dev/null \
    || sacctmgr show user "$USER_NAME" 2>/dev/null \
    || echo "  (no association info returned)"
else
  echo "  (sacctmgr unavailable)"
fi

echo
echo "==> Default account from sshare (if configured):"
sshare -U -u "$USER_NAME" 2>/dev/null | head -n 5 || echo "  (sshare unavailable)"

echo
echo "================================================================"
echo " Next step: submit with YOUR partition + account, e.g.:"
echo
echo "   sbatch --partition=<PART> --account=<ACCT> scripts/param_a100_full.slurm"
echo
echo " (CLI flags override the #SBATCH defaults in the script, so you do"
echo "  NOT need to edit the .slurm files.)"
echo "================================================================"
