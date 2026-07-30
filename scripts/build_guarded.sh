#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════
#  build_guarded.sh — lance un build sous plafond mémoire **dur** (cgroup).
#
#  Pourquoi : le 30/07/2026, un build a saturé la RAM et la machine a gelé
#  (thrashing, OOM-killer trop lent). Ici, le build tourne dans un scope
#  systemd avec MemoryMax : s'il dépasse, **seul le build est tué**, la
#  session graphique n'est jamais touchée.
#
#  Usage :
#     scripts/build_guarded.sh [args de « galsenai-sft build »]
#     MEM_MAX=6G scripts/build_guarded.sh --limit 100
#
#  Variables :
#     MEM_MAX   plafond dur      (défaut : 60 % de la RAM totale)
#     MEM_HIGH  seuil de freinage (défaut : 75 % de MEM_MAX)
#     SWAP_MAX  swap autorisé    (défaut : 0 — le swap = la cause du gel)
# ════════════════════════════════════════════════════════════════════════
set -euo pipefail

cd "$(dirname "$0")/.."

total_mb=$(awk '/MemTotal/ {print int($2/1024)}' /proc/meminfo)
default_max=$((total_mb * 60 / 100))
MEM_MAX="${MEM_MAX:-${default_max}M}"
MEM_HIGH="${MEM_HIGH:-$((default_max * 75 / 100))M}"
SWAP_MAX="${SWAP_MAX:-0}"

run=(uv run galsenai-sft build "$@")

if ! command -v systemd-run >/dev/null 2>&1; then
  echo "⚠ systemd-run absent : repli sur ulimit (protection partielle)." >&2
  ulimit -v $(( ${MEM_MAX%M} * 1024 )) || true
  exec "${run[@]}"
fi

echo "▶ build sous cgroup : MemoryMax=${MEM_MAX} MemoryHigh=${MEM_HIGH} SwapMax=${SWAP_MAX}"
echo "  (RAM totale ${total_mb} Mo — la session reste protégée quoi qu'il arrive)"

set +e
systemd-run --user --scope --collect \
  --unit "galsenai-build-$$" \
  -p MemoryMax="${MEM_MAX}" \
  -p MemoryHigh="${MEM_HIGH}" \
  -p MemorySwapMax="${SWAP_MAX}" \
  -p CPUWeight=50 \
  -- "${run[@]}"
code=$?
set -e

if [ "$code" -eq 137 ] || [ "$code" -eq 9 ]; then
  echo "✖ build tué par le plafond mémoire (${MEM_MAX}). La machine, elle, va bien." >&2
  echo "  → relance avec --limit, ou MEM_MAX plus haut si la RAM le permet." >&2
fi
exit "$code"
