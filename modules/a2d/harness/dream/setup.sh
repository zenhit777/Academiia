#!/usr/bin/env bash
# setup.sh — construction de l'environnement d'évaluation Dream-7B, DANS le dépôt.
#
# Idempotent : peut être rejoué sans effet de bord. Tout ce qui n'est pas
# partageable (venv, clone du fork, caches) vit sous modules/a2d/harness/dream/ :
#
#   modules/a2d/harness/dream/.venv/   environnement Python (torch cu128, transformers 4.46.2, fork lm_eval)
#   modules/a2d/harness/dream/.fork/  clone du fork DreamLM/Dream, epingle par COMMIT
#   modules/a2d/harness/dream/.cache/  HF_HOME, PIP_CACHE_DIR, TMPDIR, XDG_CACHE_HOME
#
# Aucun ecriture hors du depot : les variables de cache sont redirigees vers
# modules/a2d/harness/dream/.cache avant toute operation reseau.
#
# Usage : bash modules/a2d/harness/dream/setup.sh
set -euo pipefail

HARNESS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HARNESS_DIR}/../../../.." && pwd)"

VENV="${HARNESS_DIR}/.venv"
FORK="${HARNESS_DIR}/.fork"
CACHE="${HARNESS_DIR}/.cache"
FORK_URL="https://github.com/DreamLM/Dream.git"
# Commit du fork effectivement utilise pour les resultats J6 (releve dans les
# `results_*.json` : git_hash / upper_git_hash).
FORK_COMMIT="31f94a60d187e3fd481fee3bbc2c732eb94a879c"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# Redirection de TOUS les caches vers le depot (regle « aucune ecriture hors
# du depot ») : ces variables sont exportees, donc elles hereditees par pip,
# git, huggingface_hub et torch.
export HF_HOME="${CACHE}/huggingface"
export HF_DATASETS_CACHE="${CACHE}/huggingface/datasets"
export HUGGINGFACE_HUB_CACHE="${CACHE}/huggingface/hub"
export PIP_CACHE_DIR="${CACHE}/pip"
export XDG_CACHE_HOME="${CACHE}/xdg"
export TMPDIR="${CACHE}/tmp"
mkdir -p "${HF_HOME}" "${PIP_CACHE_DIR}" "${XDG_CACHE_HOME}" "${TMPDIR}"

echo "=== 1/5 clone du fork lm-eval de Dream (${FORK_COMMIT}) ==="
if [ ! -d "${FORK}/.git" ]; then
  git clone "${FORK_URL}" "${FORK}"
fi
git -C "${FORK}" fetch --depth 1 origin "${FORK_COMMIT}" 2>/dev/null || \
  git -C "${FORK}" fetch origin
git -C "${FORK}" checkout --detach "${FORK_COMMIT}"
echo "    fork : $(git -C "${FORK}" rev-parse HEAD)"

echo "=== 2/5 environnement Python (${VENV}) ==="
if [ ! -x "${VENV}/bin/python" ]; then
  "${PYTHON_BIN}" -m venv "${VENV}"
fi
"${VENV}/bin/python" -m pip install -q --upgrade pip setuptools wheel

echo "=== 3/5 torch cu128 (sm_120 / RTX 5090) ==="
"${VENV}/bin/python" -m pip install -q torch==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128

echo "=== 4/5 dependances du harnais ==="
"${VENV}/bin/python" -m pip install -q -r "${HARNESS_DIR}/requirements-dream.txt"

echo "=== 5/5 fork lm-evaluation-harness (editable) ==="
"${VENV}/bin/python" -m pip install -q -e "${FORK}/eval_instruct"

echo "=== verification ==="
"${VENV}/bin/python" - <<'PY'
import torch, transformers, lm_eval
print("torch      ", torch.__version__, "cuda:", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("lm_eval    ", getattr(lm_eval, "__version__", "?"),
      lm_eval.__file__)
PY
echo "OK — environnement prêt dans ${VENV}"
echo "   (caches redirigés vers ${CACHE})"
