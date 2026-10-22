#!/usr/bin/env bash
# diagnose_image.sh — Diagnostic de l'image Docker vLLM SANS GPU ni chargement de modèle.
# UN SEUL docker run --rm --entrypoint bash (sans --gpus) exécutant tous les greps.
#
# Répond aux questions ouvertes de l'incident du 2026-10-18 : la variable
# VLLM_WSL2_ENABLE_PIN_MEMORY est-elle connue de l'image ? Le runner V2 est-il
# le défaut ? Le chemin de diffusion utilise-t-il req_states (structures V2) ?
#
# Usage : ./diagnose_image.sh [image]     (défaut : vllm/vllm-openai:gemma)
# Sortie : rapport structuré sur stdout. NE CONCLUT PAS sur le fonctionnement.
set -euo pipefail

IMAGE="${1:-vllm/vllm-openai:gemma}"
VLLM_PKG="/usr/local/lib/python3.12/dist-packages/vllm"

if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
    echo "-> Image absente localement. Récupération..."
    docker pull "${IMAGE}"
fi

docker run --rm --entrypoint bash "${IMAGE}" -c '
set +e
echo "=== 1. Version vLLM ==="
python3 -c "import vllm; print(vllm.__version__)"
echo
echo "=== 2. envs.py : VLLM_WSL2_ENABLE_PIN_MEMORY / VLLM_WEIGHT_OFFLOADING_DISABLE_UVA / VLLM_USE_V2_MODEL_RUNNER ==="
V=$(grep -n -e VLLM_WSL2_ENABLE_PIN_MEMORY -e VLLM_WEIGHT_OFFLOADING_DISABLE_UVA -e VLLM_USE_V2_MODEL_RUNNER '"$VLLM_PKG"'/envs.py 2>/dev/null)
if [ -n "$V" ]; then echo "$V"; else echo "(aucune des 3 variables, ou envs.py absent)"; fi
echo
echo "=== 3. platforms/cuda.py : WSL2_ENABLE_PIN_MEMORY ==="
C=$(grep -n WSL2_ENABLE_PIN_MEMORY '"$VLLM_PKG"'/platforms/cuda.py 2>/dev/null)
if [ -n "$C" ]; then echo "$C"; else echo "(absent de platforms/cuda.py)"; fi
echo
echo "=== 4. diffusion_gemma.py : req_states (5 premières) ==="
F=$(ls '"$VLLM_PKG"'/model_executor/models/diffusion_gemma.py 2>/dev/null || find '"$VLLM_PKG"' -name "diffusion_gemma.py" 2>/dev/null | head -1)
if [ -n "$F" ]; then
    R=$(grep -n req_states "$F" | head -5)
    if [ -n "$R" ]; then echo "$R"; else echo "(req_states absent de $F)"; fi
else
    echo "(diffusion_gemma.py introuvable)"
fi
echo
echo "=== 5. Verdicts (facts uniquement) ==="
for VAR in VLLM_WSL2_ENABLE_PIN_MEMORY VLLM_WEIGHT_OFFLOADING_DISABLE_UVA VLLM_USE_V2_MODEL_RUNNER; do
    if grep -q -- "$VAR" '"$VLLM_PKG"'/envs.py 2>/dev/null; then
        echo "$VAR : CONNUE de l image (envs.py)"
    else
        echo "$VAR : INCONNUE/ABSENTE de envs.py — la définir na AUCUN effet garanti dans cette image"
    fi
done
if grep -q WSL2_ENABLE_PIN_MEMORY '"$VLLM_PKG"'/platforms/cuda.py 2>/dev/null; then
    echo "platforms/cuda.py : référence WSL2_ENABLE_PIN_MEMORY (logique WSL2 présente)"
else
    echo "platforms/cuda.py : NE référence PAS WSL2_ENABLE_PIN_MEMORY (mémoire épinglée reste désactivée sous WSL2 -> UVA indisponible -> runner V2 échoue probablement à l init)"
fi
if grep -q req_states "$F" 2>/dev/null; then
    echo "req_states : PRÉSENT dans diffusion_gemma.py (structure du runner V2 utilisée par le chemin diffusion)"
else
    echo "req_states : ABSENT de diffusion_gemma.py (ou fichier introuvable)"
fi
'
echo
echo "Prochaine étape (GPU requis) : si VLLM_WSL2_ENABLE_PIN_MEMORY est CONNUE, lancer le compose par défaut (runner V2)."
echo "Sinon : image plus récente ou machine Linux native. Voir infra/vllm/README.md (arbre de décision)."
