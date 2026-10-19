#!/usr/bin/env bash
# diagnose_image.sh — Diagnostic de l'image Docker vLLM SANS GPU ni chargement de modèle.
#
# Répond aux questions ouvertes de l'incident du 2026-10-18 (crash de la 1re
# génération sous WSL2) : la variable VLLM_WSL2_ENABLE_PIN_MEMORY est-elle
# connue de l'image ? Le runner V2 est-il le défaut ? Le chemin de diffusion
# utilise-t-il req_states/draft_tokens (structures V2) ?
#
# Usage : ./diagnose_image.sh [image]     (défaut : vllm/vllm-openai:gemma)
# Sortie : rapport sur stdout. NE CONCLUT PAS sur le fonctionnement —
# le verdict est à confronter au serveur réel (GPU requis).
set -euo pipefail

IMAGE="${1:-vllm/vllm-openai:gemma}"
# Chemins de vLLM dans l'image officielle (Python 3.12, dist-packages).
VLLM_PKG="/usr/local/lib/python3.12/dist-packages/vllm"

echo "=== Diagnostic image : ${IMAGE} ==="

if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
    echo "-> Image absente localement. Récupération..."
    docker pull "${IMAGE}"
fi

# docker run --rm --entrypoint bash, SANS --gpus : rien n'accède au GPU.
# Chaque bloc est tolérant à l'échec (fichier absent dans cette version).

echo
echo "--- 1. Version vLLM de l'image ---"
docker run --rm --entrypoint bash "${IMAGE}" -c \
    "python3 -c 'import vllm; print(vllm.__version__)'" \
    || echo "(version non lisible)"

echo
echo "--- 2. VLLM_WSL2_ENABLE_PIN_MEMORY / VLLM_WEIGHT_OFFLOADING_DISABLE_UVA / VLLM_USE_V2_MODEL_RUNNER dans envs.py ---"
docker run --rm --entrypoint bash "${IMAGE}" -c \
    "[ -f ${VLLM_PKG}/envs.py ] && grep -n -e VLLM_WSL2_ENABLE_PIN_MEMORY -e VLLM_WEIGHT_OFFLOADING_DISABLE_UVA -e VLLM_USE_V2_MODEL_RUNNER ${VLLM_PKG}/envs.py || echo 'fichier envs.py introuvable'"
# grep sort 1 si aucun match -> capturer les cas sans échouer le script.
if ! docker run --rm --entrypoint bash "${IMAGE}" -c \
    "grep -n -e VLLM_WSL2_ENABLE_PIN_MEMORY -e VLLM_WEIGHT_OFFLOADING_DISABLE_UVA -e VLLM_USE_V2_MODEL_RUNNER ${VLLM_PKG}/envs.py" 2>/dev/null \
    | tee /dev/stderr | grep -q .; then
    echo "(aucune des 3 variables dans envs.py, ou fichier absent)"
fi

echo
echo "--- 3. WSL2_ENABLE_PIN_MEMORY dans platforms/cuda.py ---"
docker run --rm --entrypoint bash "${IMAGE}" -c \
    "grep -n WSL2_ENABLE_PIN_MEMORY ${VLLM_PKG}/platforms/cuda.py" \
    || echo "(WSL2_ENABLE_PIN_MEMORY absent de platforms/cuda.py : la réactivation WSL2 n'est pas implémentée dans cette image)"

echo
echo "--- 4. req_states dans diffusion_gemma.py (5 premières lignes) ---"
docker run --rm --entrypoint bash "${IMAGE}" -c \
    "F=\$(ls ${VLLM_PKG}/model_executor/models/diffusion_gemma.py 2>/dev/null || find ${VLLM_PKG} -name 'diffusion_gemma.py' | head -1); \
     [ -n \"\${F:-}\" ] && grep -n req_states \"\$F\" | head -5 || echo 'diffusion_gemma.py introuvable'" \
    || echo "(diffusion_gemma.py non lisible)"

echo
echo "--- 5. Verdict (facts uniquement, pas de conclusion sur le fonctionnement) ---"
# Re-dérivation : chaque variable est CONNUE si elle apparaît dans envs.py,
# INCONNUE sinon. Rien n'est affirmé au-delà.
for VAR in VLLM_WSL2_ENABLE_PIN_MEMORY VLLM_WEIGHT_OFFLOADING_DISABLE_UVA VLLM_USE_V2_MODEL_RUNNER; do
    if docker run --rm --entrypoint bash "${IMAGE}" -c \
        "grep -q -- \"${VAR}\" ${VLLM_PKG}/envs.py" 2>/dev/null; then
        echo "${VAR} : CONNUE de l'image (présente dans envs.py)"
    else
        echo "${VAR} : INCONNUE/ABSENTE de envs.py — la définir dans le compose n'a AUCUN effet garanti dans cette image"
    fi
done
if docker run --rm --entrypoint bash "${IMAGE}" -c \
    "grep -q WSL2_ENABLE_PIN_MEMORY ${VLLM_PKG}/platforms/cuda.py" 2>/dev/null; then
    echo "platforms/cuda.py référence WSL2_ENABLE_PIN_MEMORY (logique de réactivation WSL2 présente)"
else
    echo "platforms/cuda.py ne référence PAS WSL2_ENABLE_PIN_MEMORY : la mémoire épinglée reste désactivée sous WSL2 dans cette image (UVA indisponible -> runner V2 échoue probablement à l'init)"
fi
echo
echo "Prochaine étape (GPU requis) : si VLLM_WSL2_ENABLE_PIN_MEMORY est CONNUE, lancer le compose par défaut (runner V2)."
echo "Sinon : image plus récente ou machine Linux native. Voir vLLM/README.md (arbre de décision)."
