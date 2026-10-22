#!/usr/bin/env bash
# Smoke test du serveur vLLM Jainius (API OpenAI-compatible).
# Usage (depuis WSL) : ./smoke.sh [PORT]   — defaut 8000 (VLLM_PORT du .env)
# Code retour 0 si une reponse non vide est recue.
set -euo pipefail

PORT="${1:-${VLLM_PORT:-8000}}"
MODEL="${VLLM_SERVED_MODEL_NAME:-jainius}"

echo "-> health (/v1/models) sur :$PORT ..."
curl -sf "http://localhost:${PORT}/v1/models" | head -c 300
echo
echo "-> generation courte ..."
RESP="$(curl -sf "http://localhost:${PORT}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{
    \"model\": \"${MODEL}\",
    \"messages\": [{\"role\": \"user\", \"content\": \"En une phrase, qu'est-ce que la photosynthese ?\"}],
    \"max_tokens\": 128
  }")"
echo "$RESP" | head -c 1500
echo
echo "$RESP" | grep -q '"content": *"[[:space:]]*[^"[:space:]]' \
  && echo "SMOKE OK : reponse non vide." \
  || { echo "SMOKE FAIL : reponse vide."; exit 1; }
