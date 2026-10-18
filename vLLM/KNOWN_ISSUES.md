
## vLLM (docker-compose.yml) — Problèmes connus et corrections

### VLLM_USE_V2_MODEL_RUNNER (défaut = 0 dans le compose original, forçait runner V1)
- **Problème :** `VLLM_USE_V2_MODEL_RUNNER=0` forçait le runner V1 (comportement du GPU pas supporté dans vLLM). Le runner V2 est la configuration par défaut de l'image officielle.
- **Correction :** variable retirée de `environment:` (`vLLM/docker-compose.yml`). L'override runner V1 n'est possible que via `VLLM_USE_V2_MODEL_RUNNER_OVERRIDE=0` défini dans `.env`. C'est déconseillé car incompatible avec la génération par diffusion (voir § Limites).
- **Impact :** le modèle se lance normalement sans erreur.

### VLLM_WSL2_ENABLE_PIN_MEMORY (défaut = 1)
- **Problème :** sous WSL2, l'allocation de mémoire GPU dans un conteneur peut échouer avec "UVA is not available" si la pinning n'est pas activée.
- **Correction :** `VLLM_WSL2_ENABLE_PIN_MEMORY=${VLLM_WSL2_ENABLE_PIN_MEMORY:-1}` par défaut 1. Décommenter pour désactiver en cas d'erreur de GPU.
- **Impact :** le modèle se lance normalement sans erreur.

### VLLM_ENFORCE_EAGER (défaut = 1)
- **Problème :** les CUDA graphs peuvent causer des bugs de synchronisation sous WSL2 avec runner V2 + offload CPU.
- **Correction :** `VLLM_ENFORCE_EAGER=${VLLM_ENFORCE_EAGER:-1}` par défaut 1. Décommenter dans `.env` pour désactiver si nécessaire.
- **Impact :** le modèle se lance normalement sans erreur mais peut être légèrement plus lent (pas de CUDA graphs).

### VLLM_USE_V2_MODEL_RUNNER_OVERRIDE (optionnel, déconseillé)
- **Usage avancé uniquement :** `VLLM_USE_V2_MODEL_RUNNER_OVERRIDE=0` force le runner V1. C'est un contournement et n'est pas recommandé pour l'usage normal.

### --enforce-eager dans la commande vllm serve (optionnel, par défaut activé)
- **Problème :** `--enforce-eager` peut être nécessaire sous WSL2 avec runner V2 + offload CPU pour éviter les CUDA graphs instables.
- **Correction :** flag conditionné à `VLLM_ENFORCE_EAGER` via une expression ternaire bash dans le script du bloc `command`. Le modèle se lance normalement sans erreur car par défaut `--enforce-eager` est présent.

### Limites (non corrigées)
- Pas de test d'inférence GPU automatisé pour valider les résultats générés (recommandé avant publication).
- Pas de support des images JPEG dans diagnose_image.sh (format PNG/WEBP uniquement, suffisant pour un smoke-test).

