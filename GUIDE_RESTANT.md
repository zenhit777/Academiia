# Guide — Travaux restants avant publication de Jainius

Ce guide couvre les étapes à réaliser **après** le commit `50b69bd` (pipeline A2D-Dp+ατA complet).

---

## Prérequis

- GPU NVIDIA avec ≥ 32 Go VRAM (pour vLLM BF16 offloadé ; Jainius fait 48,10 GiB (~52 Go)). INT8 n'est pas applicable ici.
- Docker Desktop / Engine avec runtime NVIDIA (`nvidia-smi` sous WSL2)
- `huggingface_hub`, `transformers>=5.10` (≥ 5.11 pour DiffusionGemmaForBlockDiffusion), `safetensors.torch`, `bitsandbytes` (pour smoke INT8)

---

## Étape 1 — Test d'inférence GPU sur Jainius

### 1a. Lancer vLLM avec le checkpoint Jainius

```bash
cd vLLM
# Vérifier que l'image est disponible :
docker images | grep vllm/vllm-openai:gemma
# → doit afficher 9c719fc0c869 (~29.5 Go)

# Lancer :
docker compose up -d
```

> **Note** : `--cpu-offload-gb 30` est configuré dans `docker-compose.yml`. Jainius fait 48,10 GiB (~52 Go) et la RAM hôte (~67 Go) absorbe le reste en mémoire CPU.

> **Statut WSL2 (test du 2026-10-18)** : avec le runner V1 forcé
> (`VLLM_USE_V2_MODEL_RUNNER=0`), le serveur démarre et charge Jainius, mais la
> **première génération échoue** (assert GPU scatter/gather ; cause probable non
> prouvée : incompatibilité du runner V1 avec la diffusion — voir
> `vLLM/KNOWN_ISSUES.md`). Le compose force donc désormais le **runner V2**
> (défaut de vLLM) avec `VLLM_WSL2_ENABLE_PIN_MEMORY=1` ; suivre l'arbre de
> décision du `vLLM/README.md` (commencer par `./diagnose_image.sh`).

> **Durées observées (2026-10-18, WSL2, /mnt/d en 9P)** : ~14,5 min jusqu'à
> "Application startup complete", dont ~3 min de lecture des poids (9P) et
> ~6 min de profilage/autotune ; 1re requête plus lente (JIT Triton). Copier le
> checkpoint dans le FS natif WSL (ex. `~/models/Jainius`, ajuster
> `VLLM_MODEL_HOST_PATH`) accélère la lecture des poids.

### 1b. Vérifier que le serveur répond

```bash
cd vLLM && docker compose ps | grep vllm
# → doit afficher "Up"
curl -s http://localhost:8000/v1/models | python3 -m json.tool
# → doit afficher "jainius" dans les modèles disponibles
```

### 1c. Test de génération (smoke)

```bash
# Depuis WSL, serveur déjà lancé (Jainius est un modèle Instruct) :
python vLLM/smoke_chat.py
# → POST /v1/chat/completions, max_tokens 64, temperature 0,
#   contrôles objectifs (HTTP 200, texte non vide, pas de répétition dégénérée),
#   sauvegarde eval/smoke_jainius_<horodatage>.json
```

> **Note** : avec `--enforce-eager` (défaut), le débit est très réduit (pas de CUDA graphs + offload CPU). La première requête peut prendre plusieurs minutes. Le timeout par défaut du serveur est 900 s.

> **En cas d'erreur 5xx** : `docker logs --tail 200 jainius-vllm`, et relancer avec
> `CUDA_LAUNCH_BLOCKING=1` dans `.env` pour localiser l'opération fautive (les
> erreurs CUDA sont asynchrones — le point affiché est trompeur).

### 1d. Témoin D_p (indispensable avant toute conclusion sur le merge)

Si la génération échoue sur Jainius, rejouer la MÊME config avec
`D_p` (`checkpoints/diffusiongemma_it`) : un `D_p` qui plante pareil = problème
de configuration, pas du merge. Un seul modèle à la fois (48,10 GiB (~52 Go) chacun) :

```bash
# .env : VLLM_MODEL_HOST_PATH=../checkpoints/diffusiongemma_it
docker compose down && docker compose up -d
python vLLM/smoke_chat.py --model diffusiongemma_it
```

### 1e. Diagnostic `smoke_int8.py` (si échec d'import AutoProcessor)

Si le script affiche `AutoProcessor indisponible`, la cause est une dépendance manquante (très probablement `torchvision`). Diagnostiquez :
```bash
python -c "from transformers import AutoProcessor"
# → si erreur, installer le paquet manquant puis relancer
pip install torchvision
```

## Étape 2 — LICENSE

Le checkpoint Jainius contient des poids dérivés de Gemma. Vérifier la licence dans les dépôts sources réels (`google/gemma-4` et `google/diffusiongemma`). Si le merge a déjà recopié un fichier `LICENSE`, le conserver ; sinon, ajouter celui du dépôt source approprié.

---

## Étape 3 — Évaluation par génération vs D_p

Les mesures de poids (`measure_effect.py`) montrent **l'amplitude** du merge mais pas sa **qualité**. Il faut évaluer la génération.

> **Note** : la perplexité classique ne s'applique pas directement à un modèle de diffusion. Restez sur l'évaluation par génération (latence, débit, format respecté).

### 3a. Préparer les prompts

Créer `eval/prompts.jsonl` avec ~100 prompts couvrant :
- Questions factuelles (géographie, sciences)
- Raisonnement court (maths simples)
- Consignes de format (JSON, liste)
- Prompts en français et anglais

Format JSONL : un objet par ligne :
```jsonl
{"id": "fact_01", "prompt": "Quelle est la capitale du Portugal ?", "expected": "Lisbonne"}
{"id": "math_03", "prompt": "Combien font 7 × 8 ?", "expected": "56"}
```

### 3b. Interroger les modèles via vLLM OpenAI-compatible (un par un)

> **Important** : D_p et Jainius font 48,10 GiB (~52 Go) chacun. Ils ne tiennent pas simultanément sur un GPU de 32 Go ni dans la RAM (~67 Go). Lancer-les l'un après l'autre.

```bash
# Lancer D_p :
#   .env : VLLM_MODEL_HOST_PATH=../checkpoints/diffusiongemma_it
#          VLLM_SERVED_MODEL_NAME=diffusiongemma_it
cd vLLM && docker compose up -d
# ... attendre le chargement (~14,5 min observés via /mnt/d en 9P) ...
python scripts/eval_compare.py \   # À ÉCRIRE (J5) — inexistant à ce jour ; sorties attendues ci-dessous
    --base http://localhost:8000/v1/completions \
    --model "diffusiongemma_it" \
    --prompts eval/prompts.jsonl \
    --output eval/results_Dp.json
# ... évaluer, puis arrêter ...
docker compose down

# Lancer Jainius :
#   .env : VLLM_MODEL_HOST_PATH=../checkpoints/Jainius
#          VLLM_SERVED_MODEL_NAME=jainius
VLLM_SERVED_MODEL_NAME=jainius docker compose up -d
# ... attendre le chargement (~14,5 min observés via /mnt/d en 9P) ...
python scripts/eval_compare.py \   # À ÉCRIRE (J5) — inexistant ; sorties attendues § 3c
    --base http://localhost:8000/v1/completions \
    --model "jainius" \
    --prompts eval/prompts.jsonl \
    --output eval/results_jainius.json
```

### 3c. Analyser les résultats

Le script doit produire un JSON avec :
- Latence moyenne (ms/token) par modèle
- Débit (tokens/s) par modèle
- Score de format respecté (%)
- Réponses non vides / répétitions dégénérées (%)
- Comparaison α ∈ {0.1, 0.3} (chaque α nécessite un checkpoint 48,10 GiB ; faire les tests un par un).

---

## Étape 4 — CHECKLIST finale avant push/upload

Exécuter **dans l'ordre** :

```bash
# 1. Intégrité du checkpoint
python scripts/verify_checkpoint.py --checkpoint checkpoints/Jainius
# → doit afficher 11/11 [OK], 1047 tenseurs BF16

# 2. Headers safetensors = "BF16" (pas "U16")
# Vérifié automatiquement par verify_checkpoint.py

# 3. Scripts compilent
python -c "import ast,glob; [ast.parse(open(f,encoding='utf-8').read()) for f in sorted(glob.glob('scripts/*.py'))]"
# → OK

# 4. Chemins par défaut -> checkpoints/ et _artefact/tau_a (pas /tmp/)
python scripts/merge_jainius.py --help | grep -E "alpha|shape-mismatch"
# → "--alpha 0.3", "--on-shape-mismatch error"

# 5. Manifest merge_config.json écrit
test -f checkpoints/Jainius/merge_config.json && echo OK || echo MANQUANT

# 6. git status propre (sauf _artefact/)
git status --short
# → doit être vide

# 7. Structure HuggingFace complète
ls checkpoints/Jainius/config.json checkpoints/Jainius/README.md
python -c "import json; c=json.load(open('checkpoints/Jainius/config.json')); assert 'DiffusionGemmaForBlockDiffusion' in c['architectures']; print('OK')"

# 8. Conformité à la source (optionnel mais recommandé)
python scripts/verify_checkpoint.py --checkpoint checkpoints/Jainius --reference checkpoints/diffusiongemma_it
```

---

## Étape 5 — Push / Upload HuggingFace

Une fois toutes les étapes ci-dessus validées :

```bash
# Push GitHub
git push origin main

# Upload HuggingFace (nécessite HF_TOKEN)
hf upload <user>/<repo> checkpoints/Jainius/ --repo-type model
```

---

## Résumé visuel

```
┌─────────────────────────────────────────────────────┐
│  Étape 1 : Test inférence GPU (vLLM + smoke)        │
│  → ~14,5 min chargement (observé via /mnt/d en 9P,  │
│    dont ~3 min lecture poids + ~6 min profilage),   │
│    1re requête plus lente (JIT Triton)              │
├─────────────────────────────────────────────────────┤
│  Étape 2 : LICENSE Gemma                            │
│  → 2 min                                            │
├─────────────────────────────────────────────────────┤
│  Étape 3 : Évaluation par génération vs D_p         │
│  → ~1-4 h (dépend du GPU et du nombre de prompts)   │
├─────────────────────────────────────────────────────┤
│  Étape 4 : CHECKLIST finale                         │
│  → ~5 min                                           │
├─────────────────────────────────────────────────────┤
│  Étape 5 : Push GitHub + Upload HF                  │
│  → instantané                                      │
└─────────────────────────────────────────────────────┘
```

---

## Ce qui est déjà fait (ne pas refaire)

- ✅ Pipeline Dp+ατA complet (compute_tau_a, merge_jainius, measure_effect v2)
- ✅ 9 tests pytest passent (merge non vide, shape mismatch, manifest τ, README préservé, --include-norms-embed, verify --reference)
- ✅ Docker compose validé structurellement (`:gemma`, entrypoint bash -c + command liste 1 élément, test `vLLM/scripts/tests/test_compose.py`)
- ✅ `verify_checkpoint.py` : strict, headers BF16, NaN/Inf, --check-values, --reference
- ✅ `merge_jainius.py` : défaut α=0.3, --include-norms-embed (opt-in), suppression index/manifest périmés, contrôle manifest τ
- ✅ `compute_tau_a.py` : défaut --filter decoder (~35 Go économisés)
- ✅ `smoke_int8.py` : docstring clarifié, max_memory_allocated ajouté
- ✅ Tailles harmonisées (~155 Go téléchargement / ~264 Gio sur disque)
- ✅ CI étendue (pytest + test compose dans `.github/workflows/ci.yml`)

---

## Outils vLLM (révisés le 2026-10-19)

- `vLLM/smoke_chat.py` — client HTTP **stdlib uniquement** : POST `/v1/chat/completions`
  sur un serveur vLLM déjà lancé (ne charge pas le modèle). Contrôles objectifs
  (HTTP 200, texte non vide, pas de répétition dégénérée évidente), sauvegarde
  `eval/smoke_<modèle>_<horodatage>.json`. Usage :
  ```bash
  python vLLM/smoke_chat.py --url http://localhost:8000 --model jainius
  ```
- `vLLM/diagnose_image.sh` — diagnostic de l'**image Docker vLLM** sans GPU ni
  chargement de modèle : version, variables WSL2/runner dans `envs.py`,
  `platforms/cuda.py`, `req_states` dans `diffusion_gemma.py`. Usage :
  ```bash
  cd vLLM && ./diagnose_image.sh [image]
  ```
- `vLLM/smoke.sh` — health-check curl rapide (`/v1/models` + génération brute) ;
  `smoke_chat.py` est l'outil de référence pour la génération contrôlée.
