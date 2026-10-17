# Guide — Travaux restants avant publication de Jainius

Ce guide couvre les étapes à réaliser **après** le commit `50b69bd` (pipeline A2D-Dp+ατA complet).

---

## Prérequis

- GPU NVIDIA avec ≥ 32 Go VRAM (pour vLLM BF16 offloadé) ou ≥ 8 Go pour INT8
- Docker Desktop / Engine avec runtime NVIDIA (`nvidia-smi` sous WSL2)
- `huggingface_hub`, `transformers==5.10.2`, `safetensors.torch`, `bitsandbytes` (pour smoke INT8)

---

## Étape 1 — Test d'inférence GPU sur Jainius

### 1a. Lancer vLLM avec le checkpoint Jainius

 # Méthode recommandée : via docker-compose (déjà configuré dans `vLLM/docker-compose.yml`)
 ```bash
 cd vLLM
 # Vérifier que l'image est disponible :
 docker images | grep vllm/vllm-openai:gemma
 # → doit afficher 9c719fc0c869 (~29.5 Go)
 
 # Lancer (démarrage ~30 min pour charger 52 Go BF16) :
 docker compose up -d
 ```

 > **Note** : le `--cpu-offload-gb 303` est configuré dans `docker-compose.yml`. Jainius fait ~52 Go et la RAM hôte (~67 Go) absorbe le reste via swap disque.

 # Méthode alternative (sans docker-compose, pour debug) :
 ```bash
 cd vLLM
 docker run --rm -it \
     --gpus all \
     --ipc host \
     -v "$(pwd)/../checkpoints/Jainius:/data/Jainius:ro" \
     vllm/vllm-openai:gemma bash -c 'exec vllm serve /data/Jainius \
         --served-model-name "jainius" \
         --host 0.0.0.0 --port 8000 \
         --dtype bfloat16 \
         --cpu-offload-gb 303 \
         --max-model-len 4096 \
         --max-num-seqs 1 \
         --gpu-memory-utilization 0.90'
 ```

### 1b. Vérifier que le serveur répond

```bash
# Dans un autre terminal :
curl -s http://localhost:8000/v1/models | python3 -m json.tool
# → doit afficher "jainius" dans les modèles disponibles
```

### 1c. Test de génération (smoke)

```bash
curl -s http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "jainius",
        "prompt": "Quel est le plus grand nombre premier inférieur à 100 ?",
        "max_tokens": 50,
        "temperature": 0.7
    }' | python3 -m json.tool
```

> **Attendu** : une réponse cohérente (ex: "97"). Si le serveur crash ou retourne des tokens illisibles → problème de checkpoint.

### 1d. Test INT8 rapide (optionnel, ~5 min)

```bash
python scripts/smoke_int8.py --checkpoint checkpoints/Jainius --no-generate
# → doit afficher "SMOKE OK (forward seul)" + max_memory_allocated en Go
```

---

## Étape 2 — Ajouter LICENSE

Le checkpoint Jainius contient des poids dérivés de Gemma. La licence **Gemma Community License** s'applique :

1. Télécharger la licence officielle depuis [gemma.ai](https://gemma.google.com/legal/) ou le repo HuggingFace `google/gemma-3-4b-it`.
2. Copier le fichier `LICENSE` dans `checkpoints/Jainius/`.
3. Vérifier que `verify_checkpoint.py --reference checkpoints/diffusiongemma_it` passe (le LICENSE est un sidecar non-poids).

---

## Étape 3 — Perplexité / Bench vs D_p (qualité d'inférence)

Les mesures de poids (`measure_effect.py`) montrent **l'amplitude** du merge mais pas sa **qualité**. Il faut évaluer la génération.

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

### 3b. Interroger les deux modèles via vLLM OpenAI-compatible

```bash
# Lancer D_p et Jainius sur des ports différents :
python scripts/eval_compare.py \
    --base http://localhost:8000/v1/completions \
    --model "diffusiongemma_it" \
    --target http://localhost:8001/v1/completions \
    --model "jainius" \
    --prompts eval/prompts.jsonl \
    --output eval/results_α0.3.json
```

> **Paramètres identiques** pour les deux modèles : `temperature=0.7`, `max_tokens=256`, `top_p=0.9`.

### 3c. Analyser les résultats

Le script doit produire un JSON avec :
- Latence moyenne (ms/token) par modèle
- Débit (tokens/s) par modèle
- Score de format respecté (%)
- Réponses non vides / répétitions dégénérées (%)
- Comparaison α ∈ {0.05, 0.1, 0.2, 0.3}

---

## Étape 4 — CHECKLIST finale avant push/upload

Exécuter **dans l'ordre** :

```bash
# 1. Intégrité du checkpoint
python scripts/verify_checkpoint.py --checkpoint checkpoints/Jainius
# → doit afficher 11/11 [OK], 1047 tenseurs BF16

# 2. Headers safetensors = "BF16" (pas "U16")
# Vérifié automatiquement par verify_checkpoint.py

# 3. Aucun JSONDecodeError au parsing
python -c "import glob, json; [json.load(open(f)) for f in sorted(glob.glob('checkpoints/Jainius/*.safetensors'))]"
# → doit passer sans erreur (les headers sont lus via safe_open, pas json.load)

# 4. Scripts compilent
python -c "import ast,glob; [ast.parse(open(f,encoding='utf-8').read()) for f in sorted(glob.glob('scripts/*.py'))]"
# → OK

# 5. Chemins par défaut -> checkpoints/ et _artefact/tau_a (pas /tmp/)
python scripts/merge_jainius.py --help | grep -E "alpha|shape-mismatch"
# → "--alpha 0.3", "--on-shape-mismatch error"

# 6. Manifest merge_config.json écrit
test -f checkpoints/Jainius/merge_config.json && echo OK || echo MANQUANT

# 7. git status propre (sauf _artefact/)
git status --short
# → doit être vide

# 8. Structure HuggingFace complète
ls checkpoints/Jainius/config.json checkpoints/Jainius/README.md
python -c "import json; c=json.load(open('checkpoints/Jainius/config.json')); assert 'DiffusionGemmaForBlockDiffusion' in c['architectures']; print('OK')"

# 9. Conformité à la source (optionnel mais recommandé)
python scripts/verify_checkpoint.py --checkpoint checkpoints/Jainius --reference checkpoints/diffusiongemma_it
```

---

## Étape 5 — Push / Upload HuggingFace

Une fois toutes les étapes ci-dessus validées :

```bash
# Push GitHub
git push origin main

# Upload HuggingFace (nécessite HF_TOKEN)
huggingface-cli upload <user>/<repo> checkpoints/Jainius/ --repo-type model
```

---

## Résumé visuel

```
┌─────────────────────────────────────────────────────┐
│  Étape 1 : Test inférence GPU (vLLM + smoke)       │
│  → ~30 min chargement, ~5 min génération            │
├─────────────────────────────────────────────────────┤
│  Étape 2 : LICENSE Gemma                            │
│  → 2 min                                            │
├─────────────────────────────────────────────────────┤
│  Étape 3 : Perplexité / Bench vs D_p                │
│  → ~1-4 h (dépend du GPU et du nombre de prompts)  │
├─────────────────────────────────────────────────────┤
│  Étape 4 : CHECKLIST finale                         │
│  → ~5 min                                           │
├─────────────────────────────────────────────────────┤
│  Étape 5 : Push GitHub + Upload HF                  │
│  → instantané                                       │
└─────────────────────────────────────────────────────┘
```

---

## Ce qui est déjà fait (ne pas refaire)

- ✅ Pipeline Dp+ατA complet (compute_tau_a, merge_jainius, measure_effect v2)
- ✅ 9 tests pytest passent (merge non vide, shape mismatch, manifest τ, README préservé, --include-norms-embed, verify --reference)
- ✅ Docker compose validé (`:gemma`, entrypoint bash -c + command liste 1 élément)
- ✅ `verify_checkpoint.py` : strict, headers BF16, NaN/Inf, --check-values, --reference
- ✅ `merge_jainius.py` : défaut α=0.3, --include-norms-embed (opt-in), suppression index/manifest périmés, contrôle manifest τ
- ✅ `compute_tau_a.py` : défaut --filter decoder (~35 Go économisés)
- ✅ `smoke_int8.py` : docstring clarifié, max_memory_allocated ajouté
- ✅ fix_dtype.py supprimé + toutes références retirées
- ✅ Tailles harmonisées (~155 Go téléchargement / ~264 Gio sur disque)
- ✅ CI étendue (pytest dans `.github/workflows/ci.yml`)
