# Guide — Travaux restants avant publication de Jainius

Ce guide couvre les étapes à réaliser **après** le commit `50b69bd` (pipeline A2D-Dp+ατA complet).

---

## Prérequis

- GPU NVIDIA avec ≥ 32 Go VRAM (pour vLLM BF16 offloadé ; Jainius fait ~52 Go). INT8 n'est pas applicable ici.
- Docker Desktop / Engine avec runtime NVIDIA (`nvidia-smi` sous WSL2)
- `huggingface_hub`, `transformers>=5.10` (≥ 5.11 pour DiffusionGemmaForBlockDiffusion), `safetensors.torch`, `bitsandbytes` (pour smoke INT8)

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

 > **Note** : `--cpu-offload-gb 30` est configuré dans `docker-compose.yml`. Jainius fait ~52 Go et la RAM hôte (~67 Go) absorbe le reste via swap disque.
 > **WSL2** : `VLLM_USE_V2_MODEL_RUNNER=0` force le runner V1 et résout l'erreur UVA (confirmé). Le chargement prend ~5 min. Si un blocage `torch.compile` survient ensuite, ajouter `--enforce-eager` (désactive les CUDA graphs ; débit réduit mais test de cohérence possible).

 ```bash
 cd vLLM
 docker run --rm -it \
     -p 8000:8000 \
     --gpus all \
     --ipc host \
     -v "$(pwd)/../checkpoints/Jainius:/data/Jainius:ro" \
     --entrypoint bash \
     vllm/vllm-openai:gemma -c 'exec vllm serve /data/Jainius \
         --served-model-name "jainius" \
         --host 0.0.0.0 --port 8000 \
         --dtype bfloat16 \
         --cpu-offload-gb 30 \
         --max-model-len 4096 \
         --max-num-seqs 1 \
         --gpu-memory-utilization 0.90 \
         --enforce-eager'
 ```

### 1b. Vérifier que le serveur répond

# Si le conteneur est déjà lancé, tester depuis l'intérieur :
```bash
cd vLLM && docker compose ps | grep vllm
# → doit afficher "Up"
docker exec -i $(docker compose ps -q vllm) python3 -c 'import urllib.request; print(urllib.request.urlopen("http://localhost:8000/v1/models", timeout=30).read().decode())'
# → doit afficher "jainius" dans les modèles disponibles
```

# Ou depuis WSL (si le port est bien publié) :
```bash
curl -s http://localhost:8000/v1/models | python3 -m json.tool
```
### 1b. Vérifier que le serveur répond

```bash
# Dans un autre terminal :
curl -s http://localhost:8000/v1/models | python3 -m json.tool
# → doit afficher "jainius" dans les modèles disponibles
```

### 1c. Test de génération (smoke)

 ```bash
# Test de génération via l'endpoint chat (Jainius est un modèle Instruct) :
curl -s http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "jainius",
        "messages": [{"role": "user", "content": "Combien font 7 × 8 ? Réponds en une phrase."}],
        "max_tokens": 50,
        "temperature": 0.7
    }' | python3 -m json.tool
```
> **Note** : avec `--enforce-eager`, le débit est très réduit (pas de CUDA graphs + offload CPU). La première requête peut prendre plusieurs minutes. Le timeout par défaut du serveur est 900 s.

> **Attendu** : une réponse cohérente (ex: "97"). Si le serveur crash ou retourne des tokens illisibles → problème de checkpoint.

 ## Étape 2 — LICENSE

 Le checkpoint Jainius contient des poids dérivés de Gemma. Vérifier la licence dans les dépôts sources réels (`google/gemma-4` et `google/diffusiongemma`). Si le merge a déjà recopié un fichier `LICENSE`, le conserver ; sinon, ajouter celui du dépôt source approprié.

 ---

 ## Étape 3 — Bench vs D_p (qualité d'inférence)

 Les mesures de poids (`measure_effect.py`) montrent **l'amplitude** du merge mais pas sa **qualité**. Il faut évaluer la génération.

 > **Note** : la perplexité classique ne s'applique pas directement à un modèle de diffusion. Restez sur l'évaluation par génération (latence, débit, format respecté).

---

 ### 3b. Interroger les modèles via vLLM OpenAI-compatible (un par un)

 > **Important** : D_p et Jainius font ~52 Go chacun. Ils ne tiennent pas simultanément sur un GPU de 32 Go ni dans la RAM (~67 Go). Lancer-les l'un après l'autre.

 ```bash
 # Lancer D_p (port 8000) :
 cd vLLM && docker compose up -d --scale vllm=1
 # ... attendre le chargement (~30 min) ...
 python scripts/eval_compare.py \
     --base http://localhost:8000/v1/completions \
     --model "diffusiongemma_it" \
     --prompts eval/prompts.jsonl \
     --output eval/results_Dp.json
 # ... évaluer, puis arrêter D_p ...
 docker compose down
 
 # Lancer Jainius (port 8001) :
 cd vLLM && VLLM_SERVED_MODEL_NAME=jainius docker compose up -d --scale vllm=1
 # ... attendre le chargement (~30 min) ...
 python scripts/eval_compare.py \
     --base http://localhost:8001/v1/completions \
     --model "jainius" \
     --prompts eval/prompts.jsonl \
     --output eval/results_jainius.json
 ```

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


### 3c. Analyser les résultats

Le script doit produire un JSON avec :
- Latence moyenne (ms/token) par modèle
- Débit (tokens/s) par modèle
- Score de format respecté (%)
- Réponses non vides / répétitions dégénérées (%)
- Comparaison α ∈ {0.1, 0.3} (chaque α nécessite un checkpoint ~52 Go ; faire les tests un par un).

---

## Étape 4 — CHECKLIST finale avant push/upload

Exécuter **dans l'ordre** :

```bash
# 1. Intégrité du checkpoint
python scripts/verify_checkpoint.py --checkpoint checkpoints/Jainius
# → doit afficher 11/11 [OK], 1047 tenseurs BF16

# 2. Headers safetensors = "BF16" (pas "U16")
# Vérifié automatiquement par verify_checkpoint.py


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
hf upload <user>/<repo> checkpoints/Jainius/ --repo-type model
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
