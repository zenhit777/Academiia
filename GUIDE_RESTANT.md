# Guide — Travaux restants avant publication de Jainius

> ## ⚠️ DOCUMENT OBSOLÈTE (rectificatif J6bis, MIN-06)
>
> Ce guide décrit le **pipeline legacy** (`D_p + α·τ_A` appliqué à DiffusionGemma-IT, build
> `checkpoints/Jainius`) et la procédure de publication d'un checkpoint **hors papier**.
> Il est **conservé pour l'historique** et n'est plus la référence :
>
> * le pipeline **conforme** est `scripts/merge_a2d.py` + `scripts/a2d/` (transfert
>   `D_b + α·τ_A`, merge `D_p + λ·(τ_A − τ_D)`) ; le build `Jainius` est une **ablation
>   hors papier** (τ_A *général*, α non sélectionné sur ancre, aucun gain mesuré) ;
> * la trajectoire de recherche est **par portes** (G1 → G2 → G3) : voir `CHECKLIST.md` ;
> * l'étape 5 (« `hf upload … Jainius` ») est **sans objet** : la publication d'un checkpoint
>   n'est plus prévue, et `git push` est réservé à l'utilisateur ;
> * la checklist de publication utile est `CHECKLIST.md` ;
> * l'état réel des portes est `docs/reports/J6bis.md` (vérifié) et
>   `docs/reports/AUDIT_J6.md` + `docs/reports/REVIEW_N2_J6.md` (revues).
>
> **Ce guide ne doit pas être suivi tel quel.** Le texte ci-dessous est inchangé hormis
> cette bannière.

Ce guide couvre les étapes à réaliser **après** le commit `50b69bd` (pipeline legacy
`D_p + α·τ_A`, **hors papier**).
**État au 2026-10-19** : Étapes 1, 2 et 3 faites (voir § « Ce qui est déjà fait »). Restent
l'étape 4 (checklist finale), l'étape 5 (push/upload) — et une décision éditoriale :
le merge n'améliore pas le témoin `D_p` (mesuré, cf. `eval/RESULTS.md`).

---

## Prérequis

- GPU NVIDIA avec ≥ 32 Go VRAM (RTX 5090 32 Go sur cette machine) ; **la seule config qui
  produit une génération** est la quantification FP8 à la volée (le modèle en BF16 fait
  48,1 GiB, il ne tient pas dans 31,84 GiB de VRAM).
- Docker Engine avec runtime NVIDIA (`nvidia-smi` sous WSL2), image vLLM **épinglée par
  digest** (0.24.0) — le tag `gemma` (0.22.1rc1) est inutilisable ici (UVA / runner V1).
- `huggingface_hub`, `transformers>=5.10` (≥ 5.11 pour DiffusionGemmaForBlockDiffusion),
  `safetensors.torch`, `bitsandbytes` (pour smoke INT8).
- **≥ 50 Go de RAM** pour `scripts/compute_tau_a.py` : `safetensors.safe_open` pré-fault
  la projection du shard AR de 46,5 GiB (ENOMEM avec les 30 GiB de la VM WSL2).
  Remède : `%UserProfile%\.wslconfig` → `[wsl2] memory=64GB` puis `wsl --shutdown`.

---

## Étape 1 — Test d'inférence GPU sur Jainius ✅ FAIT (2026-10-19)

### 1a. Lancer vLLM avec le checkpoint Jainius

```bash
cd vLLM
cp .env.template .env     # puis vérifier VLLM_MODEL_HOST_PATH / VLLM_EXTRA_ARGS
docker compose up -d
```

> **Config mesurée** (la seule qui fonctionne sur cette machine) :
> image `vllm/vllm-openai@sha256:251eba5cc7c12fed0b75da22a9240e582b1c9e39f6fbc064f86781b963bd814f`
> (vLLM 0.24.0), runner V2 (défaut, **ne pas** définir `VLLM_USE_V2_MODEL_RUNNER`),
> `--quantization fp8` (quantification à la volée des poids BF16 → 25,85 GiB en VRAM),
> `--cpu-offload-gb 0`, `--max-model-len 2048`, `--max-num-seqs 1`,
> `--gpu-memory-utilization 0.92`, `--enforce-eager`, `PYTORCH_CUDA_ALLOC_CONF=` (vide).
> Chargement ≈ 6 min, RAM hôte stable à 4-6 GiB. Logs : `vLLM/KNOWN_ISSUES.md` (E0→E6).

> ### Les trois chiffres à ne pas confondre
> (mesurés le 2026-10-19 — WSL2 / distribution **Ubuntu** 24.04, RTX 5090)
>
> | Grandeur | Valeur mesurée | Source |
> |---|---|---|
> | **Poids du checkpoint (BF16)** | **48,14 GiB** (51 680 005 469 B ≈ 51,68 Go) | `du -sb checkpoints/Jainius` |
> | **Budget VRAM alloué par vLLM** | **29,29 GiB** (0,92 × 32 607 MiB) | `--gpu-memory-utilization` |
> | **VRAM physique totale** | **31,84 GiB** (32 607 MiB) | `nvidia-smi --query-gpu=memory.total` |
> | RAM utile dans WSL2 | 29 GiB disponibles (30 GiB au total) | `free -h` |
>
> En FP8, les poids tiennent dans 25,85 GiB (budget 29,29 GiB) : le cache KV garde
> ~2,6 GiB. En BF16 il faudrait un offload CPU de ~20 Go, ce qui sature la RAM WSL2 —
> et le runner V2 compte le modèle entier contre le budget VRAM (offload inutile).

### 1b. Vérifier que le serveur répond

```bash
cd vLLM && docker compose ps | grep vllm
curl -s http://localhost:8000/v1/models | python3 -m json.tool
# → doit afficher "jainius" dans les modèles disponibles
```

### 1c. Test de génération (smoke)

```bash
# Serveur déjà lancé (Jainius est un modèle Instruct) :
python vLLM/smoke_chat.py --model jainius
# → POST /v1/chat/completions, max_tokens 64, temperature 0,
#   contrôles objectifs (HTTP 200, texte non vide, pas de répétition dégénérée),
#   sauvegarde eval/smoke_jainius_<horodatage>.json
```

### 1d. Témoin D_p (fait : indispensable avant toute conclusion sur le merge)

Rejouer la MÊME config avec `D_p` (`checkpoints/diffusiongemma_it`) : un `D_p` qui
produit le même texte dégradé = problème de configuration ; un `D_p` sain = problème
de checkpoint. Un seul modèle à la fois (48,1 GiB chacun) :

```bash
# .env : VLLM_MODEL_HOST_PATH=../checkpoints/diffusiongemma_it
#        VLLM_SERVED_MODEL_NAME=diffusiongemma_it
docker compose down && docker compose up -d
python vLLM/smoke_chat.py --model diffusiongemma_it
```

> **Résultat 2026-10-19** : `D_p` sain (3/3), Jainius α=0.3 sain (4/4 puis 103/103).
> Le « texte dégénéré » observé la veille venait du **checkpoint livré, fusionné à
> α=0.8** (déprécié) et non de la configuration de service.

### 1e. Diagnostic `smoke_int8.py` (si échec d'import AutoProcessor)

```bash
python -c "from transformers import AutoProcessor"
pip install torchvision
```

---

## Étape 2 — LICENSE ✅ FAIT (2026-10-19)

Les repos sources (`google/gemma-4-26B-A4B`, `google/gemma-4-26B-A4B-it`,
`google/diffusiongemma-26B-A4B-it`) **ne publient aucun fichier LICENSE** (22 / 10 / 12
fichiers listés, aucun nom de licence) : il n'y a donc rien à recopier. La model card de
`checkpoints/Jainius/README.md` déclare `license: apache-2.0` + `license_link` vers
https://ai.google.dev/gemma/docs/gemma_4_license, comme la card de `D_p`.

---

## Étape 3 — Évaluation par génération vs D_p ✅ FAIT (2026-10-19)

### 3a. Prompts

`eval/prompts.jsonl` : 103 prompts (faits FR/EN, maths, formats de sortie, questions
libres), un objet JSON par ligne :

```jsonl
{"id": "fact_fr_01", "category": "fact", "lang": "fr", "prompt": "Quelle est la capitale de la France ?", "expected": "Paris"}
```

### 3b. Interroger les modèles via vLLM OpenAI-compatible (un par un)

```bash
# .env : VLLM_MODEL_HOST_PATH=../checkpoints/Jainius
#        VLLM_SERVED_MODEL_NAME=jainius
cd vLLM && docker compose up -d
# ... attendre le chargement (~6 min) ...
python scripts/eval_compare.py \
    --base http://localhost:8000/v1/chat/completions \
    --model jainius \
    --prompts eval/prompts.jsonl \
    --output eval/results_jainius_a03.json
docker compose down

# Puis D_p (même config, autre checkpoint) :
#   .env : VLLM_MODEL_HOST_PATH=../checkpoints/diffusiongemma_it
#          VLLM_SERVED_MODEL_NAME=diffusiongemma_it
python scripts/eval_compare.py --base http://localhost:8000/v1/chat/completions \
    --model diffusiongemma_it --prompts eval/prompts.jsonl --output eval/results_Dp.json
```

### 3c. Analyser les résultats

`scripts/eval_compare.py` produit : latence moyenne/médiane/p95, ms par token, débit,
% de réponses non vides, % de répétitions dégénérées, % de présence de la valeur attendue,
% de format respecté, plus un détail par catégorie et par prompt.

> **Résultat 2026-10-19** (`eval/RESULTS.md`) : Jainius α=0.3 → 88,1 % de justesse,
> 75,0 % de format, 21,1 tok/s, 0 % dégénéré. `D_p` → 100 %, 93,8 %, 116,4 tok/s, 0 %
> dégénéré. **Le merge n'améliore pas le témoin ; il le dégrade** (corruptions de tokens :
> « Can로 » pour Canberra, « New area » pour New Delhi). Comparaison α ∈ {0.1, 0.3} non
> réalisée (chaque α exige un checkpoint de 48,1 GiB, et τ_A n'est pas recalculable sur
> cette machine — cf. Prérequis).

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
# → OK (α=0.3, include_norms_embed=false, 355/692)

# 6. git status propre (sauf _artefact/ et checkpoints/)
git status --short

# 7. Structure HuggingFace complète
ls checkpoints/Jainius/config.json checkpoints/Jainius/README.md
python -c "import json; c=json.load(open('checkpoints/Jainius/config.json')); assert 'DiffusionGemmaForBlockDiffusion' in c['architectures']; print('OK')"

# 8. Conformité à la source
python scripts/verify_checkpoint.py --checkpoint checkpoints/Jainius --reference checkpoints/diffusiongemma_it
# → 1047 tenseurs, 0 écart de shape/dtype  (déjà fait le 2026-10-19)
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

> ⚠️ **Avant publication, trancher** : le checkpoint livré est fonctionnel mais **moins
> bon que `D_p`** sur les 103 prompts mesurés, et 5× plus lent. Publier `Jainius` tel
> quel doit être assumé explicitement (« ablation A2D : aucun gain, dégradation
> mesurée »), ou ne pas le publier.

---

## Résumé visuel

```
┌─────────────────────────────────────────────────────┐
│  Étape 1 : Test inférence GPU (vLLM + smoke)   ✅    │
│  → config FP8 à la volée, ~6 min de chargement,     │
│    témoin D_p sain, Jainius α=0.3 sain              │
├─────────────────────────────────────────────────────┤
│  Étape 2 : LICENSE Gemma                      ✅    │
│  → pas de fichier LICENSE dans les repos sources ;  │
│    card : apache-2.0 + license_link Gemma          │
├─────────────────────────────────────────────────────┤
│  Étape 3 : Évaluation par génération vs D_p   ✅    │
│  → 103 prompts, 0 % dégénéré ; Jainius 88,1 %      │
│    vs D_p 100 % de justesse ; rapport eval/RESULTS  │
├─────────────────────────────────────────────────────┤
│  Étape 4 : CHECKLIST finale                   ⬜    │
│  → ~5 min                                           │
├─────────────────────────────────────────────────────┤
│  Étape 5 : Push GitHub + Upload HF             ⬜    │
│  → instantané                                      │
└─────────────────────────────────────────────────────┘
```

---

## Ce qui est déjà fait (ne pas refaire)

- ✅ Pipeline Dp+ατA complet (compute_tau_a, merge_jainius, measure_effect v2)
- ✅ Checkpoint livré **refusionné à α=0.3** (355 fusionnés / 692 inchangés), vérifié :
  `verify_checkpoint.py` 11/11 `[OK]`, 1047 tenseurs BF16, conforme à la référence
- ✅ 9 tests pytest passent (merge non vide, shape mismatch, manifest τ, README préservé, --include-norms-embed, verify --reference)
- ✅ Docker compose validé structurellement (`:gemma`, entrypoint bash -c + command liste 1 élément, test `vLLM/scripts/tests/test_compose.py`)
- ✅ `verify_checkpoint.py` : strict, headers BF16, NaN/Inf, --check-values, --reference
- ✅ `merge_jainius.py` : défaut α=0.3, --include-norms-embed (opt-in), suppression index/manifest périmés, contrôle manifest τ
- ✅ `compute_tau_a.py` : défaut --filter decoder (~35 Go économisés)
- ✅ `smoke_int8.py` : docstring clarifié, max_memory_allocated ajouté
- ✅ CI étendue (pytest + test compose dans `.github/workflows/ci.yml`)
- ✅ **Inférence GPU validée** (2026-10-19) : config FP8 épinglée par digest, runner V2,
  image `gemma` écartée (UVA) ; cause racine du texte dégénéré identifiée (α=0.8 périmé)
- ✅ **Évaluation par génération** : `scripts/eval_compare.py` + `eval/prompts.jsonl`
  (103 prompts) + `eval/results_*.json` + `eval/RESULTS.md`

---

## Outils vLLM (révisés le 2026-10-19)

- `vLLM/smoke_chat.py` — client HTTP **stdlib uniquement** : POST `/v1/chat/completions`
  sur un serveur vLLM déjà lancé (ne charge pas le modèle). Contrôles objectifs
  (HTTP 200, texte non vide, pas de répétition dégénérée évidente), sauvegarde
  `eval/smoke_<modèle>_<horodatage>.json`. Usage :
  ```bash
  python vLLM/smoke_chat.py --url http://localhost:8000 --model jainius
  ```
- `scripts/eval_compare.py` — évaluation par génération (latence, débit, justesse,
  format, dégénérescence) sur un jeu de prompts JSONL. Sortie : rapport JSON + résumé
  stdout. Usage :
  ```bash
  python scripts/eval_compare.py --base http://localhost:8000/v1/chat/completions \
      --model jainius --prompts eval/prompts.jsonl --output eval/results_jainius_a03.json
  ```
- `vLLM/diagnose_image.sh` — diagnostic de l'**image Docker vLLM** sans GPU ni
  chargement de modèle : version, variables WSL2/runner dans `envs.py`,
  `platforms/cuda.py`, `req_states` dans `diffusion_gemma.py`. Usage :
  ```bash
  cd vLLM && ./diagnose_image.sh [image]
  ```
- `vLLM/smoke.sh` — health-check curl rapide (`/v1/models` + génération brute) ;
  `smoke_chat.py` est l'outil de référence pour la génération contrôlée.
