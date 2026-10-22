# AGENTS.md — Instructions pour les agents IA de codage
## ⚠️ Fichiers à lire OBLIGATOIREMENT avant toute action
- [`.cursorrules`](.cursorrules) → règles Cursor IDE (lu automatiquement par l'IDE)
- [`CHECKLIST.md`](CHECKLIST.md) → vérifications obligatoires avant de considérer le projet "prêt"
- [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md) → anomalies connues (blocage mmap/RAM, J4Bis, J5 non conforme)
- [`modules/a2d/docs/A2D_SPEC.md`](modules/a2d/docs/A2D_SPEC.md) → **spécification de la méthode A2D** (équations, Table 6, cibles des Tables 1/16/21)
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) → **architecture** du dépôt (modules, dépendance, pipeline) et [`docs/adr/ADR-001-disposition.md`](docs/adr/ADR-001-disposition.md) → règles de disposition et de dépendance

**Ces fichiers contiennent des informations invisibles par un simple `ls`. Ne jamais push/upload sans les avoir lus.**

## Contexte du projet

**Academiia** est un espace de recherche sur les modèles de diffusion textuelle, centré sur l'application de la méthode **A2D** (AR-to-Diffusion Transfer, arXiv:2610.08108) à **DiffusionGemma** et à **Dream-7B**.

### Structure

```
Academiia/
├── AGENTS.md              ← Tu es ici
├── README.md              # Documentation projet (pipeline)
├── .gitignore             # Fichiers exclus du versioning
├── requirements.txt       # Runtime principal
├── pyproject.toml         # Paquets importables : academiia.core, academiia.a2d
├── academiia/             # CODE GÉNÉRIQUE (paquet `academiia.core`)
│   └── core/              # stats (bootstrap/Wilson), empreintes, E/S safetensors
├── modules/
│   └── a2d/               # MODULE A2D (paquet `academiia.a2d`)
│       ├── a2d/           # BIBLIOTHÈQUE A2D (naming, sources, moteur de fusion)
│       ├── cli/           # CLI : merge_a2d.py, verify_merge.py, measure_geometry.py…
│       ├── eval/          # scripts d'évaluation (verdict G1, bootstrap, empreintes)
│       ├── harness/       # Harnais d'évaluation (convention <famille>/)
│       │   ├── README.md  # la convention
│       │   └── dream/     # harnais Dream-7B (setup.sh, run_evals.sh, tasks/, summarize.py)
│       ├── tests/         # pytest (test_a2d_library.py, test_a2d_mutation.py…)
│       ├── docs/          # SPEC (A2D_SPEC, VP_ADAPTERS), design/, reports/
│       ├── legacy/        # pipeline Jainius, DÉPRÉCIÉ (hors papier)
│       │   ├── scripts/   # compute_tau_a.py, merge_jainius.py, measure_effect.py
│       │   ├── docs/      # AUDIT.md, GUIDE_RESTANT.md (archives)
│       │   └── eval/      # résultats legacy (RESULTS.md, prompts.jsonl)
│       └── STATUS.md      # état du module A2D
├── docs/                  # documentation DU PROJET
│   ├── STATUS.md          # index court des modules (créé en J9)
│   ├── ARCHITECTURE.md    # architecture (J9)
│   └── adr/               # ADR-001..003 + table de migration des chemins
├── tests/                 # pytest du CORE (academiia.core)
├── infra/vllm/            # service vLLM (infra J4Bis)
├── checkpoints/           # Checkpoints binaires (non partages)
│   ├── gemma4_base/       # A_b — Gemma 4 AR base
│   ├── gemma4_it/         # A_p — Gemma 4 AR post-trained
│   ├── diffusiongemma_it/ # D_p — DiffusionGemma IT
│   ├── Jainius/           # build LEGACY (ablation hors papier)
│   ├── qwen25_7b/         # A_b — Qwen2.5-7B        (famille Dream)
│   ├── qwen25_7b_it/      # A_p — Qwen2.5-7B-Instruct
│   ├── dream_7b_base/     # D_b — Dream-7B-Base (indispensable)
│   └── dream_7b_instruct/ # D_p — Dream-7B-Instruct
└── _artefact/             # Dossier temporaire agents (non partage)
```

### Règles de travail

1. **Toujours lire le code dans `modules/a2d/` avant d'agir** — il contient la logique exacte du pipeline ; le code générique est dans `academiia/core/`.
2. **Les checkpoints sont dans `checkpoints/`** — ne jamais les modifier directement ; recréer via les scripts.
3. **`checkpoints/` est à la racine** — ignoré par git, contient les modèles binaires. Ne jamais le modifier directement.
4. **`_artefact/` est un dossier temporaire agents** — tous les fichiers générés qui n'ont pas vocation à être partagés (logs, exports, résultats de tests) doivent y être placés. Ignoré par git, jamais commité.
5. **Lecture des safetensors via `safetensors.safe_open` uniquement** — parseur maison interdit (`.cursorrules`).
6. **GPU/VRAM (inférence vLLM et harnais Dream)** : Docker et le GPU (RTX 5090 32 Go) fonctionnent dans WSL2. Avant TOUTE opération GPU : mesurer `nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv` ; besoin vLLM ≈ `VLLM_GPU_MEMORY_UTILIZATION × total` (0,90 → ≈ 28,8 GiB). Si un modèle LM Studio est chargé ET `memory.free < besoin + 1 GiB` : STOPPER et demander la permission (citer mémoire libre, besoin, modèle détecté). Ne JAMAIS décharger un modèle LM Studio ni tuer de processus. Après chaque série GPU : `docker compose down`, re-mesurer. Détails : `infra/vllm/README.md` (règle VRAM).
7. **Trajectoire par portes** : **G1** mécanique validée sur Qwen2.5-7B → Dream-7B → **G2** reproduction du cas DiffusionGemma médical du papier → **G3** raisonnement (hors papier, conception d'abord). Ne pas commencer une porte avant d'avoir franchi la précédente ; consigner l'état de chaque porte dans `CHECKLIST.md`.
8. **Toute évaluation générative consigne** : modèle, dtype/quantification, décodeur, paramètres, graine, nombre de prompts, et l'intervalle de confiance si l'effectif est réduit.
9. **Aucune écriture hors du dépôt.** Tout ce qui est écrit doit vivre dans l'arborescence du dépôt (`_artefact/`, `harness/<famille>/.cache/`, `eval/`, `checkpoints/`…). Les caches d'outils (`~/.cache/huggingface`, `~/.cache/pip`, `/tmp`, `~/.cache/uv`) doivent être **redirigés** vers le dépôt, pas utilisés en plus. Une écriture déjà présente hors du dépôt est **inventoriée** (`modules/a2d/docs/reports/OUT_OF_WORKSPACE.md`) et **jamais supprimée sans accord écrit de l'utilisateur**.
10. **Harnais d'évaluation : convention `modules/a2d/harness/<famille>/`.** Un harnais (lanceur, tâches, environnement, synthèse) vit dans `modules/a2d/harness/<famille>/` : `setup.sh` (idempotent), `requirements-*.txt`, `tasks/`, `run_evals.sh`, `summarize.py`. Ce qui n'est pas partageable (`.venv/`, `.fork/`, `.cache/`) reste dans le dossier de la famille et est ignoré par git. Les résultats vont dans `modules/a2d/eval/<famille>/<modèle>/<tâche>/`, **sans chemin absolu dans les noms de dossiers**. Voir `modules/a2d/harness/README.md`.
11. **`git push` est réservé à l'utilisateur.** L'agent produit des **commits locaux atomiques** et ne pousse jamais. Le reflog n'identifie pas l'auteur d'un push : ne rien en déduire.

### Pipeline CONFORME (bibliothèque A2D) — à utiliser par défaut

```bash
# 0. checkpoints (famille Dream : 4 modeles, ~60 Go ; shards <= 5 Go)
python modules/a2d/cli/download_models.py --set dream --output-dir ./checkpoints

# 1. Transfert A2D (eq. 1) : D_b + alpha*tau_A   (tau calcule a la volee, jamais persiste)
python modules/a2d/cli/merge_a2d.py --mode transfer \
    --diffusion-base checkpoints/dream_7b_base \
    --tau-a-post checkpoints/qwen25_7b_it --tau-a-base checkpoints/qwen25_7b \
    --alpha 0.9 --mapping identity --output checkpoints/dream_transfer_a09

# 2. Merge A2D (eq. 2) : D_p + lambda*(tau_A - tau_D)
python modules/a2d/cli/merge_a2d.py --mode merge \
    --diffusion checkpoints/dream_7b_instruct \
    --tau-a-post checkpoints/qwen25_7b_it --tau-a-base checkpoints/qwen25_7b \
    --tau-d-post checkpoints/dream_7b_instruct --tau-d-base checkpoints/dream_7b_base \
    --lambda 0.6 --mapping identity --output checkpoints/dream_merge_l06

# 3. Verification post hoc (detecte un coefficient errone, ex. le build a 0.8 etiquete 0.3)
python modules/a2d/cli/verify_merge.py --merged checkpoints/dream_transfer_a09 \
    --reference checkpoints/dream_7b_base --mode transfer \
    --tau-a-post checkpoints/qwen25_7b_it --tau-a-base checkpoints/qwen25_7b --alpha 0.9

# 4. Geometrie des vecteurs (Table 21 du papier)
python modules/a2d/cli/measure_geometry.py \
    --ar-base checkpoints/qwen25_7b --ar-post checkpoints/qwen25_7b_it \
    --diffusion-base checkpoints/dream_7b_base --diffusion-post checkpoints/dream_7b_instruct \
    --output _artefact/geometry_dream.json
```

Garanties de la bibliothèque : τ **jamais** persisté (recalculé à la volée en FP32 depuis les
sources, 3-4 handles `safe_open`) ; arithmétique FP32 + **un seul** cast BF16 ; écriture
atomique ; **vérification post-écriture obligatoire** (`α_eff`/`λ_eff`, échec dur hors
tolérance ; `--no-verify` laisse une trace au manifeste) ; **coefficient nul refusé** (un
transfert à `α=0` ou un merge à `λ=0` n'écrirait qu'une copie de la référence : refus sortie 2
— `--allow-empty-coefficient` pour l'assumer explicitement, avec avertissement au manifeste) ;
manifeste `merge_config.json` (α, λ, hashes d'index des sources, commit git, versions, normes
par famille de tenseurs) ; shards écrits avec le bloc `__metadata__` (`format: pt`) **repris de
la source** — sans lui, le checkpoint fusionné n'est **pas chargeable** par `from_pretrained`
(`transformers.load_state_dict` échoue sur un fichier sans métadonnées).

### Pipeline LEGACY (Jainius) — déprécié, conservé dans `legacy/`

```bash
python modules/a2d/legacy/scripts/compute_tau_a.py --base checkpoints/gemma4_base --posttrained checkpoints/gemma4_it --output-dir _artefact/tau_a
python modules/a2d/legacy/scripts/measure_effect.py --diffusiongemma checkpoints/diffusiongemma_it --tau-dir _artefact/tau_a --alphas 0.05,0.1,0.2,0.3,0.5,0.8
python modules/a2d/legacy/scripts/merge_jainius.py --diffusiongemma checkpoints/diffusiongemma_it --tau-dir _artefact/tau_a --output checkpoints/Jainius --alpha 0.3
python modules/a2d/cli/verify_checkpoint.py --checkpoint checkpoints/Jainius
```

`modules/a2d/legacy/scripts/merge_jainius.py` est un **wrapper déprécié** de la bibliothèque (comportement
observable identique, α borné à [0,1] **côté legacy** seulement) ; `modules/a2d/legacy/scripts/compute_tau_a.py`
est un **chemin legacy** (il persiste τ en `.npy`) : il n'est pas supprimé.

### Contexte scientifique (à jour au 2026-10-20)

- **Source** : Qin et al., "Enhancing Diffusion Language Models with Autoregressive Post-Training Weights", arXiv:2610.08108, 2026 (EPFL). Spécification complète et vérifiée : `modules/a2d/docs/A2D_SPEC.md`.
- **Ce que le papier étudie pour DiffusionGemma** : `D_b` = le checkpoint **IT** (aucun base diffusion public), `τ_A` = delta LoRA **de domaine** (IT → médical), `τ_D` = delta LoRA diffusion. Les adaptateurs sont **publiés** (`gevaertlab/diffusiongemma-radiology-vqa`, LoRA r=64, α=128 ; experts/routeur/vision gelés — vérifié dans `modules/a2d/docs/VP_ADAPTERS.md`).
- **Ce que fait le build `Jainius`** : une **ablation hors papier** (`D_p` + α·τ_A général, α=0.3 non sélectionné sur ancre). Ce n'est ni le transfert du papier (Table 8), ni son ablation « Table 9 » (qui suppose un τ_A médical), ni le A2D-Merge (Table 7/12, qui exige λ et τ_D).
- **Le routeur est fusionné** par défaut (90 des 355 tenseurs du build legacy) : les docs le disaient « inchangé », c'est corrigé.
- **Évaluation J5 (2026-10-19) : non conforme** au protocole du papier (FP8, `temperature=0`, 128 tokens, 103 prompts de texte, sans juge). C'est un contrôle de non-régression, pas une mesure de qualité. Voir `modules/a2d/legacy/eval/RESULTS.md`.
- **Mesures GPU vLLM (2026-10-19)** : seule config qui produit une génération sur cette machine = image vLLM 0.24.0 épinglée par digest + `--quantization fp8` + `--cpu-offload-gb 0` + `--max-model-len 2048` + `--enforce-eager`. Outils : `infra/vllm/smoke_chat.py`, `modules/a2d/eval/eval_compare.py`, `modules/a2d/legacy/eval/prompts.jsonl`.
- **Aucun gain vs `D_p`** (mesuré) : Jainius α=0.3 → 88,1 % de justesse / 21,1 tok/s ; `D_p` → 100 % / 116,4 tok/s. `D_p` reste le témoin.
- **Le chemin legacy `compute_tau_a.py`** (déplacé dans `modules/a2d/legacy/scripts/`) **est bloqué en ENOMEM** sur cette VM (30 GiB) : `safetensors.safe_open` pré-fault la projection du shard AR de 46,5 GiB. **La bibliothèque conforme n'est pas concernée** (τ recalculé à la volée, sans matérialisation). Ne pas contourner par un parseur safetensors maison.

### Outils disponibles

- `huggingface_hub` (1.18.0) — téléchargement de modèles
- `transformers` (5.10.2) — chargement/inférence
- `safetensors.torch` — lecture/écriture de checkpoints
- `peft` (0.21.2) — pour l'équivalence LoRA/merge_and_unload (tests)
- `pytest` — tests
- Entry-points CLI (après `pip install -e .`) : `merge-a2d`, `verify-merge`, `measure-geometry`, `verify-checkpoint`, `download-models` — équivalents aux scripts `modules/a2d/cli/*.py`
- Python 3.12, PyTorch 2.12 (CUDA 13.0)
- Harnais Dream : venv **dans le dépôt**, au chemin définitif `modules/a2d/harness/dream/.venv` (idem `.fork/`, `.cache/` ; ignorés par git — voir `docs/adr/path-migration.md` §4). **Reconstruit sur ce chemin en J10** ; l'ancien dossier racine `harness/` a été supprimé. torch 2.8.0+cu128, transformers 4.46.2, fork lm-eval de Dream — voir `modules/a2d/harness/dream/requirements-dream.txt`
