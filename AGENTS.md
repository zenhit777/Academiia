# AGENTS.md — Instructions pour les agents IA de codage
## ⚠️ Fichiers à lire OBLIGATOIREMENT avant toute action
- [`.cursorrules`](.cursorrules) → règles Cursor IDE (lu automatiquement par l'IDE)
- [`CHECKLIST.md`](CHECKLIST.md) → vérifications obligatoires avant de considérer le projet "prêt"
- [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md) → anomalies connues du checkpoint Jainius (header corrompu, dtype U16/BF16)

**Ces trois fichiers contiennent des informations invisibles par un simple `ls`. Ne jamais push/upload sans les avoir lus.**

## Contexte du projet

**Academiia** est un espace de recherche sur les modèles de diffusion textuelle, centré sur l'application de la méthode **A2D** (AR-to-Diffusion Transfer) à **DiffusionGemma**.

### Structure

```
Academiia/
├── AGENTS.md              ← Tu es ici
├── README.md              # Documentation projet
├── .gitignore             # Fichiers exclus du versioning
├── docs/                  # Prépublications et références
│   └── 2610.08108v1 - A2D.pdf
├── scripts/               # Scripts reproductibles (Python)
│   ├── download_models.py
│   ├── compute_tau_a.py
│   ├── measure_effect.py    # effet reel post-BF16 + balayage α (CPU)
│   ├── merge_jainius.py
│   ├── verify_checkpoint.py # + --check-values NaN/Inf
│   ├── smoke_int8.py        # test rapide INT8 (iteration, pas preuve)
│   ├── _common.py           # helpers partages (mappings, manifest)
 ├── checkpoints/           # Checkpoints binaires (~247 Gio / ~264 Go sur disque, non partagés)
│   ├── gemma4_base/       (A_b — Gemma 4 AR base)
│   ├── gemma4_it/         (A_p — Gemma 4 IT post-trained)
│   ├── diffusiongemma_it/ (D_p — DiffusionGemma IT)
│   └── Jainius/           (Checkpoint final A2D)
├── _artefact/             # Dossier temporaire agents (non partagé)
```

### Règles de travail

1. **Toujours lire les scripts dans `scripts/` avant d'agir** — ils contiennent la logique exacte du pipeline A2D.
2. **Les checkpoints sont dans `checkpoints/`** — ne jamais les modifier directement, utiliser les scripts pour recréer.
3. **Le checkpoint final est `Jainius`** — c'est le résultat de l'application A2D sur DiffusionGemma.
4. **`checkpoints/` est à la racine** — ignoré par git, contient les modèles binaires (~245 Go). Ne jamais le modifier directement.
5. **`_artefact/` est un dossier temporaire agents** — tous les fichiers générés par les agents qui n'ont pas vocation à être partagés (logs, exports intermédiaires, résultats de tests) doivent y être placés. Ce dossier est ignoré par git et ne doit jamais être commité.

### Pipeline Dp + ατA (ablation inspirée A2D — PAS le A2D-Merge complet)

```bash
 # 1. Télécharger les modèles (~155 Go au total, mesuré : 3 × ~51.6 Go)
python scripts/download_models.py --output-dir ./checkpoints

 # Par défaut : --filter decoder (économise ~35 Go). Écrire les 1013 tenseurs (vision incluse) :
 python scripts/compute_tau_a.py --filter all ...
# 2. Calculer tau_A = A_p - A_b (~10 min, ~3 GB RAM)
python scripts/compute_tau_a.py \
    --base checkpoints/gemma4_base \
    --posttrained checkpoints/gemma4_it \
    --output-dir _artefact/tau_a

# 3. Mesurer l'effet reel apres arrondi BF16 (CPU seul, recommande)
python scripts/measure_effect.py \
    --diffusiongemma checkpoints/diffusiongemma_it \
    --tau-dir _artefact/tau_a \
    --alphas 0.05,0.1,0.2,0.3,0.5,0.8

# 4. Merge: Jainius = D_p + alpha * tau_A (~15 min, ~5 GB RAM)
python scripts/merge_jainius.py \
    --diffusiongemma checkpoints/diffusiongemma_it \
    --tau-dir _artefact/tau_a \
    --output checkpoints/Jainius \
    --alpha 0.3

# 5. Verifier (HuggingFace strict)
python scripts/verify_checkpoint.py --checkpoint checkpoints/Jainius
```
 Le merge sauvegarde déjà en BF16 canonique.

### Contexte Dp + ατA (ne pas appeler "A2D" seul)

- **Source** : Qin et al., "Enhancing Diffusion Language Models with Autoregressive Post-Training Weights", arXiv:2610.08108, 2026 (EPFL)
- **Principe** : τ_A = A_p − A_b ajouté aux couches decoder de DiffusionGemma (mapping par nom). Ablation « Dp + ατA » (Tab. 9), PAS `D_p + λ(τ_A − τ_D)` (τ_D inexistant publiquement).
- **α=0.3** par défaut (balayage 0.05–0.5 via `measure_effect.py`). L'ancien défaut 0.8 dégradait SLAKE (83.1 → 79.3).
- **656 tenseurs** fusionnés max, **391 tenseurs** inchangés (encoder vision). Variante papier `--exclude-norms-embed` (embeddings/norms inchangés).

### Outils disponibles

- `huggingface_hub` (1.18.0) — téléchargement de modèles
- `transformers` (5.10.2) — chargement/inférence
- `safetensors.torch` — lecture/écriture des checkpoints
- Python 3.12, PyTorch
