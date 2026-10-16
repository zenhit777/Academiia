# AGENTS.md — Instructions pour les agents IA de codage
## ⚠️ Fichiers critiques (lus automatiquement par les agents)
- `CLAUDE.md` → Anomalies connues + instructions critiques (Claude Code / Anthropic)
- `.cursorrules` → Règles Cursor IDE
- `CHECKLIST.md` → Vérifications obligatoires avant "prêt"
- `KNOWN_ISSUES.md` → Anomalies détaillées du checkpoint Jainius

**Toujours lire CLAUDE.md et KNOWN_ISSUES.md en premier.** Ils contiennent les anomalies qui ne sont pas visibles par un simple `ls`.

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
│   ├── merge_jainius.py
│   └── fix_dtype.py
└── _artefact/             # Checkpoints binaires (~210 Go, non partagés)
    └── checkpoints/
        ├── gemma4_base/       (A_b — Gemma 4 AR base)
        ├── gemma4_it/         (A_p — Gemma 4 IT post-trained)
        ├── diffusiongemma_it/ (D_p — DiffusionGemma IT)
        └── Jainius/           (Checkpoint final A2D)
```

### Règles de travail

1. **Toujours lire les scripts dans `scripts/` avant d'agir** — ils contiennent la logique exacte du pipeline A2D.
2. **Les checkpoints sont dans `_artefact/checkpoints/`** — ne jamais les modifier directement, utiliser les scripts pour recréer.
3. **Le checkpoint final est `Jainius`** — c'est le résultat de l'application A2D sur DiffusionGemma.
4. **Ne pas commiter `_artefact/`** — il est dans `.gitignore`. Pour partager, uploader sur HuggingFace via `huggingface-cli upload`.

### Pipeline A2D (référence rapide)

```bash
# 1. Télécharger les modèles (~165 Go)
python scripts/download_models.py --output-dir ./_artefact/checkpoints

# 2. Calculer tau_A = A_p - A_b (~10 min, ~3 GB RAM)
python scripts/compute_tau_a.py \
    --base _artefact/checkpoints/gemma4_base \
    --posttrained _artefact/checkpoints/gemma4_it \
    --output-dir /tmp/a2d_jainius

# 3. Merge A2D: Jainius = D_p + 0.8 * tau_A (~15 min, ~5 GB RAM)
python scripts/merge_jainius.py \
    --diffusiongemma _artefact/checkpoints/diffusiongemma_it \
    --tau-dir /tmp/a2d_jainius \
    --output _artefact/checkpoints/Jainius

# 4. Convertir en BF16 (format HuggingFace)
python scripts/fix_dtype.py --checkpoint _artefact/checkpoints/Jainius
```

### Contexte A2D

- **Source** : Qin et al., "Enhancing Diffusion Language Models with Autoregressive Post-Training Weights", arXiv:2610.08108, 2026 (EPFL)
- **Principe** : τ_A = A_p − A_b est ajouté aux couches decoder de DiffusionGemma (mapping par nom de tenseur).
- **α=0.8** : poids d'interpolation utilisé dans le merge.
- **656 tenseurs** fusionnés, **391 tenseurs** inchangés (encoder vision).

### Outils disponibles

- `huggingface_hub` (1.18.0) — téléchargement de modèles
- `transformers` (5.10.2) — chargement/inférence
- `safetensors.torch` — lecture/écriture des checkpoints
- Python 3.12, PyTorch
