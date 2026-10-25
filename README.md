# Academiia — A2D (AR-to-Diffusion Transfer) sur DiffusionGemma et Dream-7B

Espace de recherche qui applique la méthode **A2D** (Qin et al., EPFL, 2026,
[arXiv:2610.08108](https://arxiv.org/abs/2610.08108)) : transférer un vecteur de tâche
autoregressif vers un modèle de diffusion, puis composer ce transfert avec le vecteur de
post-training du modèle de diffusion. Le dépôt prépare également **d-OPD** : la porte
scientifique Annexe D (J14) est franchie avec réserves de lecture, renforcée J15 et
auditée J16 ; **aucun entraînement** n'a eu lieu.

**Spécification de référence** : [`modules/a2d/docs/A2D_SPEC.md`](modules/a2d/docs/A2D_SPEC.md)
(équations, Table 6, cibles des Tables 1/16/21). Le dépôt prépare également **d-OPD**
([arXiv:2609.35362](https://arxiv.org/abs/2609.35362)), une distillation on-policy
future-aware pour les modèles de diffusion par blocs : la porte scientifique Annexe D
(J14) est franchie avec réserves de lecture, renforcée J15 et auditée J16 ; **aucun
entraînement** n'a eu lieu.
**État consolidé (portes G1/G2/G1b/G3, ce qui est démontré ou non)** :
[`modules/a2d/STATUS.md`](modules/a2d/STATUS.md).
**Détail opérationnel, tableaux de scores et commandes** :
[`modules/a2d/docs/QUICKSTART.md`](modules/a2d/docs/QUICKSTART.md).

## Modules

| Module | Emplacement | Paquet importable | Statut |
|---|---|---|---|
| **core** (générique) | [`academiia/core/`](academiia/core/) | `academiia.core` | en place — bootstrap/Wilson, empreintes, E/S safetensors ; aucune dépendance vers un module |
| **a2d** (méthode) | [`modules/a2d/`](modules/a2d/) | `academiia.a2d` | validé mécaniquement — transfert, merge, ablations, vérification post hoc ; porte **G1 non franchie** |
| **a2d legacy** | [`modules/a2d/legacy/`](modules/a2d/legacy/) | — | déprécié — pipeline Jainius (hors papier), conservé pour l'historique |
| **infra vLLM** | [`infra/vllm/`](infra/vllm/) | — | en place — service d'inférence (image FP8 épinglée) |
| **d-OPD** | [`modules/dopd/`](modules/dopd/) | `academiia.dopd` | **porte Annexe D franchie avec réserves de lecture (J14), renforcée J15, auditée J16** — cible future-aware (NumPy/CPU), configuration, contrats d'interface ; diagnostic multi-prompt sur Qwen3-0.6B versionné (`ΔKL = +0,0802`, IC 95 % `[+0,0638 ; +0,0962]`) ; **aucun entraînement**, aucun `TeacherScorer` mis en œuvre |

Règle de dépendance : un module importe `academiia.core`, **jamais** un autre module
→ [ADR-001](docs/adr/ADR-001-disposition.md).

| Porte | Question | État |
|---|---|---|
| **G1** | l'A2D-Transfer / A2D-Merge améliorent-ils Dream-7B ? | **ouverte** (C2 atteint, C1/C3/C4 non) — [`modules/a2d/STATUS.md`](modules/a2d/STATUS.md) §1 |
| **G1b** | faisabilité Nemotron-Labs-Diffusion-3B | non démarrée |
| **G2** | reproduction du setup médical DiffusionGemma | débloquée côté bibliothèque, **non évaluée** |
| **G3** | raisonnement (hors papier) | non démarrée |

## Installation

```bash
python3 -m pip install -e .            # runtime (numpy, safetensors, torch) + entry-points CLI
python3 -m pip install -e ".[dev,eval,peft]"   # dev + evaluation + PEFT
# Alternative : `pip install -r requirements.txt` (équivalent à `-e .`)
```
> **PEP 668** (`externally-managed-environment`) : si l'installation échoue avec
> « This environment is externally managed », créer un venv dans le dépôt :
> ```bash
> python3 -m venv _artefact/venv-dev --system-site-packages
> source _artefact/venv-dev/bin/activate
> pip install -e .
> ```

L'installation editable fournit les **entry-points** :

| Commande | Rôle |
|---|---|
| `merge-a2d` | fusion A2D : transfert, merge, ablations, mode DiffusionGemma |
| `verify-merge` | **vérification A2D** post hoc (`α_eff` / `λ_eff`, ULP, couverture) |
| `measure-geometry` | géométrie des vecteurs (Table 21 du papier) |
| `verify-checkpoint` | intégrité d'un checkpoint fusionné (format HF strict) — outil **général**, héritage Jainius ; ce n'est pas l'outil de vérification A2D (voir `verify-merge`) |
| `download-models` | téléchargement des checkpoints (`--set dream\|gemma\|all`) |

## Commandes rapides

```bash
# Transfert A2D (eq. 1) : D_b + alpha*tau_A
merge-a2d --mode transfer \
    --diffusion-base checkpoints/dream_7b_base \
    --tau-a-post checkpoints/qwen25_7b_it --tau-a-base checkpoints/qwen25_7b \
    --alpha 0.9 --mapping identity --output checkpoints/dream_transfer_a09

# Merge A2D (eq. 2) : D_p + lambda*(tau_A - tau_D)
merge-a2d --mode merge \
    --diffusion checkpoints/dream_7b_instruct \
    --tau-a-post checkpoints/qwen25_7b_it --tau-a-base checkpoints/qwen25_7b \
    --tau-d-post checkpoints/dream_7b_instruct --tau-d-base checkpoints/dream_7b_base \
    --lambda 0.6 --mapping identity --output checkpoints/dream_merge_l06

# Verification post hoc (detecte un coefficient errone)
verify-merge --merged checkpoints/dream_transfer_a09 \
    --reference checkpoints/dream_7b_base --mode transfer \
    --tau-a-post checkpoints/qwen25_7b_it --tau-a-base checkpoints/qwen25_7b --alpha 0.9

# Geometrie des vecteurs (Table 21)
measure-geometry \
    --ar-base checkpoints/qwen25_7b --ar-post checkpoints/qwen25_7b_it \
    --diffusion-base checkpoints/dream_7b_base --diffusion-post checkpoints/dream_7b_instruct \
    --output _artefact/geometry_dream.json
```

Les scripts restent utilisables directement (`python modules/a2d/cli/merge_a2d.py …`) :
mêmes arguments, mêmes sorties.

Harnais d'évaluation Dream-7B (venv propre au dépôt, hors GPU pour l'installation) :

```bash
bash modules/a2d/harness/dream/setup.sh     # .venv/, .fork/, .cache/ sous le harnais
```

## Documentation

| Document | Contenu |
|---|---|
| [`modules/a2d/STATUS.md`](modules/a2d/STATUS.md) | état des portes, ce qui est démontré ou non, contraintes machine, décisions ouvertes |
| [`modules/a2d/docs/QUICKSTART.md`](modules/a2d/docs/QUICKSTART.md) | détail de la bibliothèque, protocole d'évaluation App. A.1, géométrie, workflow legacy, scripts vLLM |
| [`modules/a2d/docs/A2D_SPEC.md`](modules/a2d/docs/A2D_SPEC.md) | spécification de la méthode (source primaire) |
| [`modules/dopd/docs/DOPD_SPEC.md`](modules/dopd/docs/DOPD_SPEC.md) | brief d-OPD : mismatch, correction future-aware, interfaces, périmètre |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | architecture du dépôt, dépendances, pipeline |
| [`docs/STATUS.md`](docs/STATUS.md) | index court des modules |
| [`docs/CHECKLIST.md`](docs/CHECKLIST.md) / [`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md) | vérifications avant « prêt » / anomalies connues |
| [`modules/a2d/docs/reports/`](modules/a2d/docs/reports/) | rapports de mission A2D (J6 → J11bis) |
| [`modules/dopd/docs/reports/`](modules/dopd/docs/reports/) | rapports du module d-OPD (J12 →) |

## Licence

Apache-2.0 — voir [`LICENSE`](LICENSE). Les poids de modèles ne sont pas inclus dans ce
dépôt ; ils obéissent à leurs propres conditions (Gemma, Dream, DiffusionGemma).
Citation : [`CITATION.cff`](CITATION.cff).
