# modules/dopd — d-OPD (squelette J12, interfaces J13)

Module **d-OPD** (*Future-Aware On-Policy Distillation for Block Diffusion
Language Models*, arXiv:2609.35362).

**État : cibles et contrats formalisés (J12 + J13).** Le paquet contient la
**cible future-aware**, la **configuration** du papier et les **contrats
d'interface** (professeur, rollout, vérificateur, lignée), en NumPy pur (CPU).
Il ne contient **aucun** entraînement, aucun rollout réel, aucun scoring réel
et n'importe pas `academiia.a2d`.

**J18R phase 1 (2026-10-27)** : le harnais de décision et les tests T1–T8 sont livrés et la CI est
verte ; **aucun run GPU, aucune décision GO/NO-GO** — la phase 2 reste bloquée jusqu'à l'acceptation
écrite du pré-enregistrement (§11). `--dry-run` affiche le plan (bras, configurations, shards, estimation
de pic RAM) sans charger torch.

| Élément | Contenu |
|---|---|
| `dopd/` | paquet `academiia.dopd` : `config.py`, `target.py`, `interfaces.py` |
| `harness/` | scripts **hors paquet** (autorisés à utiliser torch) : `smoke_qwen3.py`, `diagnostic_annexe_d.py`, `teacher_scorer_qwen3.py`, `run_j18r_decision.py` (harnais de décision J18R — voie NumPy sans torch, refuse tout run sans `--confirm-accepted`) |
| `docs/DOPD_SPEC.md` | brief de conception (problème, méthode, interfaces, périmètre) |
| `docs/reports/J12.md` | rapport de mission J12 |
| `docs/reports/J13.md` | rapport de mission J13 (interfaces + smoke) |
| `docs/reports/J18R_PRE.md` | pré-enregistrement **figé** du protocole J18R (géométrie, cibles, critères, phases) |
| `docs/reports/J18R.md` | rapport de mission J18R — **phase 1** (harnais + tests), aucune décision |
| `tests/` | tests CPU (import, ADR-001, cible, interfaces) + tests J18R : T1–T7 **purs** (`test_dopd_harness_j18r_decision.py`, aucun `import torch`) et T8 isolé sous torch (`test_dopd_harness_j18r_torch.py`, `importorskip` au niveau fichier) |

La cible applique le contrat numérique du papier (App. B) : deltas, facteurs
`exp(Δs)`, poids et normaliseur **plein vocabulaire** sont calculés en
`float32` ; la correction n'est validée que si ce normaliseur est fini et
strictement positif. Tout débordement, sous-débitement total ou score non fini
retombe donc sur la **cible causale** (fallback), sans lever d'exception.

Les **contrats** de `dopd/interfaces.py` sont des classes abstraites mockables :
`TeacherScorer` (prior causal `r_i(v)` + score de suffixe `s_i(v)`),
`RolloutRecorder` (états de débruitage `X_t` + bloc complété `x̂`),
`TaskVerifier` (porte `R(x̂)`) et `LineageWriter` (manifeste ADR-002,
`producer.module = "dopd"`). Voir `docs/DOPD_SPEC.md` §6.2.

## Vérifier le module

```bash
pip install -e .                 # expose academiia.dopd
python -c "import academiia.core, academiia.a2d, academiia.dopd; print('OK')"
pytest modules/dopd/tests -q
```

## Utilisation minimale (jouet, sans modèle)

```python
import numpy as np
from academiia.dopd import compute_future_aware_target

prior = np.array([0.4, 0.3, 0.2, 0.1])       # r_i(v)
result = compute_future_aware_target(
    prior,
    candidate_indices=[1, 2],                 # union top-k professeur/élève
    candidate_loglikelihoods=[-5.0, -5.0 + np.log(2.0)],
    reference_loglikelihood=-5.0,             # s_i(x_hat_i)
)
print(result.target, result.reason)
```

## Smoke de chargement Qwen3-0.6B (hors paquet)

Le harnais charge le checkpoint, fait un forward et une génération triviaux,
échantillonne le prior causal `r_i(v)` et construit un manifeste ADR-002. Il
**n'entraîne rien** : ni optimizer, ni backward, ni mise à jour de poids.

```bash
# CPU (mode par défaut de la mission)
python modules/dopd/harness/smoke_qwen3.py --model Qwen/Qwen3-0.6B --device cpu

# GPU (optionnel, à déclarer)
python modules/dopd/harness/smoke_qwen3.py --model Qwen/Qwen3-0.6B --device cuda
```

Mesures J13 sur cette machine (rapports dans `_artefact/`, non commités) :
CPU bfloat16 → 2 531 MiB RSS max, chargement 1,05 s ; CUDA bfloat16 →
1 173 MiB GPU, chargement 0,65 s. Détail dans
`docs/reports/J13.md` §5.

## Périmètre

- **Dedans** : cible `p̃_T,i` (eq. 15–18), fallback causal, configuration,
  contrats d'interface (J13).
- **Dehors (J13)** : entraînement, rollout réel, scoring professeur réel,
  benchmarks, reproduction des chiffres du papier, modification des
  résultats/verdicts A2D.

Le dtype d'arithmétique est explicite : `float32` par défaut (App. B du
papier), surchargeable par `arithmetic_dtype="float64"` ou par
`DOPDConfig(arithmetic_dtype=...)`.

Voir [`docs/DOPD_SPEC.md`](docs/DOPD_SPEC.md) pour la spécification complète et
[`../a2d/docs/A2D_SPEC.md`](../a2d/docs/A2D_SPEC.md) pour la méthode A2D, qui est
indépendante.
