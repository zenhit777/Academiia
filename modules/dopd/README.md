# modules/dopd — d-OPD (squelette J12)

Module **d-OPD** (*Future-Aware On-Policy Distillation for Block Diffusion
Language Models*, arXiv:2609.35362).

**État : squelette J12.** Le module contient la **cible future-aware** et la
**configuration** du papier, en NumPy pur (CPU). Il ne contient **aucun**
entraînement, aucun rollout, aucun téléchargement de checkpoint et n'importe
pas `academiia.a2d`.

| Élément | Contenu |
|---|---|
| `dopd/` | paquet `academiia.dopd` : `config.py`, `target.py` |
| `docs/DOPD_SPEC.md` | brief de conception (problème, méthode, interfaces, périmètre) |
| `docs/reports/J12.md` | rapport de mission J12 |
| `tests/` | tests CPU (import, ADR-001, cible) |

La cible applique le contrat numérique du papier (App. B) : deltas, facteurs
`exp(Δs)`, poids et normaliseur **plein vocabulaire** sont calculés en
`float32` ; la correction n'est validée que si ce normaliseur est fini et
strictement positif. Tout débordement, sous-débitement total ou score non fini
retombe donc sur la **cible causale** (fallback), sans lever d'exception.

## Vérifier le squelette

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

## Périmètre

- **Dedans** : cible `p̃_T,i` (eq. 15–18), fallback causal, configuration,
  interfaces spécifiées.
- **Dehors (J12)** : entraînement, rollout, scoring professeur réel, benchmarks,
  GPU, modification des résultats/verdicts A2D.

Le dtype d'arithmétique est explicite : `float32` par défaut (App. B du
papier), surchargeable par `arithmetic_dtype="float64"` ou par
`DOPDConfig(arithmetic_dtype=...)`.

Voir [`docs/DOPD_SPEC.md`](docs/DOPD_SPEC.md) pour la spécification complète et
[`../a2d/docs/A2D_SPEC.md`](../a2d/docs/A2D_SPEC.md) pour la méthode A2D, qui est
indépendante.
