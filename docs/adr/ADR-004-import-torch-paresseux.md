# ADR-004 — Import de `torch` paresseux : **reporté** (M3)

**Statut** : accepté — *reporté* · **Date** : 2026-10-22 (mission J11) ·
**Audit d'origine** : constat externe **M3** (« imports torch paresseux »).

## 1. Contexte

`import academiia.a2d` charge aujourd'hui `torch` :

```
$ python -X importtime -c "import academiia.a2d"
import time:   10431 |  1191861 | academiia.a2d      # ≈ 1,2 s cumulés
$ python -c "import time; t=time.perf_counter(); import torch; print(time.perf_counter()-t)"
1.03                                          # torch seul ≈ 1,0 s
```

Cause : `modules/a2d/a2d/__init__.py` réexporte `adapters`, `merge` et `sources`, et ces
modules utilisent `torch` **au niveau module**, pas seulement dans les fonctions :

| Emplacement | Usage au niveau module |
|---|---|
| `modules/a2d/a2d/adapters.py:39` | `ADAPTER_DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32}` |
| `modules/a2d/a2d/merge.py:171` | `dtype: torch.dtype = torch.bfloat16` (défaut de champ de dataclass) |
| `modules/a2d/a2d/merge.py:154` | annotations `torch.Tensor` évaluées à la création de la classe (pas de `from __future__ import annotations`) |
| `modules/a2d/a2d/sources.py:28` | import de `torch` + annotations `torch.Tensor` |

Conséquence : `academiia.a2d.mapping`, `academiia.a2d.textcheck` et
`academiia.a2d.verdict` — qui n'ont **aucun** besoin de torch — ne sont pas importables
sans torch. C'est principally un coût de **temps d'import** (≈ 1 s), pas un blocage
fonctionnel : `torch` est une dépendance runtime déclarée de la bibliothèque.

## 2. Décision

**M3 est reporté** : l'import de `torch` reste *eager* jusqu'à un refactor dédié.
Aucune modification du cœur de la bibliothèque n'est faite dans J11 (mission de
stabilisation : « sans changer la disposition du dépôt ni la logique A2D »).

## 3. Ce qu'exigerait le refactor (à faire dans une mission dédiée)

1. `from __future__ import annotations` dans `merge.py`, `sources.py`, `adapters.py`
   (annotations différées) ;
2. résolution **paresseuse** des dtypes : `ADAPTER_DTYPES` et le défaut
   `dtype = torch.bfloat16` ne doivent plus être évalués à l'import ;
3. `__init__.py` en PEP 562 (`__getattr__`) pour différer l'import de `adapters`,
   `merge` et `sources` ;
4. **tests** : valeurs par défaut (BF16), `dataclasses.fields()` / `typing.get_type_hints`
   sur `MergeConfig`, et non-régression du premier cast BF16 (App. A.3).

## 4. Risques du refactor (motif du report)

* les **défauts de dtype** sont le cœur de la garantie « un seul cast BF16 » : une
  résolution paresseuse mal câblée changerait silencieusement le résultat numérique ;
* les annotations lues à l'exécution (`dataclasses.fields`, `typing.get_type_hints`)
  deviennent des chaînes avec `from __future__ import annotations` ;
* la surface touchée est le **cœur** de la bibliothèque (`merge.py`, `sources.py`,
  `adapters.py`), donc hors périmètre d'une mission de stabilisation documentaire.

## 5. Condition de reprise

À faire **avant** l'introduction du module **d-OPD** (J12) si celui-ci a besoin
d'importer `academiia.a2d` sans torch (par exemple pour un utilitaire de
manifeste léger), sinon au plus tard lors d'une mission « performance » dédiée.

## 6. Conséquences

* `import academiia.a2d` : ≈ 1,2 s (inchangé, mesuré le 2026-10-22) ;
* la CI, les entry-points et les tests ne sont pas impactés ;
* aucune mesure, aucun seuil, aucun verdict n'est modifié par ce report.
