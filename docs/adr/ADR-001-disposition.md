# ADR-001 — Disposition du dépôt et règles de dépendance

- **Statut** : accepté (J9, 2026-10-22)
- **Décide par** : mission J9, phase 2/3
- **Remplace** : disposition antérieure (`scripts/`, `docs/`, `eval/`, `harness/`, `legacy/`, `vLLM/` à la racine)
- **Table de migration** : [`path-migration.md`](path-migration.md)

## Contexte

Le dépôt a grandi par missions successives (J6 → J8) : la bibliothèque A2D, les
scripts d'évaluation, le harnais Dream, l'infra vLLM et le pipeline legacy
Jainius vivaient dans des dossiers « à plat » (`scripts/`, `eval/`, `harness/`,
`legacy/`, `vLLM/`). Deux conséquences :

1. **Aucun code n'est adressable** : le code générique (statistiques, empreintes,
   E/S safetensors) était mêlé au code A2D (`scripts/_common.py`), et la
   bibliothèque A2D était importée par aménagement de `sys.path`.
2. **Les chemins sont ambigus** : `scripts/tests/` (tests A2D) cohabitait avec
   les tests d'infra ; rien ne distinguait « code » et « résultats ».

## Décision

### 1. Disposition

```
academiia/core/      code générique          → paquet `academiia.core`
modules/a2d/         module A2D              → paquet `academiia.a2d`
  ├── a2d/  cli/  eval/  harness/  tests/  docs/  legacy/  STATUS.md
infra/vllm/          service d'inférence
tests/               tests du core
docs/                documentation du projet (STATUS.md, ARCHITECTURE.md, adr/)
tools/               utilitaires du dépôt (check_links.py)
```

Les résultats de mesure restent **avec** le code qui les produit
(`modules/a2d/eval/`), sauf le legacy (`modules/a2d/legacy/eval/`).

### 2. Règle de dépendance

```
core  ←  a2d  ←  (d-OPD, à venir)
```

* `academiia.core` **n'importe aucun module** du dépôt ;
* un module importe `academiia.core`, **jamais un autre module** ;
* les imports sont **absolus** (`from academiia.a2d.sources import …`) :
  plus aucun `sys.path.insert`, plus aucun import par chemin de fichier ;
* le code d'un module est **auto-suffisant** : le copier ailleurs suppose de
  copier ses dépendances vers `core`.

### 3. Empaquetage

`pyproject.toml` (setuptools) déclare deux paquets installables avec
`pip install -e .` :

| Paquet | Répertoire source | Remarque |
|---|---|---|
| `academiia.core` | `academiia/core` | chemin standard |
| `academiia.a2d` | `modules/a2d/a2d` | `package-dir` : le paquet est dispersé dans l'arborescence du module |
| `academiia.a2d.eval` | `modules/a2d/eval` | scripts d'évaluation importables (verdict G1, bootstrap) |

`academiia.a2d.eval` a été retenu plutôt qu'un repli `import a2d` : les
scripts d'évaluation contiennent de la logique **pure** testable
(`verdict_gain`, `bootstrap_ci95`, `fingerprint`) ; les exposer comme paquet
permet de les tester sans aménagement de `sys.path`. Coût : un `__init__.py`
dans `modules/a2d/eval/`.

### 4. Ce qui n'est pas fait

* **Pas de réécriture d'historique** : les rapports J6–J8 restent intacts et
  portent un bandeau renvoyant à la table de migration.
* **Pas de module d-OPD** en J9 : il est *planifié* (arXiv 2609.35362 ; dev sur
  Qwen3-0.6B). Aucun code, aucun dossier. *(renommé d-OPD en J11)*
* **Pas de tag de release** : le tag `archive/pre-restructure-J8` fige l'état
  d'avant réorganisation ; ce n'est pas une release.

## Conséquences

* `pytest modules/a2d/tests tests infra/vllm/scripts/tests` fonctionne après
  `pip install -e .` ;
* la CI installe le projet (`pip install -e .`) avant toute invocation de CLI ;
* un module futur (d-OPD) s'ajoute sous `modules/dopd/` avec un paquet
  `academiia.dopd` qui n'importe que `academiia.core` ;
* tout nouveau `sys.path.insert` est une régression (à bloquer par relecture).
