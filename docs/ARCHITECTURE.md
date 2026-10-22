# Architecture

Document de référence de l'architecture **matérielle** du dépôt (J9, 2026-10-22).
Il décrit *ce qui est*, pas ce qui devrait être : les décisions sont dans les
ADR (`docs/adr/`), les états dans `docs/STATUS.md` et
`modules/a2d/STATUS.md`.

## 1. Vue d'ensemble

```
                      ┌───────────────────────────────┐
   sources AR         │  modules/a2d (paquet          │
   (A_b, A_p)  ──────▶│  academiia.a2d)               │
                      │  • a2d/  bibliothèque          │
   adaptateurs LoRA ─▶│  • cli/  fusion, vérification │
   (τ_D)              │  • eval/ verdict, bootstrap   │
                      │  • harness/ évaluation Dream  │
   D_b / D_p ────────▶│  • legacy/ Jainius (déprécié) │
                      └───────────────┬───────────────┘
                                      │ produit des checkpoints
                                      ▼
                      ┌───────────────────────────────┐
   measurements       │  modules/a2d/eval/            │
   (samples, JSON)  ─▶│  résultats versionnés          │
                      └───────────────┬───────────────┘
                                      │ analysés par
                                      ▼
                      ┌───────────────────────────────┐
                      │  academiia.core (générique)   │
                      │  stats · empreintes · E/S      │
                      └───────────────────────────────┘

   infra/vllm/  : service d'inférence (compose FP8 épinglé), sans dépendance
                   vers un module.
```

## 2. Modules

| Module | Rôle | Dépend de | Tests |
|---|---|---|---|
| `academiia.core` | code **générique** : IC bootstrap apparié, intervalle de Wilson, empreintes SHA-256, index HuggingFace, E/S safetensors, heuristiques de texte | — | `tests/test_core.py` |
| `modules/a2d` | **méthode A2D** : mapping de noms, sources de vecteur (paire de checkpoints / LoRA), moteur de fusion, adaptateurs mis à l'échelle, CLI, scripts d'évaluation, harnais, documentation, pipeline legacy | `academiia.core` | `modules/a2d/tests/` |
| `infra/vllm` | service d'inférence (compose, smoke, diagnostic) | — | `infra/vllm/scripts/tests/` |
| `modules/opdlm` (**planifié**) | encapsulation du code officiel OPDLM + checkpoints 4B (Qwen3-4B comme modèle de référence) | `academiia.core` | — |

### Règle de dépendance (ADR-001)

```
core  ←  a2d  ←  opdlm (à venir)
```

* `academiia.core` n'importe **aucun** module du dépôt ;
* un module importe `academiia.core` et **jamais** un autre module ;
* les imports sont absolus (`from academiia.a2d.sources import …`) ;
* aucun `sys.path.insert` : le code est importé depuis le paquet installé.

## 3. Paquets et installation

```bash
pip install -e .        # installe academiia.core et academiia.a2d
python -c "import academiia.core, academiia.a2d; print(academiia.a2d.__file__)"
```

| Paquet | Source (`package-dir`) |
|---|---|
| `academiia.core` | `academiia/core` |
| `academiia.a2d` | `modules/a2d/a2d` |
| `academiia.a2d.eval` | `modules/a2d/eval` |

Les CLI restent des **scripts** (`modules/a2d/cli/*.py`) : elles s'invoquent par
leur chemin, elles importent le paquet installé.

## 4. Le pipeline A2D (module `a2d`)

```
sources ──▶ vecteur τ (recalculé à la volée, FP32, jamais persisté)
        ──▶ plan de fusion (UNION des modules de τ_A et τ_D)
        ──▶ fusion (arithmétique FP32, UN SEUL cast BF16, écriture atomique)
        ──▶ vérification post-écriture (α_eff / λ_eff signés, 0 ULP)
        ──▶ vérification post hoc (verify_merge : α/λ attendus vs effectifs)
        ──▶ évaluation (harnais Dream, protocole App. A.1)
        ──▶ verdict de porte (G1 : critères + IC appariés)
```

| Étape | Code | Sortie |
|---|---|---|
| transfert / merge / ablations | `modules/a2d/cli/merge_a2d.py` | checkpoint fusionné + `merge_config.json` |
| vérification post hoc | `modules/a2d/cli/verify_merge.py` | verdict, code de sortie |
| géométrie (Table 21) | `modules/a2d/cli/measure_geometry.py` | JSON de mesures |
| intégrité HF | `modules/a2d/cli/verify_checkpoint.py` | contrôle stricte |
| adaptation à l'échelle | `academiia.a2d.adapters` | adaptateur `lora_B × (1−λ)` |
| verdict de porte | `modules/a2d/eval/g1_verdict.py` | `G1_VERDICT.json` |
| IC appariés | `academiia.core.stats` + `modules/a2d/eval/paired_bootstrap.py` | IC 95 % |

## 5. Règle de deux (ADR-003)

Une règle scientifique a **une** implémentation canonique. Le pipeline legacy
(`modules/a2d/legacy/`) est un *wrapper* déprécié de la bibliothèque, tenu
par un **test d'égalité**
(`modules/a2d/tests/test_pipeline.py::test_wrapper_legacy_egal_bibliotheque_a2d`)
: si le wrapper diverge, le test échoue.

## 6. Provenance (ADR-002, brouillon)

Chaque fusion écrit `merge_config.json` (coefficients, hashes d'index des
sources, commit git, normes par famille). Le **manifeste de lignée**
(`lineage.json` : parent, module producteur, version, hashes des fichiers
produits) est **schématisé mais non implémenté** : ADR-002 fixe le format.

## 7. Documentation

| Document | Contenu |
|---|---|
| `README.md` | pipeline, modules, portes, résultats |
| `docs/STATUS.md` | index court des modules |
| `docs/ARCHITECTURE.md` | ce document |
| `docs/adr/ADR-001-disposition.md` | disposition et règles de dépendance |
| `docs/adr/ADR-002-provenance.md` | manifeste de provenance (brouillon) |
| `docs/adr/ADR-003-regle-de-deux.md` | règle de deux |
| `docs/adr/path-migration.md` | ancien → nouveau chemin, exceptions |
| `modules/a2d/STATUS.md` | état détaillé du module A2D |
| `modules/a2d/docs/reports/**` | rapports J6–J8 (historiques, bandeau de migration) |

## 8. Ce qui n'existe pas encore

* `modules/opdlm/` : **planifié**, aucun code (J9).
* Le manifeste de lignée `lineage.json` : **schéma** sans implémentation.
* Le module OPDLM devra produire le même format de manifeste qu'A2D
  (`producer.module = "opdlm"`) et ne **pas** dupliquer les règles A2D.
