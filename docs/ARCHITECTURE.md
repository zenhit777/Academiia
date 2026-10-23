# Architecture

Document de référence de l'architecture **matérielle** du dépôt (J9, 2026-10-22 ;
mis à jour J12, 2026-10-23). Il décrit *ce qui est*, pas ce qui devrait être :
les décisions sont dans les ADR (`docs/adr/`), les états dans `docs/STATUS.md`
et `modules/a2d/STATUS.md`.

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
                                      ▲
                                      │ (futur harnais)
                      ┌───────────────┴───────────────┐
                      │  modules/dopd (paquet          │
                      │  academiia.dopd)               │
                      │  • dopd/ cible future-aware    │
                      │  • docs/ brief + rapports      │
                      │  • tests/ CPU                  │
                      └───────────────────────────────┘

   infra/vllm/  : service d'inférence (compose FP8 épinglé), sans dépendance
                   vers un module.
```

## 2. Modules

| Module | Rôle | Dépend de | Tests |
|---|---|---|---|
| `academiia.core` | code **générique** : IC bootstrap apparié, intervalle de Wilson, empreintes SHA-256, index HuggingFace, E/S safetensors, heuristiques de texte | — | `tests/test_core.py` |
| `modules/a2d` | **méthode A2D** : mapping de noms, sources de vecteur (paire de checkpoints / LoRA), moteur de fusion, adaptateurs mis à l'échelle, CLI, scripts d'évaluation, harnais, documentation, pipeline legacy | `academiia.core` | `modules/a2d/tests/` |
| `modules/dopd` | **d-OPD** (arXiv 2609.35362) : cible future-aware, configuration ; développement prévu sur Qwen3-0.6B | `academiia.core` | `modules/dopd/tests/` |
| `infra/vllm` | service d'inférence (compose, smoke, diagnostic) | — | `infra/vllm/scripts/tests/` |

### Règle de dépendance (ADR-001)

```
core  ←  a2d
core  ←  dopd

a2d   ↛  dopd        (interdit)
dopd  ↛  a2d         (interdit)
```

* `academiia.core` n'importe **aucun** module du dépôt ;
* un module importe `academiia.core` et **jamais** un autre module ;
* les imports sont absolus (`from academiia.a2d.sources import …`) ;
* aucun `sys.path.insert` : le code est importé depuis le paquet installé.

`modules/dopd/` applique la règle à la lettre : un test statique
(`modules/dopd/tests/test_dopd_package.py`) refuse tout import de
`academiia.a2d` ou d'une bibliothèque d'entraînement lourde.

## 3. Paquets et installation

```bash
pip install -e .        # installe academiia.core, academiia.a2d et academiia.dopd
python -c "import academiia.core, academiia.a2d, academiia.dopd; print(academiia.dopd.__file__)"
```

| Paquet | Source (`package-dir`) |
|---|---|
| `academiia.core` | `academiia/core` |
| `academiia.a2d` | `modules/a2d/a2d` |
| `academiia.a2d.eval` | `modules/a2d/eval` |
| `academiia.dopd` | `modules/dopd/dopd` |

Les CLI restent des **scripts** (`modules/a2d/cli/*.py`) : elles s'invoquent par
leur chemin, elles importent le paquet installé. d-OPD n'expose **aucun**
entry-point en J12.

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

## 5. Le module d-OPD (squelette J12)

d-OPD n'est **pas** une fusion de poids : c'est une distillation on-policy qui
corrige la cible du professeur AR avec le futur visible du bloc. Le module ne
partage donc aucune règle avec A2D (ADR-003) et ne l'importe pas.

```
rollout élève ──▶ état de débruitage X_t + bloc complété x̂
             ──▶ prior causal r_i(v) = p_T(v | q, z, x̂_<i)
             ──▶ scores de suffixe s_i(v) pour l'union top-k
             ──▶ Δs_i(v) = s_i(v) − s_i(x̂_i)
             ──▶ cible p̃_T,i ∝ r_i · exp(Δs_i)   (fallback causal)
             ──▶ KL avant D_KL(p̃_T,i ‖ p_θ,i)
```

**État J12** : seule la brique de cible est implémentée, en NumPy pur et sans
torch (`academiia.dopd.compute_future_aware_target`). Le scoring professeur, le
rollout, le vérificateur et l'entraînement sont **spécifiés** dans
`modules/dopd/docs/DOPD_SPEC.md` mais non implémentés. Aucun checkpoint n'est
téléchargé, aucun GPU n'est requis.

## 6. Règle de deux (ADR-003)

Une règle scientifique a **une** implémentation canonique. Le pipeline legacy
(`modules/a2d/legacy/`) est un *wrapper* déprécié de la bibliothèque, tenu
par un **test d'égalité**
(`modules/a2d/tests/test_pipeline.py::test_wrapper_legacy_egal_bibliotheque_a2d`)
: si le wrapper diverge, le test échoue.

d-OPD n'appelle pas A2D et ne duplique aucune règle A2D : les deux méthodes sont
indépendantes (pas de `τ`, pas de coefficient `α`/`λ`). Si une règle commune
apparaissait (statistiques, empreintes), elle irait dans `academiia.core`.

## 7. Provenance (ADR-002, brouillon)

Chaque fusion A2D écrit `merge_config.json` (coefficients, hashes d'index des
sources, commit git, normes par famille). Le **manifeste de lignée**
(`lineage.json` : parent, module producteur, version, hashes des fichiers
produits) est **schématisé mais non implémenté** : ADR-002 fixe le format.

Un artefact d-OPD devra utiliser le **même** schéma avec
`producer.module = "dopd"` et un bloc `config` contenant `DOPDConfig`, la seed,
la taille de bloc et `k`. Il ne doit pas inventer un `merge_config.json`
concurrent.

## 8. Documentation

| Document | Contenu |
|---|---|
| `README.md` | pipeline, modules, portes, résultats |
| `docs/STATUS.md` | index court des modules |
| `docs/ARCHITECTURE.md` | ce document |
| `docs/adr/ADR-001-disposition.md` | disposition et règles de dépendance |
| `docs/adr/ADR-002-provenance.md` | manifeste de provenance (brouillon) |
| `docs/adr/ADR-003-regle-de-deux.md` | règle de deux |
| `docs/adr/ADR-004-import-torch-paresseux.md` | import de torch paresseux (M3 reporté) |
| `docs/adr/path-migration.md` | ancien → nouveau chemin, exceptions |
| `modules/a2d/STATUS.md` | état détaillé du module A2D |
| `modules/a2d/docs/reports/**` | rapports A2D J6–J11bis (historiques, bandeau de migration) |
| `modules/dopd/docs/DOPD_SPEC.md` | brief de conception d-OPD |
| `modules/dopd/docs/reports/**` | rapports du module d-OPD (J12 →) |

## 9. Ce qui n'existe pas encore

* **d-OPD entraînement** : scoring professeur, rollout, vérificateur, boucle
  on-policy et évaluation ne sont pas implémentés (J12 ne livre que la cible et
  la configuration).
* Le manifeste de lignée `lineage.json` : **schéma** sans implémentation.
* Le module d-OPD devra produire le même format de manifeste qu'A2D
  (`producer.module = "dopd"`) et ne **pas** dupliquer les règles A2D.
