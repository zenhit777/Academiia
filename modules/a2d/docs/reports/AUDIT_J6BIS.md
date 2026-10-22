# Audit indépendant de la mission J6bis — vérification en lecture seule

> **Document historique (J6–J8).** Les chemins cités dans ce rapport sont
> ceux de l'époque : la table de migration des chemins est
> [`docs/adr/path-migration.md`](../../../../docs/adr/path-migration.md).

**Date** : 2026-10-20 · **Auditeur** : agent (OpenCode) · **Mode** : lecture seule,
plus des écritures temporaires **dans le dépôt** (`_artefact/audit_j6bis/`).
**Objet** : vérifier les livrables A1–A3, B1–B7, C1–C7, D1–D3, E1–E4 de
`docs/reports/J6bis.md`, et **détecter toute déviation silencieuse** de la
méthode A2D (`docs/2610.08108v1 - A2D.pdf`, `docs/A2D_SPEC.md`).
**Aucun fichier suivi n'a été modifié. Aucune suppression n'a été faite.
Aucun job GPU n'a été lancé. `git status` est propre.**

---

## 0. Verdict global

**Aucune déviation silencieuse des équations, coefficients, mappings ou du
schéma numérique A2D n'a été trouvée dans le chemin Qwen2.5-7B → Dream-7B.**
Les écarts trouvés sont des défauts de **documentation**, de **garde-fous CLI**
et d'**inventaire** ; aucun ne change le résultat d'une fusion déjà produite.

| Constat | Gravité | Silencieux ? |
|---|---|---|
| F1 — `G1_PROTOCOL.md` attribue les gains +19,4 / +15,2 aux mauvaises tables/paires | **modérée** | non (visible, mais faux) |
| F2 — `--alpha`/`--lambda` par défaut à **0,0** : un « transfert » sans α écrit une copie de `D_b`, un « merge » sans λ une copie de `D_p`, **et la vérification post-écriture passe** | **élevée (risque d'usage)** | non silencieux techniquement, mais indiscernable à la lecture du résultat |
| F3 — `--no-verify` contourne la vérification dite « obligatoire » et ne laisse **aucune trace** au manifeste | modérée | non |
| F4 — compteurs de tests faux (`KNOWN_ISSUES.md` « 37 tests », `CHECKLIST.md` « 19 mutations ») | faible | non |
| F5 — latences IFEval contradictoires (162 s vs 326 s/prompt) et arithmétiques de budget fausses | faible | non |
| F6 — inventaire C5 incomplet et périmé (+ une écriture `/tmp` de l'audit lui-même) | faible | non |
| F7 — aide de `--lambda` annonce un seuil de 1,5, le code avertit à 1,0 | faible | non |
| F8 — `ITEM_FINGERPRINTS.json:git_hash` = commit du **fork Dream**, non étiqueté | info | non |
| F9 — YAML de tâche `do_sample: false` / `temperature: 0.0` vs décodage réel T=0,1 | info | **déjà déclaré** (J6bis §4 E14, AUDIT J6 MIN-04) |

---

## 1. Ce que l'audit a **re-vérifié par exécution** (pas par lecture)

### 1.1 Suite de tests reproduite

```
$ python -m pytest scripts/tests -q
57 passed in 39.59s
```

Collecte par fichier (`--collect-only`) :
`test_a2d_library.py` 22 · `test_a2d_mutation.py` 20 · `test_pipeline.py` 9 ·
`test_eval_compare.py` 6 → **57**. La décomposition de `J6bis.md` §6.4
(22 + 20 + 15) est **exacte** ; c'est la ligne A2 du même rapport qui est
fausse (cf. F4).

### 1.2 Chemin CLI CI rejoué intégralement

`_artefact/audit_j6bis/ci_replay_full.sh` (fixtures minimales, 5 tenseurs) :
transfert, merge, ablation-minus, refus d'α erroné, `merge` sans τ_D (exit 2),
τ_D partiel refusé en `strict`, `zero-fill` déclaré → **7/7 conformes** au
comportement documenté, y compris les codes de sortie.

### 1.3 Équations et moteur de fusion

* `mode_coefficients` (`scripts/a2d/merge.py:175`) : `transfer` → (α, 0) ;
  `merge` → (λ, −λ) ; `ablation-dp-plus-tau-a` → (α, 0) ;
  `ablation-dp-minus-tau-d` → (0, −λ).
  Avec `D_p = D_b + τ_D`, l'écriture `D_p + λτ_A − λτ_D` est **algébriquement**
  l'équation (2) `D_b + (1−λ)τ_D + λτ_A`. Conforme.
* **Une seule** définition de la valeur attendue (`expected_tensor`,
  `scripts/a2d/merge.py:511`) partagée par l'écrivain (`merge_checkpoint`) et le
  vérificateur (`verify_checkpoint`, donc `verify_merge`).
* Arithmétique **FP32, un seul cast BF16** (App. A.3) : vérifié sur la fusion
  réelle `_artefact/j6/smoke_merge_manifest.json` (α_eff = **0,912842** pour 0,9,
  borne BF16 mesurée ; familles `norms` α_eff = 0,9827 dans la borne élargie
  0,169 — ce sont les mêmes ordres que ceux rapportés par J6bis).
* Mapping App. A.2 : `--mapping identity` pour Qwen→Dream ; les vocabulaires
  sont **identiques** (152 064 des deux côtés, vérifié dans les `config.json`),
  donc l'intersection des lignes d'embedding est un no-op ici ; le chemin
  partagé la gère pour d'autres architectures.

### 1.4 Vérificateur v2

`verify_checkpoint` : régression des moindres carrés **signés** (Gram float64),
bornes dérivées du résidu `round_bf16(expected) − expected` (jamais de la
sortie écrite), repli sur projections simples si |cos| ≥ 0,99, tenseurs non
fusionnés **bit-identiques**, écart à `round_bf16(expected)` ≤ 1 ULP.
`verify_merge` → `validate_config` : `--mode merge` **exige** τ_D (exit 2) —
le trou MAJ-01 est bien fermé.

### 1.5 Livrables C/D déjà contrôlés (session d'audit)

* `ITEM_FINGERPRINTS.json` : 6 runs, empreintes `items/prompts/targets`
  **identiques base/instruct** par tâche (GSM8K 300, MATH-175, IFEval 8).
* `RESTRUCTURE_MAP.json` : **24/24** fichiers, taille + SHA-256 conformes
  après renommage ; plus aucun dossier `__mnt__d__…`.
* C4 : smoke GSM8K n=8 rejoué (3 min 46 s), items strictement inclus dans les
  300 de J6, fork reconstruit au commit `31f94a6`.
* D3 : identité des sorties inter-batch reproduite : **0/16 (b2), 5/16 (b4),
  4/16 (b8), 4/16 (b16)** — identique à `EVAL_BUDGET.md` §4. (Comptage par
  `doc_id` : les `samples_*.jsonl` GSM8K contiennent **une ligne par item et
  par filtre** — 600 lignes pour 300 items, `strict-match` et
  `flexible-extract` ; la déduplication par `doc_id` redonne exactement 16.)
* E1–E4 : IDs/licences/tailles HF vérifiés en direct (dont **HTTP 401** pour
  `nvidia/Nemotron-Labs-Diffusion-3B-Instruct`) ; 237 tenseurs de diffusion
  (Σ = 3 831 659 520 = `safetensors.total`) contre 458 côté AR ; lignes
  protégées `10–19, 24, 25, 32–35` + ligne de masque `100` ; RoPE 0,25 vs 16,0 ;
  EOS 2 vs 11 ; `docs.nvidia.com/nemo/skills` → 404, canonique
  `nvidia-nemo.github.io/Skills` → 200 ; TPF 5,89 confirmé comme **mesuré sur
  8B** (les coûts 3B sont bien étiquetés `[INFERENCE]`).
* C5 : les huit tailles principales sont exactes au bit près
  (`dream-harness` 7 588 593 361 ; HF 92 586 706 003 ; uv 40 329 711 647 ;
  pip 8 869 011 825 ; vLLM 3 641 214 554 ; nltk 15 204 406 ; `/tmp/tau_ref`
  4 013 949 312), cf. F6 pour les omissions.

### 1.6 Déviation A2D : limites **déclarées**, pas silencieuses

* **Adaptateurs DG mis à l'échelle (App. A.2)** : non implémenté ; G2 est
  explicitement **bloquée (B7)** et le mode « base + adaptateur remis à
  l'échelle » est une conception (`docs/design/G2_adapter_scaling.md`). La
  politique `strict` **refuse** d'ailleurs le cas réel G2 (413 modules
  d'adaptateur contre 205 AR) plutôt que de dégrader en transfert.
* **`--coverage strict`** ferme le trou de J6 (fusion τ_D conditionnelle qui
  transformait un merge en transfert sans alerte).

---

## 2. Anomalies détaillées

### F1 — Ancrages de `G1_PROTOCOL.md` attribués aux mauvaises tables (modérée)

`docs/design/G1_PROTOCOL.md:133-134` :

> « le papier reports **+19,4** points (Table 8, transfert Nemotron) et
> **+15,2** points (Table 7/12, merge) »

Dans le PDF :
* **+19,4** = Dream-7B **MATH-500 transfert** (20,8 → 40,2 ; Figure 2a, Tableaux 1/16) —
  pas une ligne de la Table 8, et pas Nemotron ;
* **+15,2** = Dream-7B **GSM8K transfert** (66,5 → 81,7 ; Figure 2a) — **pas un gain de
  merge** : les gains de merge de la Figure 2b / Table 16 sont IFEval +9,8,
  MATH-500 +6,6, GSM8K **+0,7** ;
* la Table 8 (transfert) pour Nemotron affiche des gains bien plus **grands** que
  +19,4 : **+34,4** (Ministral-3-3B-Instruct → Nemotron-LD-3B, IFEval 25,2 → 59,6),
  **+36,7** (8B-Instruct, 28,2 → 64,9) et **+29,4** (8B-Reasoning, 28,2 → 57,6).

**Impact** : les seuils C1 (+10 sur MATH-175) / C2 (+8 sur GSM8K) restent
défendables comme « frange basse » des gains de **transfert** de la paire
Dream, mais la justification écrite est fausse sur trois points (nature du
gain, table, famille). C3/C4 (±2 non-régression) ne dépendent pas de cet
ancrage.

### F2 — Coefficients par défaut à 0,0 : transfert/merge « no-op » vérifié OK (élevée)

`scripts/merge_a2d.py:56-59` : `--alpha` et `--lambda` ont pour défaut **0.0**.

Reproduit sur les fixtures CI :

| Commande | Sortie écrite | Vérification |
|---|---|---|
| `--mode transfer` **sans** `--alpha` | **copie BF16 de `D_b`** (bit-identique après cast) | `c_a_eff ≈ −4,2e−05` → OK |
| `--mode merge` **sans** `--lambda` | **copie BF16 de `D_p`** (bit-identique après cast) | `c_a_eff ≈ −9,9e−05`, `c_d_eff ≈ +1,4e−04` → OK |

Aucun avertissement n'est émis (`warnings: []`). La vérification est **correcte
par construction** : elle compare les coefficients effectifs aux coefficients
**demandés** (0,0). Elle ne peut pas détecter qu'on a demandé un merge sans λ.

Correctif de forme : mon analyse intermédiaire (« merge sans λ = `D_p − τ_D` »)
était **fausse** ; c'est bien une copie de `D_p`.

**Impact** : n'affecte aucun artefact existant (le run réel J6 passait α=0,9
explicite), mais le chemin J7 est exposé : la commande documentée est fournie
avec les coefficients, mais rien n'empêche un run « merge » à λ implicite de
produire un modèle identique à `D_p` que l'on croira ensuite « merge ».
Suggestion : refuser (ou exiger `--allow-empty-coefficient`) un
transfert/merge à coefficient nul quand le plan contient des deltas.

### F3 — `--no-verify` sans trace (modérée)

`README.md:50-51` annonce une « vérification post-écriture **obligatoire** »,
mais `--no-verify` (`scripts/merge_a2d.py:82`) la désactive et le manifeste
n'en garde **aucune trace** : plus de bloc `post_write_verification`, et
`warnings`/`notes` restent vides (vérifié sur
`_artefact/audit_j6bis/ci/transfer_noverify/merge_config.json`).
Suggestion : soit retirer l'option, soit inscrire
`"post_write_verification": {"skipped": true, "reason": "--no-verify"}`.

### F4 — Compteurs de tests (faible)

| Emplacement | Dit | Réel |
|---|---|---|
| `KNOWN_ISSUES.md:10` | « 37 tests au total dans `scripts/tests` » | **57** (22+20+9+6) |
| `J6bis.md` ligne A2 | répète « 22 tests A2D / 37 au total » | **57** — contredit par `J6bis.md` §6.4 lui-même |
| `CHECKLIST.md:35` | « 19 tests de mutation » | **20** (`test_a2d_mutation.py`) |

La correction MIN-05 a donc remplacé « 19 » par « 37 » — un nombre toujours
faux.

### F5 — Latences IFEval et projections (faible)

* `docs/reports/J6.md:231` : « **162 s/prompt** (batch 1), ~320 s pour un lot de
  4 ⇒ ≈ **24 h** ». 541 × 162 ≈ **24,3 h** (taux batch 1) ; au taux mesuré
  batch 4 de `EVAL_BUDGET.md` (326 / 327,6 s/prompt), IFEval complet coûte
  **≈ 49 h** par modèle — ce que dit correctement `EVAL_BUDGET.md` §3. La
  phrase mélange deux taux et conclut avec l'ancien.
* Vestige du même 162 dans `harness/dream/summarize.py:149` et
  `eval/dream_a2d/SUMMARY.md:32` (« 162 s/prompt pour IFEval ») au lieu de
  ~326 s.
* `EVAL_BUDGET.md:36-37` : « quasi linéaire (256 → 512 ×3,4 ; 256 → 1280
  ×32) » compare des **tâches différentes** (GSM8K vs MATH vs IFEval) ; la
  superposition « un pas par token » prédit ×2 pour 256→512, pas ×3,4. Le
  qualificatif « quasi linéaire » n'est pas démontré par ces trois points.
* `EVAL_BUDGET.md:64` : « MATH-500 : ×10,3 ⇒ 4 h 39 » — 5 000→500 est un
  facteur **10** (⇒ **4 h 47**), et la notation « ×10,3 » est inversée.

### F6 — Inventaire C5 : omissions et taille périmée (faible)

Manquent de `docs/reports/OUT_OF_WORKSPACE.md` :
* `/tmp/SUMMARY_test.json` (48 867 o) et `/tmp/SUMMARY_test.md` (2 814 o),
  2026-10-20 13:03 — écrits hors dépôt, **jamais inventoriés** ;
* `/tmp/a2d_smoke_transfer` (checkpoint de transfert α=0,9 de J6, cité en
  `J6.md:63` et `J6.md:309` ; ≈ 15 Go, poids 7B BF16) : absent de l'inventaire
  et **plus présent sur disque**. Aucune suppression n'est imputée ici
  (nettoyage `/tmp` possible), mais la ligne manque ;
* `/tmp/pytest-of-sjteb` : 591 207 o annoncés, **8 483 981 o** mesurés.

**Transparence sur l'audit lui-même** : les relances de `pytest` de l'audit ont
contribué à cette croissance (une relance avec `TMPDIR` par défaut avant que je
le redirige vers `_artefact/audit_j6bis/pytest_tmp/`). Je n'ai **rien supprimé**
hors du dépôt ; `/tmp` reste en l'état, en attente de votre accord écrit.

### F7 — Aide de `--lambda` (faible)

`merge_a2d.py:57` et `:59` annoncent tous deux « avertissement > 1.5 ». Le code
avertit pour **α > 1,5** mais pour **λ > 1,0** (`LAMBDA_MAX`, `merge.py:68`),
seuil justifié par l'équation (2) dans [0,1]. Aide à corriger (et le nom
`LAMBDA_MAX` est un seuil d'avertissement, pas un plafond dur).

### F8 — Provenance du champ `git_hash` (info)

`eval/dream_a2d/ITEM_FINGERPRINTS.json` porte `"git_hash": "31f94a6"` : c'est le
commit du **fork `DreamLM/Dream`** recopié depuis les `results_*.json` du fork
(`scripts/eval_item_fingerprint.py`), **pas** un commit d'Academiia. Le nom du
champ est ambigu ; renommer `fork_git_hash` et ajouter le commit du dépôt.

### F9 — `do_sample: false` vs T=0,1 (info, déjà déclaré)

`harness/dream/tasks/ifeval_a2d.yaml:19-20` (et `gsm8k_cot_a2d.yaml:35`)
déclarent `do_sample: false` / `temperature: 0.0`, alors que le run impose
`temperature=0.1, top_p=0.9, alg=entropy` via `--model_args`
(`run_evals.sh:112`). Le fork `diffllm` utilise les seconds. L'écart est
**signalé** (AUDIT J6 MIN-04, J6bis §4 E14) : pas silencieux, mais le YAML reste
trompeur pour un tiers qui lit la tâche seule.

---

## 3. Ce qui reste non atteint (conforme aux déclarations)

* **G1 : ouverte.** Aucun modèle fusionné n'a été évalué. Le seul artefact de
  fusion réelle (transfert α=0,9) est mécanique, et il n'existe plus sur disque.
* **C0.5 : NON ATTEINT** (Base MATH-175 +3,2 hors ±3 ; IFEval n=8 non mesuré).
* **J7** non exécuté (hors périmètre), **G2** bloquée (mode adaptateur absent),
  **G1b** faisabilité seule, **G3** non démarrée.
* Le push des 5 commits J6bis n'a pas été fait par l'agent ; depuis, 7 commits
  locaux sont en attente (`origin/main` = `284fb62`), dont `1d03e27` (rapport) et
  `35682cb` (fix familles à source unique) — postérieurs au rapport.

## 4. Limites de cet audit

Non rejoué : les 6 évaluations GPU J6 (pas de job GPU), les téléchargements de
poids, le `du` complet des caches (valeurs de C5 relues après recoupement partiel
uniquement). Les contrôles HF/GitHub sont des lectures d'API en direct. Le texte
du PDF utilisé est l'extraction locale `_artefact/audit_j6bis/a2d.txt`.

## 5. Recommandations minimales (par ordre d'utilité)

1. **F2** — transférer/merge avec coefficient nul : refus par défaut ou option
   explicite ; au minimum un `warnings` au manifeste.
2. **F3** — tracer `--no-verify` dans le manifeste (ou supprimer l'option).
3. **F1** — corriger les ancrages de `G1_PROTOCOL.md` : +19,4 = MATH-500
   transfert Dream (Figure 2a, Tableaux 1/16) ; +15,2 = GSM8K transfert Dream ;
   le plus grand gain de merge Dream est +9,8 (IFEval) ; les gains Nemotron de
   la Table 8 sont ≈ +29 à +37 (IFEval).
4. **F4** — `KNOWN_ISSUES.md` : 57 tests ; `CHECKLIST.md` : 20 mutations ;
   corriger la ligne A2 de `J6bis.md`.
5. **F5** — harmoniser le récit IFEval sur le taux batch 4 (~326 s/prompt,
   ≈ 49 h) dans `J6.md`, `summarize.py`, `SUMMARY.md` ; corriger ×3,4 et ×10,3.
6. **F6** — compléter C5 (SUMMARY_test, smoke_transfer, taille pytest) ;
   décider du sort de `/tmp`.
7. **F7** — aide de `--lambda` : seuil 1,0.

---

## Annexe — commandes de reproduction

```bash
# tests
python -m pytest scripts/tests -q

# no-op par défaut (fixtures CI)
python scripts/merge_a2d.py --mode transfer \
  --diffusion-base _artefact/audit_j6bis/ci/d_base \
  --tau-a-post _artefact/audit_j6bis/ci/ar_post \
  --tau-a-base _artefact/audit_j6bis/ci/ar_base \
  --mapping identity --output _artefact/audit_j6bis/ci/transfer_default
python scripts/merge_a2d.py --mode merge \
  --diffusion _artefact/audit_j6bis/ci/d_post \
  --tau-a-post _artefact/audit_j6bis/ci/ar_post \
  --tau-a-base _artefact/audit_j6bis/ci/ar_base \
  --tau-d-post _artefact/audit_j6bis/ci/d_post \
  --tau-d-base _artefact/audit_j6bis/ci/d_base \
  --mapping identity --output _artefact/audit_j6bis/ci/merge_default
# → comparer à cast_bf16(D_b) / cast_bf16(D_p) : bit-identiques

# --no-verify
python scripts/merge_a2d.py --mode transfer ... --alpha 0.5 --no-verify \
  --output _artefact/audit_j6bis/ci/transfer_noverify
# → manifeste sans bloc post_write_verification

# chemin CI complet
bash _artefact/audit_j6bis/ci_replay_full.sh
```

Artefacts de l'audit : `_artefact/audit_j6bis/` (ignoré par git), dont
`ci_replay_full.sh`, `check_*.py`, `a2d.txt` (extraction PDF), `ci/`.
