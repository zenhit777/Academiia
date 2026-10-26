# Audit de la mission J18 — décision à budget borné (d-OPD)

**Date** : 2026-10-26 · **HEAD audité** : `10c9d31ff8b3e80765393b4348ebc0ff81c5b2ba` · **HEAD de départ J18** : `659e32a7d22f5d570c5ce43b4ae068e2aebbab2b`
**Auditeur** : agent de codage (OhMyPi) · **Périmètre** : mission J18, Phases 0–4 · **Mode** : lecture des artefacts versionnés, recalculs ciblés sur `results.json`, exécution de la CI locale, **aucune exécution GPU**.

> **Source primaire.** arXiv:2609.35362v1 (Liu, Hu, Han, *d-OPD: Future-Aware On-Policy Distillation for Block Diffusion Language Models*, 28 sept. 2026) : <https://arxiv.org/html/2609.35362v1>. Les affirmations « papier » ci-dessous sont vérifiées sur le HTML arXiv, section par section.

> **Résumé.** Le cœur scientifique de J18 est **intègre** : le pré-enregistrement est intact (Phase 0, 1), le sous-ensemble de prompts est fixé et hashé avant tout run GPU, le budget est tenu, le manifeste ADR-002 est complet, aucune écriture hors périmètre, la CI est verte (266 passed, ruff vert). Le verdict **GO** est la décision correcte au sens des critères C1–C10. **Sept écarts** ont cependant été relevés, dont quatre sont des **affirmations fausses ou non traçables** dans les documents versionnés : (A) la référence d'exactitude déclarée « rejeu intégral FP32 sans cache » n'est pas le chemin utilisé (le scorer passe par le chemin en cache) ; (B) l'énoncé C4 « Test des signes par prompt » est implémenté sur les marginaux token (4680) sans aucune parité déclarée ; (C) la commande exacte consignée (`producer.command`) n'est pas rejouable ; (D) le rapport revendique « sans réserve » et « 120 blocs/config », ce qui est en partie factuellement erroné (1 560 blocs échantillonnés ; 4680/4680 positifs sont des marginaux token). Trois écarts de méthode : (E) le manifeste omet `mean_causal_kl`/`mean_future_kl`, rendant le ΔKL=3,047 non interprétable en réduction relative, et son seul correctif (modification de `diagnostic.py`) est interdit en J18 par C10 ; (F) la suite de tests du harnais ne pinne ni la construction future-aware (faute F4 indétectable) ni l'unité de bootstrap (faute F3 indétectable) ; (G) la mesure du temps mélange `time.time` et `time.monotonic`, ce qui corrompt l'affichage de progression.

---

## 1. Périmètre et méthode

### 1.1 Artefacts audités

| Élément | Rôle |
|---|---|
| `modules/dopd/docs/reports/J18_PRE.md` | pré-enregistrement figé (HEAD `659e32a7`) |
| `modules/dopd/docs/reports/J18.md` | rapport de mission J18 |
| `modules/dopd/harness/run_j18_decision.py` | harnais GPU (inférence seule, ADR-001) |
| `modules/dopd/harness/teacher_scorer_qwen3.py` | TeacherScorer de production (isolé sous `harness/`) |
| `modules/dopd/dopd/diagnostic.py` | noyau NumPy pur (WRAPPER, J16 correctifs) |
| `modules/dopd/eval/j18_decision/prompts.json` | sous-ensemble de prompts (sha256 fixé avant run) |
| `modules/dopd/eval/j18_decision/results.json` | résultat versionné (sha256 `4e891a3d…225e`) |
| `modules/dopd/tests/test_dopd_harness_j18_decision.py` | suite de tests du nouveau harnais |
| `docs/STATUS.md`, `modules/a2d/STATUS.md` | bandeaux J18 |
| `docs/adr/ADR-002-provenance.md` | contrat du manifeste |
| `_artefact/j18_run.log` | log du run GPU |
| arXiv:2609.35362v1 | source primaire |

### 1.2 Méthode

1. **Vérification d'ascendance du pré-enregistrement** : `git diff 659e32a7..HEAD -- J18_PRE.md` est vide ; le commit d'acceptation Phase 0 (`18feb51`) porte la mention « Pré-enregistrement J18 accepté sans modification. »
2. **Vérification de l'immutabilité du sous-ensemble** : `prompts.json` est commité dans `18feb51`, avant tout code producteur (`e246066` et suivants).
3. **Recalcul des chiffres de tête** à partir de `results.json` et dénombrement des critères.
4. **Injection de fautes** rigoureuse (patch de toutes les liaisons de fonction, pas un seul module) pour valider que la suite de tests détecte les régressions sémantiques (sign KL, axe KL, bootstrap, future-aware).
5. **Exécution réelle de la CI** : `ruff check` vert, suite 266 passed, 0 échec.
6. **Confrontation à la source primaire** et à J15/J16/J17 pour la comparabilité des échelles.
7. **Vérification de périmètre** : aucun `StudentPolicy`, aucun entraînement/backward/optimizer, aucune écriture de checkpoint, aucune échelle > Qwen3-0.6B, aucun push, aucun artefact J14–J17 régénéré, aucun chargement de modèle hors dépôt.

### 1.3 Environnement d'audit

| Élément | Valeur |
|---|---|
| HEAD audité | `10c9d31ff8b3e80765393b4348ebc0ff81c5b2ba` |
| HEAD preregistration | `659e32a7d22f5d570c5ce43b4ae068e2aebbab2b` |
| Commit producteur | `30594e30bbe7a2eac880ce0f39a23df016da82b1` |
| Python | 3.12.3 |
| Install | `pip install -e . --no-deps` dans `_artefact/` (règle 9) |
| GPU | **aucun** — audit entièrement CPU |
| CI reproduite | `ruff check` → All checks passed ; `pytest ...` → 266 passed |

---

## 2. Ce qui est correct

Ces points ont été vérifiés indépendamment et sont conformes.

| # | Point | Preuve |
|---|---|---|
| P1 | **Pré-enregistrement intègre** | `git diff 659e32a7..HEAD -- modules/dopd/docs/reports/J18_PRE.md` → vide ; acceptation Phase 0 consignée dans `18feb51` |
| P2 | **Sous-ensemble fixé avant le run** | `prompts.json` commité dans `18feb51` (Phase 0/1), hash `593c32e5…b203f35` dans `lineage.data.prompt_subset_hash` |
| P3 | **Budget tenu** | wall_seconds `433,85 s` ≤ 14 400 s ; `cuda_allocated_peak_bytes` = 2876,5 MiB ≤ 16 GiB ; 13 prompts ≤ 50 ; ≤ 32 candidats + référence = 33 branches |
| P4 | **Modèle lu dans le dépôt** | `model_source_in_repo = true`, snapshot présent sous `_artefact/hf-cache/` (règle 9) ; le harnais refuse tout chargement hors dépôt |
| P5 | **Professeur gelé** | `_teacher_is_frozen(scorer)` = true : `requires_grad=False`, `model.training=False`, aucun `backward`/optimizer |
| P6 | **C9 provenance complète** | empreintes de `model.safetensors` + index dans `lineage.parent.hashes` |
| P7 | **Intégrité numérique** | `max_sum_error` = 4,44e-16 ; aucune NaN/Inf |
| P8 | **C10 non-régression** | `git diff 659e32a7..HEAD -- target.py diagnostic.py dream_a2d/` est vide au moment du run ; commit producteur `30594e3` == blob exécute (`git diff 30594e3..HEAD -- harness diagnostic.py` est vide) |
| P9 | **C1/C2/C3 tenus** | ΔKL = 3,047, CI 95 % [2,968 ; 3,127] exclut zéro ; médiane 1,961 > 0 ; 13/13 prompts significatifs |
| P10 | **C6–C8 tenus** | intégrité numérique, manifeste ADR-002, professeur gelé |
| P11 | **CI verte** | 266 passed ; ruff All checks passed |

---

## 3. Écarts majeurs

### Écart A — CRITIQUE — « rejeu intégral FP32 sans cache » est le chemin de référence déclaré, pas le chemin utilisé

**Constat.** `J18_PRE.md` §1 (table, ligne « Référence d'exactitude ») prescrit :

> **Rejeu intégral FP32** (contexte complet rejoué par branche, **sans cache**)

et le même paragraphe justifie :

> Le cache est plus lent que le rejeu aux préfixes courts : mesurer l'effet **sans** le biais du cache

La mission Phase 2 confirme : « Le cache de préfixe peut être instrumenté pour diagnostic de latence, mais n'est pas la référence d'exactitude. »

**Implémentation.** `run_j18_decision.py` appelle `scorer.causal_prior(prefix)` puis `scorer.suffix_loglikelihood(prefix, candidates, suffix)` — qui est le **chemin en cache** (préremplissage prefix + réutilisation du KV-cache entre branches, `teacher_scorer_qwen3.py:279-323`). `grep replay run_j18_decision.py` → **aucune occurrence**. Le chemin de référence complet `replay_scores`/`replay_job` (`teacher_scorer_qwen3.py:348-423`, `use_cache=False`) n'est jamais appelé.

**Preuve empirique.** Le log du run (`_artefact/j18_run.log`) affiche une durée de progression négative (voir Écart G) ; aucune comparaison cache/rejeu n'est enregistrée par le harnais, et le champ `numerical_integrity` ne contient qu'un `max_sum_error`.

**Portée.** L'écart numérique entre cache et rejeu en FP32 a été mesuré en J17 (C3 : ≤ 6,2e-05 sur état réel GPU) — le ΔKL publié n'est donc **pas matériellement corrompu** par le cache. Mais le protocole de J18_PRE avait précisément fixé le rejeu intégral comme référence valide et le cache comme hors-sujet d'exactitude ; le rapport `J18.md` §3 et §51/63 affirme « rejeu intégral FP32 comme référence d'exactitude », ce qui est **inexact** au sens du chemin de code exécuté.

**Cause racine.** Le harnais réutilise l'API de scoring existante (causal_prior + suffix_loglikelihood) qui implémente le cache, plutôt que le chemin dédié `replay_scores`.

**Correction requise.** Soit appeler `replay_scores` comme chemin de référence (ce qui change les chiffres), soit réécrire la formulation de `J18_PRE` §1 et `J18.md` §3 pour déclarer le chemin en cache comme la mesure (ce qui modifie le pré-enregistrement — interdit). L'écart ne modifie pas le verdict GO mais affecte la **traçabilité** du protocole.

### Écart B — MAJEUR — C4 « Test des signes par prompt » implémenté sur les marginaux token

**Constat.** `J18_PRE.md` §C4 : « Test des signes **par prompt**, p < 0,05 » avec rappel J16 « 17/20 positifs → p = 0,00258 » — l'exemple est au niveau **prompt**.

**Implémentation.** `run_j18_decision.py:902-904` consomme `statistics["sign_test"]["positive"]`/`negative` où `sign_test` porte sur tous les marginaux (4680). Résultat publié : `{'n': 4680, 'positive': 4680, 'negative': 0, 'p_value': 0.0}`.

**Recalcul au niveau prompt.** 13/13 prompts en direction positive → p exact bilatéral = 2·0.5¹³ = **0,000244**.

**Portée.** Le critère est atteint sous les deux lectures (p = 0,0 et p = 0,000244 sont < 0,05). **Verdict inchangé.** Mais l'évidence publiée (4680/4680 marginaux) ne correspond pas à l'unité gelée (par prompt), et `J18.md` §1 écrit « **p = 0,0** (4680/4680 positifs) » sous l'en-tête C4.

**Correction requise.** Publier une statistique séparée au niveau prompt (`sign_test_prompt` : n=13, positive=13, p=0.000244) et faire consommer le critère par cette statistique ; ou réécrire l'énoncé de C4 dans le rapport pour déclarer l'unité marginale utilisée (ce qui modifie la formulation gelée — à éviter).

### Écart C — MAJEUR — la commande exacte consignée n'est pas rejouable

**Constat.** `lineage.producer.command` = `'python python3.12' --model Qwen/Qwen3-0.6B ...`.

**Vérification.** Exécuter la commande telle quelle échoue : `python3.12: can't open file '…/python3.12': No such file or directory`. Le chemin du script est absent.

**Cause racine.** `_command_line()` (`run_j18_decision.py:825`) construit `parts = [f"python {sys_executable_name()}"]` puis `parts.append(argv[0])` où `argv = sys.argv[1:]` — donc `argv[0]` est le **premier argument utilisateur** (ici `--model`), pas le chemin du script (`sys.argv[0]`). La commande enregistrée est donc une concaténation invalide.

**Portée.** C'est un défaut de **traçabilité** au sens ADR-002 (« commande exacte »). Il se répète par rapport à l'écart C de J15 (qui fut corrigé en J16). Le verdict n'est pas affecté.

**Correction requise.** Corriger `_command_line` pour inclure le chemin du script (`sys.argv[0]` si présent, sinon `Path(__file__).name`), et régénérer le manifeste. Comme le manifest fait partie de `results.json` (sha256 publié), toute régénération modifie l'empreinte — à convenir avec l'architecte.

### Écart D — MAJEUR — le rapport revendique « sans réserve » et publie un effectif erroné

**Constat 1 (réserve).** `J18.md` §1 : « Tous les critères pré-enregistrés C1–C10 sont tenus, **sans réserve** » et « Absence de réserve : contrairement au smoke de validation… le run à 120 blocs n'a aucune queue ni effet porté sur un sous-ensemble. » Or : (i) C4 est mesuré au niveau token pas par prompt (Écart B) ; (ii) C5 est compté sur 120 indices entiers, pas 1560 blocs (ci-dessous) ; (iii) `mean_causal_kl`/`mean_future_kl` ne sont pas publiés (Écart E). Le mot « sans réserve » est trop fort pour un protocole qui présente ces écarts.

**Constat 2 (effectif).** `J18.md` §1 table C5 : « **120 blocs/config** ». L'implémentation (`run_j18_decision.py:750` `for block_index in range(args.blocks)` imbriqué dans `for prompt in prompts`) fait **13 prompts × 120 blocs = 1 560 blocs de base échantillonnés** ; `block_index` redémarre à 0 pour chaque prompt, donc `distinct[config].add(block_index)` ne compte que les 120 entiers distincts. `results.json` publie bien 120.

**Portée.** C5 est atteint sous les deux lectures (120 ≥ 100 et 1 560 ≥ 100). **Verdict inchangé.** Mais le rapport publie un nombre qui sous-évalue l'échantillon de 13× et la réserve « sans réserve » n'est pas étayée.

**Correction requise.** Remplacer « sans réserve » par une liste explicite des réserves (C4 unité token, C5 comptage, absence des moyennes causale/future), et publier le nombre de blocs échantillonnés par configuration (1 560) ou renommer le champ en précisant qu'il compte les entiers `block_index` distincts.

### Écart E — MAJEUR — `mean_causal_kl` / `mean_future_kl` omis du manifeste (non corrigible en J18)

**Constat.** `evaluate_multiprompt` appelle `summarize_marginals(records)` (`diagnostic.py:362-363` calcule `mean_causal_kl` et `mean_future_kl`) mais le dictionnaire retourné (`diagnostic.py`, bloc `return`) ne conserve que `mean_delta_kl`, `median_delta_kl`, l'IC, et les stats par prompt. Le JSON publié n'expose donc que ΔKL.

**Conséquence.** Un réviseur ne peut pas :
- calculer la réduction relative `ΔKL / KL_causal` (métrique de référence de J15, 8,58 %) ;
- juger si ΔKL = 3,047 est une amélioration modeste sur un gros désaccord de base ou une forte amélioration ;
- comparer à J14 (−16,1 %), J15 (−8,58 %) ou le papier (−71,9 %).

**Correction requise.** Ajouter `mean_causal_kl` et `mean_future_kl` aux statistiques de `evaluate_multiprompt` dans `diagnostic.py`. **Bloqué en J18 :** `diagnostic.py` est hors périmètre (correctifs J16) et protégé par C10 (`git diff ... -- diagnostic.py` doit rester vide). Ce correctif appartient à **J19**.

---

## 4. Écarts de méthode

### Écart F — MAJEUR — la suite de tests ne pinne pas la construction future-aware ni l'unité de bootstrap

**Méthode.** Injection de fautes rigoureuse : patch de **toutes** les liaisons de fonction dans `sys.modules` (le harnais, le paquet `academiia.dopd.diagnostic` et le fichier de test importent ces fonctions par nom, donc chaque module garde sa propre liaison). Un plugin `fault_plugin` injecte FAULT=F1..F4 via `pytest_collection_finish`.

| Faute | Signification | Détectée ? |
|---|---|---|
| F1 | `delta_kl` signe inversé | ✅ 1 failed (suite complète) |
| F2 | axe KL inversé (`KL(target‖exact)`) | ✅ 1 failed (`test_dopd_diagnostic.py`) |
| F3 | bootstrap sur les marginaux au lieu des blocs | ❌ **172 passed** — réelle lacune |
| F4 | `future := causal` (cible future-aware supprimée) | ❌ **172 passed** — réelle lacune |

**Interprétation.**
- F4 : le harnais entier peut être remplacé par le cible causal (ΔKL ≡ 0 par construction) et **tous les 172 tests passent**. Rien ne pinne que la future-aware construction soit effectivement future-aware (`delta_kl > 0` n'est jamais asserté ; `future_kl ≈ 0` non plus).
- F3 : l'unité de bootstrap est correcte dans l'implémentation (`group_fields=("prompt","config_index","block_index")` passé explicitement), mais **aucun test** ne l'exerce, donc une régression sur l'unité de résampling passerait inaperçue.
- F1/F2 sont détectés (le signe et l'axe sont pincés par `test_dopd_diagnostic.py`).

**Lacunes supplémentaires de la suite.** Les branches d'échec C4, C5, C6, C7, C8, C9 du tableau de décision ne sont pas testées ; le test C10 (ligne 139) assert `status != "GO"` plutôt que `status == "GO conditionnel"`. Le fichier est aussi **entièrement sauté** sans `torch` via `pytest.importorskip` au niveau module (ligne 13).

**Correction requise.** Ajouter des tests de régression pour : direction de ΔKL (`delta_kl > 0`), axe KL au site d'appel du harnais, bootstrap par blocs vs marginaux, future-aware non-vide (`future_kl < causal_kl`), chaque branche C4–C9, et l'exactitude du statut C10. Séparer les tests purs (binomie, porte de décision) des tests torch pour la collecte sans torch.

### Écart G — MAJEUR — mesure du temps avec deux horloges différentes

**Constat.** `run_j18_decision.py` : ligne 728 `started = time.time()` (horloge système, époche) ; ligne 777 `elapsed = time.monotonic() - started` (horloge monotone, depuis le boot). Résultat dans le log : `−1792748433,9 s` (négatif, hors de toute plage).

**Conséquence.** Le garde-budge `_check_budget` utilise `time.time() - started` (cohérent) : le plafond hard de 4 h a donc été respecté réellement. Mais l'affichage de progression est corrompu (la seule visibilité pendant un run de 7 min), et le `wall_seconds` rapporté (433,85 s) ne correspond pas au « 436,5 s » affirmé dans `J18.md` §3 (qui ne figure ni dans l'artefact versionné (433,85) ni dans la copie `_artefact` (530,588)).

**Correction requise.** Unifier l'horloge (`time.monotonic()` partout ou `time.time()` partout) ; corriger la valeur du temps dans le rapport.

---

## 5. Comparabilité d'échelle (lecture)

**Constat.** J15 (20 prompts, 10 configurations, Qwen3-0.6B) : ΔKL moyen = **0,0802**. J18 (13 prompts, 1 configuration N4_V16, Qwen3-0.6B) : ΔKL moyen = **3,0471** — ratio **38,0×**. De plus, p10 et p18 étaient en direction **négative** dans J15 (−0,1098 / −0,2760) et sont fortement positifs dans J18 (+2,6977 / +2,5624) — inversion de signe pour les mêmes prompts.

**Cause.** La géométrie de J18 (préfixes 12–21, un seul masque à la fois, N=4, V=16, support restreint renormalisé) n'est pas la même mesure que J15 (10 configurations, 4 niveaux σ, états multi-masque du papier). La réduction relative ne peut pas être calculée (pas de `mean_causal_kl`).

**Conséquence.** Le rapport `J18.md` ne compare jamais J18 à J15/J14 ni au papier, et affirme « l'effet future-aware est fort et homogène » sans situer l'amplitude. La décision GO est valide au sens des critères pré-enregistrés, mais sa **portée comparative** n'est pas établie.

---

## 6. Conformité à la gouvernance

| Règle | Verdict | Note |
|---|---|---|
| Règle 9 (écritures dans le dépôt) | **✓ conforme** | Aucun écriture hors dépôt ; modèle dans `_artefact/hf-cache/` |
| Règle 11 (push interdit) | **✓ conforme** | Aucun push, aucun tag |
| Règle 12 (pas de commit de pure consignation) | **✓ conforme** | Les 9 commits sont substantiels ; ce journal est un rapport d'audit séparé |
| Règle hors-scope J18 | **✓ conforme** | Aucun `StudentPolicy`, aucun entraînement/backward/optimizer/checkpoint, aucune échelle > 0,6B, aucun artefact J14–J17 régénéré |
| ADR-001 (paquet pur) | **✓ conforme** | `torch`/`transformers` confinés à `harness/` |
| Non-régression G1 | **✓ conforme** | `git diff ... -- modules/a2d/eval/dream_a2d/` vide ; 266 passed |
| Pré-enregistrement gelé | **✓ conforme** | `J18_PRE.md` inchangé ; critères non modifiés |

---

## 7. Synthèse et verdict de certification

### 7.1 Verdict

**Le verdict GO pour ouvrir un `StudentPolicy` réduit (≤ 0,6B) en J19 est correct au sens des critères C1–C10 pré-enregistrés.** Le noyau numérique est correct, le pré-enregistrement est intact, le budget est tenu, le manifeste ADR-002 est complet, et la CI est verte.

**Toutefois, la communication de J18 est en plusieurs points au-dessus des résultats.** Quatre affirmations versionnées sont factuellement ou traçablement fausses (A, C, D partie 1, G), deux omissions empêchent la vérification indépendante de l'amplitude de l'effet (D partie 2, E), et la suite de tests ne protège pas le cœur scientifique (F). Aucune de ces erreurs ne remet en cause la **décision** GO ; toutes remettent en cause la **traçabilité** et la **reproductibilité** de l'expérience.

### 7.2 Corrections prioritaires

| Priorité | Correction | Fichiers |
|---|---|---|
| **P0** | Corriger `_command_line` pour inclure le chemin du script ; régénérer le manifeste (nouvelle empreinte `results.json`) | `run_j18_decision.py:825-835` |
| **P0** | Remplacer « sans réserve » par les réserves explicites (C4 unité token, C5 comptage 1 560 blocs, absence des moyennes causale/future) ; publier le nombre de blocs échantillonnés par configuration | `J18.md` §1 |
| **P0** | Publier une statistique de signes au niveau prompt (n=13, positive=13, p=0,000244) | `run_j18_decision.py`, `results.json` |
| **P0** | Unifier l'horloge de mesure du temps ; corriger la durée dans le rapport | `run_j18_decision.py:728,777` ; `J18.md` §3 |
| **P0** | Ajouter les tests de régression (F1–F4, branches C4–C9, statut C10 exact) | `test_dopd_harness_j18_decision.py` |
| **P1** | Appeler `replay_scores` comme chemin de référence, ou réécrire l'énoncé de la référence d'exactitude (modifie le pré-enregistrement — à arbitrer) | `run_j18_decision.py` scoring path ou `J18_PRE.md` §1 / `J18.md` §3 |
| **P1** | Corriger `J18.md` « 120 blocs/config » → « 1 560 blocs échantillonnés (120 entiers distincts par configuration) » | `J18.md` §1 |
| **P2** (J19) | Publier `mean_causal_kl` / `mean_future_kl` dans `evaluate_multiprompt` → calculer la réduction relative et comparer à J15/papier | `diagnostic.py` (bloqué C10 en J18) |

### 7.3 Ce qui ne doit pas être touché

- `modules/dopd/dopd/target.py` et `diagnostic.py` — correctifs J16, protégés par C10 ; le correctif du manque des moyennes KL appartient à J19.
- `modules/dopd/docs/reports/J18_PRE.md` — pré-enregistrement figé.
- Les critères C1–C10 — ne pas les modifier (toute correction porte sur le rapport et l'implémentation, pas sur les seuils).
- `modules/a2d/eval/dream_a2d/` et le verdict G1 — intacts.

### 7.4 Pour J19

Avant d'ouvrir un `StudentPolicy`, il est recommandé de :
1. Publier `mean_causal_kl` / `mean_future_kl` et la réduction relative (correctif P2 ci-dessus).
2. Vérifier que le chemin de référence est bien le rejeu intégral sans cache (corriger Écart A ou réécrire le pré-enregistrement).
3. Renforcer la suite de tests du harnais sur la future-aware et le bootstrap (corriger Écart F).
4. Réconcilier l'amplitude ΔKL de J18 (3,047) avec J15 (0,0802) avant toute interprétation comparative.

---

## 8. Preuves

Toutes les vérifications de cet audit sont reproductibles sans GPU.

```bash
# 1. Ascendance du pré-enregistrement
git diff 659e32a7..HEAD -- modules/dopd/docs/reports/J18_PRE.md   # vide

# 2. Sous-ensemble fixé avant le run
git log --oneline --format="%h %s" 18feb51 -1   # Phase 0/1, avant le code producteur

# 3. CI locale
_artefact/venv/bin/python -m ruff check academiia/ modules/ tests/ tools/ infra/   # All checks passed
_artefact/venv/bin/python -m pytest modules/a2d/tests modules/dopd/tests tests \
    infra/vllm/scripts/tests -q --ignore=modules/a2d/tests/test_a2d_mutation.py   # 266 passed

# 4. Recalcul des chiffres de tête
_artefact/venv/bin/python - <<'PY'
import json
d = json.load(open("modules/dopd/eval/j18_decision/results.json"))
s = d["statistics"]
print("mean_delta_kl", s["mean_delta_kl"])
print("CI95", s["aggregate_ci95_delta_kl"])
print("median_delta_kl", s["median_delta_kl"])
print("prompts_significant", s["n_prompts_ci_excludes_zero"], "/", s["n_prompts"])
print("sign_test", s["sign_test"])
PY

# 5. Injection de fautes (reproductible, CPU uniquement)
export PYTHONPATH="$PWD/_artefact"
FAULT=F4 python -m pytest modules/dopd/tests -q -p fault_plugin -p no:cacheprovider   # 172 passed → faute future-aware non pinée
FAULT=F3 python -m pytest modules/dopd/tests -q -p fault_plugin -p no:cacheprovider   # 172 passed → faute bootstrap non pinée

# 6. Commande non rejouable
python3.12 python3.12 --help   # can't open file '.../python3.12'

# 7. Horloge mélangée
grep -n "started = time.time()\|time.monotonic() - started" modules/dopd/harness/run_j18_decision.py   # lignes 728 / 777
```

**Environnement de l'audit** : `_artefact/venv/` sous le dépôt, conforme à la règle 9.

---

*Fin de l'audit J18. Aucun fichier de mission n'a été modifié par cet audit ; seul ce journal d'audit est ajouté (règle 12 : les désaccords avec un audit vont dans un journal d'audit séparé, pas dans le rapport de mission).*
