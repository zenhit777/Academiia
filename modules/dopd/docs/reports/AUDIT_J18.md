# Audit de la mission J18 — décision à budget borné (d-OPD)

**Date** : 2026-10-26, **corrigé le 2026-10-27** · **HEAD audité** : `10c9d31ff8b3e80765393b4348ebc0ff81c5b2ba` · **HEAD de départ J18** : `659e32a7d22f5d570c5ce43b4ae068e2aebbab2b`
**Auditeur** : agent de codage (OhMyPi) · **Périmètre** : mission J18, Phases 0–4 · **Mode** : lecture des artefacts versionnés, recalculs ciblés sur `results.json`, exécution de la CI locale, **aucun entraînement** (inférence CPU pour la vérification de l'écart H).

> **⚠ Correction du 2026-10-27.** Une revue d'opposition indépendante a mis en évidence un écart **plus grave** que ceux initialement rapportés : la cible future-aware mesurée par J18 est **identique à la posterior exacte** (écart H), ce qui fait de ΔKL une **identité algébrique** et non un résultat expérimental. Le verdict initial de cet audit (« GO correct ») était **trop favorable** : il jugeait la *conformité de protocole* sans voir que la *grandeur mesurée* était dégénérée. Le verdict est corrigé en §7.1. Les constats initiaux (A–E) sont conservés : ils restent exacts, mais leur portée était sous-estimée. Onze constats supplémentaires ont été ajoutés (F–P) : **1 critique** (H), **7 majeurs** (F, G, I, J, K, O, P) et **3 mineurs** (L, M, N). Bilan : **16 constats** — 2 critiques, 11 majeurs, 3 mineurs.

> **Source primaire.** arXiv:2609.35362v1 (Liu, Hu, Han, *d-OPD: Future-Aware On-Policy Distillation for Block Diffusion Language Models*, 28 sept. 2026) : <https://arxiv.org/html/2609.35362v1>. Les affirmations « papier » ci-dessous sont vérifiées sur le HTML arXiv, section par section.

> **Résumé.** Le protocole d'exécution de J18 est conforme sur les points de gouvernance : pré-enregistrement intact, sous-ensemble de prompts fixé et hashé avant tout run GPU, budget tenu, CI verte (266 passed), aucun entraînement, aucun push, aucun artefact J14–J17 régénéré. **Mais la grandeur mesurée est dégénérée.** Dans la géométrie J18 (un seul masque, toutes les autres positions du bloc visibles), la cible future-aware coïncide avec la posterior exacte énumérée à 4,5e-10 près : `future_kl ≡ 0` et `ΔKL ≡ KL(exact‖causal) ≥ 0` par construction (écart **H**, vérifié sur le vrai Qwen3-0.6B). Les critères C1–C4 sont donc des **identités**, et « 4680/4680 marginaux positifs, aucune queue » en est une conséquence structurelle. **Le GO n'est pas transférable à J19.** Quatre autres écarts **majeurs** s'ajoutent : (F) le support est construit sur un **mélange** pondéré et non par préfixe (couverture 91,58 % du prior réel, chevauchement 11/16) ; (G) l'arithmétique est **float64** alors que le manifeste déclare `float32` ; (I) le manifeste porte `correctness_gate: true` sans aucun vérificateur ; (J) les blocs sont tirés **uniformément** au lieu d'être pondérés par `log_joint`, ce qui exacerbe l'écart d'échelle contre J15. Les écarts initiaux A–G (rejeu/cache non exécuté, C4 par token, commande non rejouable, « sans réserve », moyennes KL absentes, tests non sensibles, horloges mélangées) restent **exacts**.

---

## 0. Écart H — CRITIQUE — la cible future-aware est la posterior exacte : ΔKL est une identité

**Constat.** Dans `_marginal_for_position` (`run_j18_decision.py:630-713`), l'état visible est construit comme :

```python
visible = {p: int(block[p]) for p in range(block_size) if p != position}   # :647
exact = exact_token_posteriors(sequences, log_joint, visible, ...)[position] # :648-652
```

c'est-à-dire **toutes** les positions du bloc sauf la cible. Or la cible future-aware (§ eq. 15–18) est construite sur le **même** préfixe et le **même** suffixe :

```python
prefix = prompt_ids + block[:position]        # :653
suffix = block[position + 1 :]                # :654
scores = scorer.suffix_loglikelihood(prefix, tuple(score_set), suffix)  # :670
weights = causal * np.exp(shifted)            # :677-678
future  = weights / weight_total              # :682
```

**Preuve algébrique.** Le joint énuméré factorise sur le support restreint :
$p_T(t_0\mid q)\prod_{j>0} p_T(t_j\mid q,t_{<j})$. La posterior exacte sommant sur les séquences compatibles avec l'état visible (donc conditionnant sur le suffixe $t_{>i}$) donne

$$\pi_{\mathrm{exact}}(v) \;\propto\; p_T(v\mid q,t_{<i})\cdot p_T(t_{>i}\mid q,t_{<i},v)$$

tandis que la cible future-aware vaut

$$p_{\mathrm{future}}(v) \;\propto\; p_T(v\mid q,t_{<i})\cdot\exp\!\big(s_i(v)-s_i(\hat x_i)\big),\qquad s_i(v)=\log p_T(t_{>i}\mid q,t_{<i},v)$$

Les deux expressions sont **identiques** à la constante multiplicative $\exp(-s_i(\hat x_i))$ près, qui s'annule dans la normalisation. Donc $p_{\mathrm{future}}\equiv\pi_{\mathrm{exact}}$, d'où

$${\tt future\_kl}\equiv 0,\qquad \Delta KL \equiv KL(\pi_{\mathrm{exact}}\| r_{\mathrm{causal}})\ \ge 0 .$$

**Preuve empirique (indépendante).** Reproduction avec le **vrai code du harnais** (`_build_restricted_support` + `exact_token_posteriors` + `kl_exact_to_target`) sur le **vrai Qwen3-0.6B** (CPU, inférence seule, 2 prompts, 36 marginaux) :

| Convention | `future_kl` moyen | `max\|ΔKL − KL_causal\|` | `frac(ΔKL>0)` | ΔKL moyen |
|---|---:|---:|---:|---:|
| **Géométrie J18** (visible = toutes positions sauf la cible) | **5,56e-11** | **4,49e-10** | **1,0000** | **+3,292** |
| Variante paper-compatible (futur **marginalisé**) | 4,485 | 17,19 | 0,1944 | −3,490 |

La même reproduction sur un modèle autoregressif jouet (`_artefact/verify_ft1.py`) donne `future_kl` = 6,6e-15 en géométrie J18 contre 3,2e-01 en variante paper-compatible. Le ΔKL moyen mesuré (+3,292) est cohérent avec le ΔKL publié (+3,047) : **ce que J18 mesure, c'est `KL(exact‖causal)`, pas un « effet future-aware ».**

**Cause racine.** J15 (`diagnostic_annexe_d.py:575-589`, `:1010-1025`) utilise **deux blocs distincts** : `block` définit l'état visible (et donc la posterior exacte), tandis que `future_block` est échantillonné **séparément** (`sample_compatible_index(perturbed_log, compatible, ...)`) et sert à construire la cible future-aware. J18 utilise **le même bloc pour les deux**, ce qui supprime toute incertitude sur le futur et rend la correction triviale.

**Portée.** Les critères C1 (ΔKL > 0), C2 (médiane > 0), C3 (proportion) et C4 (test des signes) sont des **conséquences algébriques**, pas des observations. Les affirmations de `J18.md:33` (« chaque marginal est positif ») et `J18.md:37` (« aucune queue ni effet porté sur un sous-ensemble ») décrivent la géométrie, pas l'effet de la correction future-aware. L'hypothèse qui autorise l'ouverture d'un `StudentPolicy` en J19 — « l'effet future-aware est fort et homogène » — **n'est pas testée** par cette expérience.

**Correction requise.** Reconstruire `_marginal_for_position` sur la convention J15 : **un futur compatible échantillonné séparément**, avec la posterior exacte conditionnée sur l'état visible qui n'inclut pas ce futur. Réviser ensuite `J18.md` et re-lancer la mesure.

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
8. **Contre-test de la validité de la grandeur** (2026-10-27) : reproduction de `_marginal_for_position` avec le **vrai code du harnais** sur un modèle autoregressif jouet puis sur le **vrai Qwen3-0.6B**, en comparant la géométrie J18 à une variante paper-compatible (futur marginalisé). C'est ce contre-test qui a établi l'écart H (§0) : le verdict initial, fondé sur la seule conformité de protocole, était insuffisant.

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

Ces points ont été vérifiés indépendamment et sont conformes. Ils portent sur le **protocole d'exécution et la gouvernance** — pas sur la validité de la grandeur mesurée, qui est traitée en §0 (écart H).

| # | Point | Preuve |
|---|---|---|
| P1 | **Pré-enregistrement intègre** | `git diff 659e32a7..HEAD -- modules/dopd/docs/reports/J18_PRE.md` → vide ; acceptation Phase 0 consignée dans `18feb51` |
| P2 | **Sous-ensemble fixé avant le run** | `prompts.json` commité dans `18feb51` (Phase 0/1), hash `593c32e5…b203f35` dans `lineage.data.prompt_subset_hash` |
| P3 | **Budget tenu** | wall_seconds `433,85 s` ≤ 14 400 s ; `cuda_allocated_peak_bytes` = 2876,5 MiB ≤ 16 GiB ; 13 prompts ≤ 50 ; ≤ 32 candidats + référence = 33 branches |
| P4 | **Modèle lu dans le dépôt** | `model_source_in_repo = true`, snapshot présent sous `_artefact/hf-cache/` (règle 9) — mais le garde-fou s'exécute après le chargement (écart L) |
| P5 | **Professeur gelé** | `_teacher_is_frozen(scorer)` = true : `requires_grad=False`, `model.training=False`, aucun `backward`/optimizer |
| P6 | **C9 provenance complète** | empreintes de `model.safetensors` dans `lineage.parent.hashes` (l'empreinte de l'artefact produit manque : écart K) |
| P7 | **Intégrité numérique** | `max_sum_error` = 4,44e-16 ; aucune NaN/Inf |
| P8 | **C10 non-régression** | `git diff 659e32a7..HEAD -- target.py diagnostic.py dream_a2d/` est vide au moment du run ; commit producteur `30594e3` == blob exécuté |
| P9 | **C1–C3 : le calcul est correct, mais il est **dégénéré** | ΔKL = 3,047, CI 95 % [2,968 ; 3,127], médiane 1,961, 13/13 prompts significatifs — **tenus par construction** (écart H), non par observation |
| P10 | **C6, C8 tenus ; C7 surestimé** | intégrité numérique et professeur gelé ; C7 est infalsifiable et l'empreinte d'artefact manque (écart K) |
| P11 | **CI verte** | 266 passed ; ruff All checks passed |
| P12 | **Sélection non triée** | le sous-ensemble inclut `p10` et `p18`, les deux prompts à ΔKL moyen **négatif** en J15 — aucune sélection orientée par le résultat |

---

## 3. Écarts majeurs

### Écart A — CRITIQUE — « rejeu intégral FP32 sans cache » est le chemin de référence déclaré, pas le chemin utilisé

**Constat.** `J18_PRE.md` §1 (table, ligne « Référence d'exactitude ») prescrit :

> **Rejeu intégral FP32** (contexte complet rejoué par branche, **sans cache**)

et le même paragraphe justifie :

> Le cache est plus lent que le rejeu aux préfixes courts : mesurer l'effet **sans** le biais du cache

La mission Phase 2 confirme : « Le cache de préfixe peut être instrumenté pour diagnostic de latence, mais n'est pas la référence d'exactitude. »

**Implémentation.** `run_j18_decision.py` appelle `scorer.causal_prior(prefix)` puis `scorer.suffix_loglikelihood(prefix, candidates, suffix)` — qui est le **chemin en cache** (préremplissage prefix + réutilisation du KV-cache entre branches, `teacher_scorer_qwen3.py:279-323`). `grep replay run_j18_decision.py` → **aucune occurrence**. Le chemin de référence complet `replay_scores`/`replay_job` (`teacher_scorer_qwen3.py:348-423`, `use_cache=False`) n'est jamais appelé.

**Preuve empirique.** Le log du run (`_artefact/j18_run.log`) affiche une durée de progression négative (voir Écart P) ; aucune comparaison cache/rejeu n'est enregistrée par le harnais, et le champ `numerical_integrity` ne contient qu'un `max_sum_error`.

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

### Écart F — MAJEUR — le support est construit sur un mélange pondéré, pas par préfixe

**Constat.** `_build_restricted_support` (`run_j18_decision.py:228-252`) choisit, à chaque position, les `vocab_width` (16) meilleurs tokens du **mélange** pondéré par la probabilité des préfixes restreints. L'Annexe D / `DOPD_SPEC.md:636-637` et `diagnostic.py:157-170` (`RestrictedSupport.causal_prior`) construisent le support **par préfixe**.

**Portée.** La « cible causale » comparée n'est donc pas $r_i(v)$ de l'équation 11, et le support n'est pas la construction de l'Annexe D. Mesuré : couverture du prior réel **91,58 %** (8,42 % écartés), chevauchement prior-top-16 / mélange-top-16 = **11/16**.

**Correction requise.** Construire le support par préfixe, ou documenter explicitement la variante mélange dans l'artefact et le rapport.

### Écart G — MAJEUR — l'arithmétique est float64, le manifeste déclare float32

**Constat.** `run_j18_decision.py:675` (`np.array(..., dtype=np.float64)`), `:677-682` (exp, poids, normaliseur), `diagnostic.py:281-283` (`np.asarray(..., dtype=np.float64)`) : les deltas, les facteurs `exp(Δs)`, les poids et la KL sont en **float64**. L'arithmétique *de score* est bien float32 côté torch (`teacher_scorer_qwen3.py:482`, `:638` `.float()`).

**Portée.** Aucun bug numérique (float64 a une plage et une précision strictement plus grandes). Mais la **déclaration** est fausse : `run_j18_decision.py:953` (`arithmetic_dtype="float32"`), `results.json` (`"arithmetic_dtype": "float32"`) et `J18.md:62-63` (« arithmétique des scores et normalisation en float32 »). `DOPD_SPEC.md:282-284` impose float32 pour les différences et la normalisation.

**Correction requise.** Soit aligner la déclaration sur le code (`float64`), soit aligner le code sur la spec (float32).

### Écart I — MAJEUR — le manifeste porte `correctness_gate: true` sans aucun vérificateur

**Constat.** `run_j18_decision.py:947` injecte `DOPDConfig()` par défaut dans `lineage.config` ; `results.json` porte donc `"correctness_gate": true`. Or J18 n'implémente **aucun** `TaskVerifier`, aucun $R(\hat x)$, et applique la correction à tous les blocs échantillonnés. `DOPD_SPEC.md:593-594` impose : « Sans vérificateur, la porte doit être désactivée et le fallback causal documenté. »

**Portée.** Le manifeste décrit une configuration de **production** jamais utilisée par l'expérience.

**Correction requise.** Forcer `correctness_gate=false` (ou le champ correspondant) et le documenter.

### Écart J — MAJEUR — les blocs sont tirés uniformément, pas pondérés par `log_joint`

**Constat.** `run_j18_decision.py:751` tire `idx = int(rng.integers(0, sequences.shape[0]))` — **uniforme** sur les 65 536 séquences énumérées. L'implémentation établie de J15 (`diagnostic_annexe_d.py:356-362` → `sample_compatible_index(log_joint, compatible, rng)`, `diagnostic.py:300-319`) tire **proportionnel à la probabilité jointe du professeur**.

**Portée.** Quantifié sur un joint de teacher concentrée : **50 % de la masse** tient dans 0,18 % du support, **90 %** dans 1,91 %, **99 %** dans 9,11 %. Un tirage uniforme n'atteint l'ensemble portant 90 % de la masse qu'avec probabilité **1,91 %**, contre **90 %** pour le tirage pondéré. Sous tirage uniforme, le bloc observé est quasi toujours atypique : le prior causal y est très médiocre et la preuve future très forte. C'est l'explication la plus plausible de l'amplitude ΔKL ≈ 3,05 (≈ 7 × le KL causal du papier, 0,4444) et du facteur **38 ×** contre J15 (0,0802). À noter : `_generate_block` (décodeur glouton) est du **code mort**, alors que la docstring annonce un bloc « généré greedily ».

**Correction requise.** Utiliser `sample_compatible_index(log_joint, compatible, rng)`, comme J15 ; et supprimer ou réécrire la docstring.

---

## 4. Écarts de méthode

### Écart K — MAJEUR — C7 est infalsifiable et l'empreinte de l'artefact est absente du manifeste

**Constat.** `run_j18_decision.py:932` passe `manifest_present=True` en littéral ; `:345` en déduit `c7 = manifest_present`. Le critère C7 ne peut donc pas échouer. `:941` construit `ArtifactRef(..., hashes={})` : l'artefact produit n'est **jamais** hashé dans son propre manifeste (le digest n'est calculé qu'après écriture, `:1050-1052`, et affiché sur stdout). `LineageWriter.write` lève `NotImplementedError` (`:446`) : aucun `lineage.json` séparé, alors qu'ADR-002 prescrit « écrit **à côté** de l'artefact (`lineage.json`) ».

Vérifié : `lineage.artifact.hashes == {}`. Le digest `4e891a3d…fa225e` n'existe que dans `J18.md:116` (où il est exact).

**Portée.** `J18.md:44` affirme « sha256 consigné **dans le manifeste ADR-002** » et `J18.md:25` porte C7 « TENU » : les deux sont inexacts.

**Correction requise.** Renseigner `artifact.hashes` (auto-empreinte, ou empreinte d'une passe précédente documentée) et rendre C7 vérifiable.

### Écart L — MINEUR — le garde-fou « modèle hors dépôt » s'exécute après le chargement

**Constat.** `run_j18_decision.py:864` appelle `load_teacher(...)` ; le garde-fou `_warn_if_cache_is_outside_repo` n'est appelé qu'en `:868`, après `_cache_environment(source)` en `:867`. Le message d'erreur (`:513-521`) affirme pourtant « Le run est refusé **avant tout chargement** ». `teacher_scorer_qwen3.py:142-150` explore `Path.home()/.cache/huggingface/hub`, donc un snapshot hors dépôt **est** atteignable.

**Portée.** Aucune violation sur ce run (`model_source` in-repo), mais l'enforcement annoncé n'existe pas : c'est un défaut d'enforcement, pas un incident.

**Correction requise.** Résoudre la source et vérifier l'appartenance au dépôt **avant** `load_teacher`.

### Écart M — MINEUR — `protocol.torch_version` contient la version *transformers*

**Constat.** `run_j18_decision.py:1001` : `"torch_version": transformers.__version__` → `"5.10.2"`. Le torch installé est `2.12.0+cu130`. `DOPD_SPEC.md:723` exige la traçabilité « versions PyTorch/Transformers ».

**Correction requise.** Consigner les deux versions sous des clés distinctes.

### Écart N — MINEUR — trois champs `training.*` sont des littéraux

**Constat.** `run_j18_decision.py:973-975` : `training.performed`, `training.backward`, `training.weights_updated` sont des `False` littéraux (valeurs vraies, mais indiscernables d'une mesure dans le JSON). `training.requires_grad_all_false` (`:976-977`) est, lui, **mesuré** via `_teacher_is_frozen(scorer)`.

**Correction requise.** Documenter la distinction, ou dériver les trois champs de l'absence effective de code d'entraînement.

### Écart O — MAJEUR — la suite de tests ne pinne pas la construction future-aware ni l'unité de bootstrap

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

### Écart P — MAJEUR — mesure du temps avec deux horloges différentes

**Constat.** `run_j18_decision.py` : ligne 728 `started = time.time()` (horloge système, époche) ; ligne 777 `elapsed = time.monotonic() - started` (horloge monotone, depuis le boot). Résultat dans le log : `−1792748433,9 s` (négatif, hors de toute plage).

**Conséquence.** Le garde-budge `_check_budget` utilise `time.time() - started` (cohérent) : le plafond hard de 4 h a donc été respecté réellement. Mais l'affichage de progression est corrompu (la seule visibilité pendant un run de 7 min), et le `wall_seconds` rapporté (433,85 s) ne correspond pas au « 436,5 s » affirmé dans `J18.md` §3 (qui ne figure ni dans l'artefact versionné (433,85) ni dans la copie `_artefact` (530,588)).

**Correction requise.** Unifier l'horloge (`time.monotonic()` partout ou `time.time()` partout) ; corriger la valeur du temps dans le rapport.

---

## 5. Comparabilité d'échelle (lecture)

**Constat.** J15 (20 prompts, 10 configurations, Qwen3-0.6B) : ΔKL moyen = **0,0802**. J18 (13 prompts, 1 configuration N4_V16, Qwen3-0.6B) : ΔKL moyen = **3,0471** — ratio **38,0×**. De plus, p10 et p18 étaient en direction **négative** dans J15 (−0,1098 / −0,2760) et sont fortement positifs dans J18 (+2,6977 / +2,5624) — inversion de signe pour les mêmes prompts.

**Cause.** La géométrie de J18 (préfixes 12–21, un seul masque à la fois, N=4, V=16, support restreint renormalisé) n'est pas la même mesure que J15 (10 configurations, 4 niveaux σ, états multi-masque du papier). La réduction relative ne peut pas être calculée (pas de `mean_causal_kl`).

**Conséquence.** Le rapport `J18.md` ne compare jamais J18 à J15/J14 ni au papier, et affirme « l'effet future-aware est fort et homogène » sans situer l'amplitude. La **portée comparative** n'est pas établie, et l'écart H interdit d'attribuer cette amplitude à la correction future-aware (ΔKL y est, par construction, le KL causal).

---

## 6. Conformité à la gouvernance

| Règle | Verdict | Note |
|---|---|---|
| Règle 9 (écritures dans le dépôt) | **⚠ réserve** | Aucune écriture hors dépôt ; modèle dans `_artefact/hf-cache/`. Mais le garde-fou annoncé « refusé avant tout chargement » s'exécute **après** `load_teacher` (écart L) |
| Règle 11 (push interdit) | **✓ conforme** | Aucun push, aucun tag |
| Règle 12 (pas de commit de pure consignation) | **✓ conforme** | Les 9 commits sont substantiels ; ce journal est un rapport d'audit séparé |
| Règle hors-scope J18 | **✓ conforme** | Aucun `StudentPolicy`, aucun entraînement/backward/optimizer/checkpoint, aucune échelle > 0,6B, aucun artefact J14–J17 régénéré |
| ADR-001 (paquet pur) | **✓ conforme** | `torch`/`transformers` confinés à `harness/` |
| Non-régression G1 | **✓ conforme** | `git diff ... -- modules/a2d/eval/dream_a2d/` vide ; 266 passed |
| Pré-enregistrement gelé | **✓ conforme** | `J18_PRE.md` inchangé ; critères non modifiés |
| ADR-002 (manifeste complet) | **✗ non conforme** | `artifact.hashes` vide, C7 infalsifiable, `correctness_gate: true` sans vérificateur, `arithmetic_dtype` faux (écarts J, K, M) |

---

## 7. Synthèse et verdict de certification

### 7.1 Verdict (corrigé le 2026-10-27)

**Le GO de J18 n'est pas certifié.** L'expérience est **conforme au protocole d'exécution** mais **non valide scientifiquement** : la grandeur mesurée est une identité algébrique (écart H), non un effet experimental.

**Pourquoi.** Dans la géométrie J18, la cible future-aware coïncide avec la posterior exacte énumérée (écart mesuré : 4,49e-10 sur le vrai Qwen3-0.6B). Il s'ensuit que :

- `future_kl ≡ 0` et `ΔKL ≡ KL(exact‖causal) ≥ 0` — le signe de ΔKL, sa médiane, sa proportion et son test des signes sont des **conséquences de la construction**, pas des observations ;
- C1, C2, C3 et C4 sont **tenus par construction** : les signaler « TENU » ne dit rien du bien-fondé de la correction future-aware ;
- « chaque marginal est positif (4680/4680) » et « aucune queue » sont des propriétés de la géométrie, pas des propriétés de l'effet ;
- l'amplitude ΔKL ≈ 3,05 (≈ 7 × le KL causal du papier) mesure le désaccord entre posterior exacte et prior causal **sur un bloc atypique** (écart J : tirage uniforme au lieu de pondéré par `log_joint`), et non un gain apporté par la cible future-aware.

**Ce qui reste certifié conforme.** Pré-enregistrement intact ; sous-ensemble de prompts fixé et hashé avant le run GPU ; budget tenu (433,85 s, 2 876,5 MiB, 13 prompts) ; CI verte (266 passed, ruff vert) ; aucun entraînement, aucun push, aucun artefact J14–J17 régénéré ; périmètre strict respecté ; `target.py` / `diagnostic.py` / résultats A2D intacts.

**Ce qui doit être tranché par l'architecte.** L'écart H impose de **reconstruire la mesure** avant toute ouverture de `StudentPolicy` : J19 ne doit pas s'appuyer sur le GO de J18. Deux voies possibles :

1. **Reconstruire `_marginal_for_position`** sur la convention J15 (futur compatible échantillonné **séparément**, posterior exacte conditionnée sur un état visible qui n'inclut pas ce futur), corriger le tirage pondéré (J), le support par préfixe (F) et le chemin de référence (A), puis **re-lancer** la mesure et rééditer `J18.md` et l'artefact.
2. **Requalifier J18** en expérience de *diagnostic de la cible* (mesure du KL causal vs exact sur géométrie d'entraînement), ce qui est un résultat utile mais **distinct** de l'effet future-aware — et alors renommer le critère primaire et la décision.

**Corrections de communication indépendantes de la voie retenue.** Cinq affirmations versionnées sont factuellement fausses ou non traçables (A, C, D partie 1, P, plus K), deux omissions empêchent la vérification indépendante de l'amplitude (D partie 2, E), et la suite de tests ne protège pas le cœur scientifique (O).

**Sur le plan de la méthode, ce qui fut fait.** L'audit initial (commit `d8cbcd0`) avait conclu « GO correct » : il avait vérifié la **conformité** du protocole et la **traçabilité** du manifeste, sans mettre en cause la **validité de la grandeur**. La revue d'opposition a fourni le contre-test manquant ; l'écart H a été ensuite **reproduit indépendamment** (modèle jouet puis Qwen3-0.6B réel) avant d'être retenu. Les constats initiaux A–E sont maintenus : ils sont exacts, et leur portée était sous-estimée ; les constats F–P ont été ajoutés après la revue d'opposition.

### 7.2 Corrections prioritaires

| Priorité | Correction | Fichiers |
|---|---|---|
| **P0** | **Reconstruire la mesure** : futur échantillonné séparément (convention J15), posterior exacte conditionnée sur un état visible excluant ce futur ; le harnais doit produire un `future_kl` **non nul** | `run_j18_decision.py:630-713` |
| **P0** | Tirer les blocs **pondérés** par `log_joint` (`sample_compatible_index`), comme J15 ; supprimer le code mort `_generate_block` et corriger la docstring — écart J | `run_j18_decision.py:751`, `:604-628`, `:11-12` |
| **P0** | Si J19 s'appuie sur J18 : ne pas transférer le GO — re-lancer la mesure reconstruite et re-éditer `J18.md` + l'artefact (nouvelle empreinte `results.json`) | `J18.md`, `results.json` |
| **P0** | Corriger `_command_line` pour inclure le chemin du script — écart C | `run_j18_decision.py:825-835` |
| **P0** | Remplacer « sans réserve » par les réserves explicites (C4 unité token — écart B, C5 comptage 1 560 blocs, moyennes causale/future absentes — écart E, écart H) | `J18.md` §1 |
| **P0** | Publier une statistique de signes au niveau prompt (n=13, positive=13, p=0,000244) — écart B | `run_j18_decision.py`, `results.json` |
| **P0** | Renseigner `artifact.hashes` et rendre C7 vérifiable ; forcer `correctness_gate=false` ; corriger `arithmetic_dtype` — écarts G, I, K | `run_j18_decision.py:932,941,947,953` |
| **P0** | Unifier l'horloge de mesure du temps ; corriger la durée dans le rapport — écart P | `run_j18_decision.py:728,777` ; `J18.md` §3 |
| **P0** | Ajouter les tests de régression (F1–F4, branches C4–C9, statut C10 exact) **et un test d'identité `future_kl ≫ 0`** qui aurait détecté l'écart H — écart O | `test_dopd_harness_j18_decision.py` |
| **P1** | Construire le support par préfixe (ou documenter la variante mélange) — écart F | `run_j18_decision.py:228-252` |
| **P1** | Appeler `replay_scores` comme chemin de référence, ou réécrire l'énoncé de la référence d'exactitude (modifie le pré-enregistrement — à arbitrer) — écart A | `run_j18_decision.py` scoring path ou `J18_PRE.md` §1 / `J18.md` §3 |
| **P1** | Corriger `J18.md` « 120 blocs/config » → « 1 560 blocs échantillonnés (120 entiers distincts par configuration) » — écart D | `J18.md` §1 |
| **P1** | Résoudre la source du modèle et vérifier l'appartenance au dépôt **avant** `load_teacher` — écart L | `run_j18_decision.py:864-868` |
| **P2** | Consigner `torch_version` et `transformers_version` séparément — écart M | `run_j18_decision.py:1001` |
| **P2** | Dériver les trois champs `training.*` de l'absence effective de code d'entraînement plutôt que les écrire en littéral — écart N | `run_j18_decision.py:970-980` |
| **P2** (J19) | Publier `mean_causal_kl` / `mean_future_kl` dans `evaluate_multiprompt` → calculer la réduction relative et comparer à J15/papier — écart E | `diagnostic.py` (bloqué C10 en J18) |

### 7.3 Ce qui ne doit pas être touché

- `modules/dopd/dopd/target.py` et `diagnostic.py` — correctifs J16, protégés par C10 ; le correctif du manque des moyennes KL appartient à J19.
- `modules/dopd/docs/reports/J18_PRE.md` — pré-enregistrement figé.
- Les critères C1–C10 — ne pas les modifier (toute correction porte sur le rapport et l'implémentation, pas sur les seuils).
- `modules/a2d/eval/dream_a2d/` et le verdict G1 — intacts.
- **`results.json` versionné** — son empreinte `4e891a3d…fa225e` est celle citée dans `J18.md` ; toute régénération change l'empreinte et doit être assumée explicitement.

### 7.4 Pour J19

**J19 ne doit pas s'appuyer sur le GO de J18.** L'écart H impose de re-lancer une mesure valide avant toute ouverture de `StudentPolicy` :

1. **Reconstruire la mesure** (§0, P0 de §7.2) : futur compatible échantillonné **séparément** (convention J15), posterior exacte conditionnée sur un état visible qui exclut ce futur ; vérifier que `future_kl` sort **non nul**.
2. **Tirer les blocs pondérés** par `log_joint` (écart J) et construire le support par préfixe (écart F).
3. **Exécuter réellement le chemin de référence** annoncé (écart A) ou réécrire l'énoncé pré-enregistré — arbitrage architecte requis.
4. **Publier `mean_causal_kl` / `mean_future_kl`** et la réduction relative (écart E), puis comparer à J15 (0,0802) et au papier (0,4444 → 0,1249).
5. **Renforcer la suite de tests** (écart O) : en particulier un test qui vérifie `future_kl ≫ 0` et un test qui vérifie que le futur est échantillonné indépendamment de l'état visible — ces deux tests auraient détecté l'écart H.
6. **Rappeler les réserves J15/J16** (médiane nulle, queue nuisible 13,3 %, 3/20 prompts dégradés) dans tout rapport successif, comme l'impose `DOPD_SPEC.md:1136-1139`.

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

# 8. Contre-test de l'écart H (tautologie future ≡ exact)
#    8a. Sur le vrai Qwen3-0.6B (CPU, inférence seule, ~5 min)
export HF_HOME="$PWD/_artefact/hf-cache"
python3.12 _artefact/verify_ft1_real.py
#    => géométrie J18 : future_kl moyen 5,56e-11, max|ΔKL − KL_causal| 4,49e-10, frac(ΔKL>0)=1,0000
#    => variante paper-compatible : future_kl moyen 4,485, frac(ΔKL>0)=0,1944
#    8b. Sur un modèle autoregressif jouet (instantané)
python3.12 _artefact/verify_ft1.py
#    => J18 : future_kl 6,6e-15 ; variante : future_kl 3,2e-01

# 9. Constats I–P (relevés dans le code)
grep -n "rng.integers(0, sequences.shape\[0\])" modules/dopd/harness/run_j18_decision.py   # écart J (tirage uniforme)
grep -n "dtype=np.float64\|arithmetic_dtype" modules/dopd/harness/run_j18_decision.py       # écart G (float64 vs déclaré float32)
grep -n "manifest_present=True\|hashes={}" modules/dopd/harness/run_j18_decision.py         # écart K (C7 infalsifiable, hash absent)
grep -n "load_teacher(\|_warn_if_cache_is_outside_repo" modules/dopd/harness/run_j18_decision.py  # écart L (ordre du garde-fou)
grep -n "torch_version" modules/dopd/harness/run_j18_decision.py                            # écart M (version transformers)
```

**Environnement de l'audit** : `_artefact/venv/` sous le dépôt, conforme à la règle 9. Les scripts de contre-test (`_artefact/verify_ft1.py`, `_artefact/verify_ft1_real.py`) et le plugin d'injection de fautes (`_artefact/fault_plugin.py`) sont **hors versionnement**, comme les artefacts temporaires ; ils sont reproduits par les blocs de commande ci-dessus.

---

*Fin de l'audit J18 (version corrigée le 2026-10-27). Aucun fichier de mission n'a été modifié par cet audit ; seul ce journal d'audit est ajouté (règle 12 : les désaccords avec un audit vont dans un journal d'audit séparé, pas dans le rapport de mission). La correction du 2026-10-27 a été motivée par une revue d opposition indépendante ; l'écart H a été reproduit indépendamment avant d'être retenu.*
