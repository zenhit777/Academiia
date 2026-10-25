# Audit de la mission J15 — diagnostic multi-prompt d-OPD + écart top-k = 16

**Date** : 2026-10-25 · **HEAD audité** : `d6a2a74` · **HEAD de départ J15** : `1966111`
· **Auditeur** : agent de codage (OhMyPi), sur demande de l'utilisateur
· **Périmètre** : mission J15, Phases 0–2 · **Mode** : lecture des artefacts
versionnés, recalculs ciblés sur `results.json`, exécution de la CI locale,
**aucune exécution GPU**.

> **Source primaire.** arXiv:2609.35362v1 (Liu, Hu, Han, *d-OPD: Future-Aware
> On-Policy Distillation for Block Diffusion Language Models*, 28 sept. 2026) :
> <https://arxiv.org/html/2609.35362v1>. Les affirmations « papier » ci-dessous
> sont vérifiées sur le HTML arXiv, section par section.

> **Résumé.** Le cœur scientifique de J15 est **fiable** : la pré-enregistrement
> est intègre, les 61 120 marginaux sont recalculables, l'effet future-aware est
> réel, significatif et stable en direction, et le noyau numérique implémente
> correctement les équations 20–22, 25–26 et 28 du papier. **Cinq écarts
> majeurs** ont cependant été relevés, dont trois sont des **affirmations
> fausses dans les documents versionnés** : (A) le rapport affirme « 20/20
> prompts en direction positive (aucun ne dégrade la KL) » alors que 3 prompts
> sur 20 ont une moyenne négative et que 13,3 % des marginaux à futur visible
> sont dégradés ; (B) la réduction relative annoncée de « −16,1 % » est le
> chiffre de J14 (prompt unique) et vaut en réalité **8,58 %** sur les 20
> prompts ; (C) la « commande exacte » citée dans le rapport **n'existe pas**
> (`--prompts` n'est pas un drapeau du harnais) et diffère de la commande
> réellement consignée dans le manifeste. Deux écarts de méthode : (D) la
> distance de variation totale implémentée est la **moitié** de la TV réelle
> contre la troncature renormalisée, rendant les deux métriques top-k
> publicationnées **algébriquement dépendantes** ; (E) la preuve CI porte sur
> `fe499ff` et non sur le HEAD d'arrivée `d6a2a74`.

---

## 1. Périmètre et méthode

### 1.1 Artefacts audités

| Élément | Rôle |
|---|---|
| `modules/dopd/docs/reports/J15.md` | rapport de mission J15 |
| `modules/dopd/docs/reports/J14.md` | rapport J14 (résidus corrigés) |
| `modules/dopd/docs/DOPD_SPEC.md` §6.9.5, §6.9.7–6.9.9 | spécification, pré-enregistrement, résultats |
| `modules/dopd/dopd/diagnostic.py` | noyau NumPy pur (WRAPPER) |
| `modules/dopd/harness/diagnostic_annexe_d.py` | harnais GPU (inférence seule) |
| `modules/dopd/eval/annexe_d_qwen3_0.6b_multiprompt/results.json` | résultat versionné (17 Mo, 672 973 lignes) |
| `modules/a2d/STATUS.md` | bandeaux J14 / J15 |
| `.github/workflows/ci.yml` | étapes CI reproduites localement |
| arXiv:2609.35362v1 | source primaire |

### 1.2 Méthode

1. **Recalcul indépendant** de chaque chiffre de tête à partir des 61 120
   lignes de `rows` (moyennes, IC, médiane, test des signes, proportions,
   masses de troncature).
2. **Vérification d'ascendance** du pré-enregistrement (`git merge-base
   --is-ancestor`, diffs `4a74023..fe499ff` et `4a74023..d6a2a74`).
3. **Exécution réelle de la CI** dans un environnement reproductible, y compris
   un venv **sans `torch`** pour valider la régression de collecte N3.
4. **Confrontation à la source primaire** (arXiv HTML), section par section.

### 1.3 Environnement d'audit

| Élément | Valeur |
|---|---|
| HEAD audité | `d6a2a74aad8476a38fe036764f9b7d85155f1382` |
| Python | 3.12 |
| Install | `pip install -e . --no-deps` dans `_artefact/venv/` (venv **dans le dépôt**, règle 9) |
| Venv sans torch | `_artefact/venv_notorch/` (numpy + pytest uniquement) |
| GPU | **aucun** — audit entièrement CPU |

---

## 2. Ce qui est correct

Ces points ont été **vérifiés indépendamment** et sont conformes.

| # | Point | Preuve |
|---|---|---|
| P1 | **Pré-enregistrement intègre.** §6.9.7–6.9.8 figés dans `4a74023`, code producteur dans `fe499ff` (ancêtre), résultats ajoutés dans `d6a2a74` **sans retoucher les critères**. | `git merge-base --is-ancestor 4a74023 fe499ff` → vrai ; `git diff 4a74023..fe499ff -- DOPD_SPEC.md` → **vide** ; les 71 lignes ajoutées par `d6a2a74` sont uniquement §6.9.9 (résultats) |
| P2 | **N1 — « sans réserve » supprimé** dans `J14.md` (§2.3, §9), `DOPD_SPEC.md` §6.9.5 et `STATUS.md`. Les 5 réserves sont listées (prompt trivial, amplitude, `N6_V6` non concluant, queue, marginaux sans futur). | grep : 0 occurrence de « sans réserve » hors la ligne décrivant la correction elle-même |
| P3 | **N3 — collecte sans `torch` rétablie.** `pytest.importorskip("torch")` au niveau module dans `test_dopd_harness_annexe_d.py` (l. 23) et `test_dopd_harness_budget.py` (l. 19). | venv **sans torch** : `100 passed, 2 skipped`, **0 erreur** ; collecte seule : 100 tests collectés |
| P4 | **« 210 passed » exact.** | `pytest modules/a2d/tests modules/dopd/tests tests infra/vllm/scripts/tests -q --ignore=…mutation.py` → **210 passed** |
| P5 | **Lint et liens verts.** | `ruff check academiia/ modules/ tests/ tools/ infra/` → **All checks passed!** ; `tools/check_links.py` → **OK, 0 lien cassé** |
| P6 | **Chaque chiffre de `results.json` est reproductible.** | Recalcul indépendant : ΔKL agrégé `+0,080191`, IC `[0,063815 ; 0,096170]`, médiane `0`, proportion `0,75`, top-k `2,9534 %` / `37,3190 %` / `1,4767 %`, 61 120 marginaux, 3 configurations (764 / 806 / 1 660 s) — **tous identiques** au JSON |
| P7 | **Noyau numérique conforme au papier.** Éq. 21 (mélange pondéré par `ω(A)`), éq. 22 (factorisation AR + renormalisation sur support restreint), éq. 25/26 (score de log-vraisemblance du suffixe), éq. 28 (KL `π_exact ‖ cible` sur support restreint) : **corrects**. | `modules/dopd/dopd/diagnostic.py:118-346` ; intégrité numérique `3,33e-16` (erreur d'énumération), `6,66e-16` (somme de distribution) |
| P8 | **Correction appliquée au support restreint complet**, comme l'exige l'App. D.3 du papier (« applies the correction to the full restricted vocabulary rather than introducing the practical top-k approximation »). | `diagnostic_annexe_d.py:332-372` ; `_causal_and_future_targets` corrige `position_support` entier |
| P9 | **Statistiques multi-prompt correctes.** Bootstrap de blocs `(prompt, config_index, block_index)`, IC per-prompt par bootstrap de blocs, test des signes avec approximation normale + correction de continuité. | `diagnostic.py:533-704` ; recoupement manuel du p par binomiale exacte : `p ≈ 0` |
| P10 | **G1 intact, `target.py` intact, aucun import interdit.** | `git diff 1966111..HEAD -- modules/a2d/eval/dream_a2d/` → **vide** ; `git log 5351bd4..HEAD -- modules/dopd/dopd/target.py` → **vide** ; grep `torch|transformers|accelerate|peft|deepspeed` dans `modules/dopd/dopd/` → **aucun** |
| P11 | **Traçabilité du producteur.** `_assert_producer_committed()` compare le blob `HEAD:` au blob de l'arbre de travail et **refuse** de produire un résultat non versionné. Les empreintes consignées correspondent aux blobs de `fe499ff`. | `diagnostic_annexe_d.py:387-433` ; `source_sha256` = `62e963…` / `9cedd7…`, recalculés identiques |
| P12 | **Aucun entraînement.** `training.performed = false`, `backward = false`, `weights_updated = false`, `requires_grad_all_false = true`. | `results.json` → bloc `training` |

---

## 3. Écarts majeurs

### Écart A — CRITIQUE — « 20/20 prompts en direction positive (aucun ne dégrade la KL) » est faux

**Constat.** `J15.md` §3.3 (l. 168-172) et `DOPD_SPEC.md` §6.9.9 affirment :

> « **Test des signes.** Les 20 prompts sont tous en direction positive (aucun ne
> dégrade la KL), ce qui donne un test des signes hautement significatif
> (p ≈ 0). La cohérence de direction est donc forte. »

et le tableau de décision §6.9.7-4 porte la mesure **« `p ≈ 0` (20/20 prompts
direction positive) »**, ainsi que `J15.md` §4 :

> « cohérent en direction (20/20 prompts, test des signes p ≈ 0) »

**Recalcul indépendant** sur `per_prompt` et `rows` :

| Mesure | Valeur recalculée |
|---|---|
| Prompts à moyenne `ΔKL` **positive** | **17 / 20** |
| Prompts à moyenne `ΔKL` **négative** | **3 / 20** — `p06` (−0,0114), `p10` (−0,1098), `p18` (−0,2760) |
| Prompts significativement **positifs** (IC exclut 0, borne basse > 0) | **14 / 20** |
| Prompts significativement **négatifs** (IC exclut 0, borne haute < 0) | **1 / 20** — `p18`, IC `[-0,4151 ; -0,1381]` |
| Prompts non significatifs | 5 / 20 |
| Marginaux à futur visible **dégradés** par la correction | **4 661 / 35 040 = 13,30 %** |
| `ΔKL` moyen des marginaux dégradés | **−1,1863** (pire cas : **−28,71**) |
| Proportion de prompts significatifs (§6.9.7-3) | 15/20 = 0,75 — **dont 1 dans le mauvais sens** |

**Portée.** Le critère §6.9.7-3 est formellement atteint (l'IC de 15 prompts
exclut zéro), mais **la direction n'est pas documentée** : 1 des 15 « significatifs »
l'est en **sens négatif**. L'argument §4 (« la réserve R1 de J14 — un seul
prompt trivial — est levée ») repose sur la cohérence de direction, qui est
partiellement fausse : 13,3 % des marginaux sur lesquels la correction est
appliquée sont **dégradés**, avec une moyenne de −1,19.

**Cause racine.** L'implémentation est correcte ; c'est la **dérivation** qui
est fausse. `evaluate_multiprompt` (`diagnostic.py:680-690`) compte
`prompts_excluding_zero` par `(lower > 0) OR (upper < 0)` — c'est-à-dire
« IC exclut zéro », sans distinguer le signe. Le rapport en a déduit « 20/20 en
direction positive » alors que le critère ne mesure que l'exclusion de zéro.

**Correction requise.** Remplacer « 20/20 prompts en direction positive (aucun
ne dégrade la KL) » par un énoncé exact : « 17/20 prompts en direction positive,
3/20 en direction négative ; 14 significativement positifs, 1 significativement
négatif (`p18`) ; 13,3 % des marginaux à futur visible dégradés (moyenne
−1,19) ». Le critère §6.9.7-3 doit être annoté « atteint, mais 1 des 15
significatifs est en sens négatif ».

---

### Écart B — MAJEUR — la réduction relative « −16,1 % » est le chiffre de J14, pas de J15

**Constat.** `J15.md` §4 et §7-2, ainsi que `DOPD_SPEC.md` §6.9.9 limite 1,
annoncent :

> « l'amplitude observée (**−16,1 %** vs −71,9 % du papier) reste faible »

**Recalcul indépendant** sur les 61 120 marginaux de J15 :

| Grandeur | J14 (1 prompt) | **J15 (20 prompts)** |
|---|---:|---:|
| KL causale moyenne | 1,007316 | **0,935078** |
| KL future-aware moyenne | 0,844975 | **0,854887** |
| `ΔKL` moyen | +0,162342 | **+0,080191** |
| **Réduction relative** `ΔKL / KL_causale` | **16,12 %** | **8,58 %** |

**Constat.** Le chiffre « 16,1 % » est **exactement** la valeur de J14
(1,007316 → 0,844975). Il a été reconduit à J15 alors que le multi-prompt donne
**8,58 %**, soit **la moitié**. La comparaison au papier (−71,9 %) est donc
affaiblie deux fois : mauvais numérateur, et non-déclaration de la divergence
d'échelle.

**Correction requise.** Remplacer « −16,1 % » par « **−8,6 %** » dans `J15.md`
§4, §7-2 et `DOPD_SPEC.md` §6.9.9, et corriger l'écart avec le papier
(`8,6 %` vs `71,9 %`).

---

### Écart C — MAJEUR — la « commande exacte » citée n'existe pas et diffère de la commande réelle

**Constat.** `J15.md` §3.2 (l. 112-116) titre « **Commande exacte** (consignée
dans le manifeste ADR-002 du résultat versionné) » et affiche :

```bash
HF_HOME="$PWD/_artefact/hf-cache" python modules/dopd/harness/diagnostic_annexe_d.py \
  --model Qwen/Qwen3-0.6B --device cuda --dtype bfloat16 --blocks 100 \
  --prompts modules/dopd/eval/annexe_d_qwen3_0.6b_multiprompt/prompts.json \
  --seed 20261023 --bootstrap-seed 20261024 --local-files-only \
  --output modules/dopd/eval/annexe_d_qwen3_0.6b_multiprompt/results.json
```

**Vérification.** Le harnais définit `--prompt-file` (et `--prompt`), **jamais
`--prompts`** : `grep '"--prompts"' diagnostic_annexe_d.py` → **aucune
occurrence**. La commande citée **échouerait** en `argparse: unrecognized
argument`. De plus, elle **omet `--top-k 16`** — c'est-à-dire la mesure top-k
elle-même — alors que le manifeste la consigne.

**Commande réellement consignée** dans `results.json` → `lineage.producer.command` :

```
python modules/dopd/harness/diagnostic_annexe_d.py --model Qwen/Qwen3-0.6B
--device cuda --dtype bfloat16
--prompt-file modules/dopd/eval/annexe_d_qwen3_0.6b_multiprompt/prompts.json
--blocks 100 --seed 20261023 --bootstrap-seed 20261024 --local-files-only
--output modules/dopd/eval/annexe_d_qwen3_0.6b_multiprompt/results.json
--top-k 16
```

**Divergences** : `--prompts` → `--prompt-file` ; `--top-k 16` absent de la
version citée ; `HF_HOME=…` présent dans la version citée mais **absent** du
manifeste.

**Effet de bord non déclaré.** Le `HF_HOME` cité en redirection n'a **pas** été
appliqué : `results.json` → `protocol.model_source` =
`/home/sjteb/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de…`,
et `_artefact/hf-cache/` **n'existe pas**. La règle 9 (toute écriture dans le
dépôt) a donc été **contournée pendant le run**, même si le modèle était déjà
en cache hors dépôt — ce qui est inventorié dans
`modules/a2d/docs/reports/OUT_OF_WORKSPACE.md` (l. 49, 55).

**Correction requise.** Remplacer le bloc de commande par la commande
**réellement exécutée** (celle du manifeste), et corriger la mention
`HF_HOME` — soit en le supprimant, soit en déclarant que le run a utilisé le
cache hors dépôt déjà inventorié.

---

### Écart D — MAJEUR — la « distance de variation totale » implémentée est la moitié de la TV réelle

**Constat.** `DOPD_SPEC.md` §6.9.8 (pré-enregistré) prescrit :

> « on compare par position **la distribution causale complète à sa troncade aux
> `k = 16` plus grandes probabilités, renormalisée** »

et attend `tv_distance` : « distance de variation totale entre la distribution
complète et **la version tronquée renormalisée** (0 = identique, 1 = disjoint) ».

**Implémentation** (`diagnostic.py:645-652`) :

```python
truncated = np.zeros_like(normalized)
truncated[top_indices] = normalized[top_indices]
discarded_mass = float(1.0 - np.sum(truncated))
tv_distance = float(np.sum(np.abs(normalized - truncated))) / 2.0
```

`truncated` n'est **pas renormalisé**. La TV calculée est donc celle contre la
troncature **non renormalisée**, soit exactement `discarded_mass / 2`.

**Preuve empirique** (400 tirages, distributions de Dirichlet de tailles 20/50/200/1000) :

| Identité | Erreur maximale observée |
|---|---|
| `\|tv_impl − discarded_mass/2\|` | **1,67e-16** → **identité exacte** |
| `\|TV_rénormalisée − discarded_mass\|` | **7,77e-16** → la TV réelle **est** la masse discardée |

**Deux conséquences.**

1. **Les deux métriques publicationnées ne sont pas indépendantes.** Le rapport
   présente « TV moyenne 1,48 % » et « masse discardée moyenne 2,95 % » comme
   deux grandeurs complémentaires ; elles sont **la même information**, avec un
   facteur 2. La « confirmation » par la seconde métrique est circulaire.
2. **Les valeurs publiées sont sous-évaluées d'un facteur 2** par rapport au
   contrat pré-enregistré : la TV réelle contre la troncature renormalisée
   vaut **2,95 %** en moyenne (et non 1,48 %), et **37,32 %** au pire cas (et
   non 37,3 % — mais ce dernier est déjà la masse, donc correct par hasard).

**Point hors périmètre mais à noter.** La mesure est además **hors sujet par
rapport au papier** : l'App. B ne contient aucune étude TV, et la Table 3 du
papier ne rapporte que la **couverture de masse** séparément sous le professeur
et l'élève (99,45 % / 99,44 % à `k = 16`), jamais la TV ni la masse discardée
comme le fait le rapport. La valeur publiée « 2,95 % » est d'ailleurs
**cohérente** avec la couverture papier (≈ 99 % de masse conservée) — ce qui
suggère que la mesure locale est pertinente, mais sa **formulation** (« écart
cible exacte vs top-k », « distance de variation totale ») ne l'est pas.

**Correction requise.** Corriger `truncate_causal_gap` pour renormaliser
`truncated` avant de calculer la TV, **re-publier** les deux métriques (elles
deviendront redondantes, et il faudra en garder une seule), et reformuler
§3.4 / §6.9.9 pour comparer la couverture à la Table 3 du papier plutôt que
d'invoquer une TV que le papier n'utilise pas.

---

### Écart E — MAJEUR — la preuve CI porte sur `fe499ff`, pas sur le HEAD d'arrivée

**Constat.** `J15.md` §6 (« Preuves de CI ») et §5 ligne f affirment :

> « **Run vert sur le HEAD d'arrivée** — exécution locale des étapes `ci.yml`
> sur `fe499ff` (`feat(dopd): harnais Annexe D multi-prompt…`), 2026-10-25 »

et le tableau N2 (§2)claims « **deux commits cohérents seulement** ».

**Vérification.**

| Élément | Constat |
|---|---|
| HEAD d'arrivée réel | **`d6a2a74`** (4 commits depuis `1966111`, pas 2) |
| Commit couvert par la CI | `fe499ff` (3ᵉ commit) |
| Écart | `fe499ff` (00:02) → `d6a2a74` (02:29) : **10 fichiers modifiés**, dont **`modules/dopd/tests/test_dopd_harness_annexe_d.py`**, **`test_dopd_harness_budget.py`**, `modules/a2d/tests/test_a2d_library.py`, `test_a2d_mutation.py`, `test_pipeline.py` — **les fichiers de test modifiés sont précisément ceux qui portent la correction N3** |

**Portée.** C'est un **cercle vicieux** : la preuve CI visait à certifier la
correction N3 (le `importorskip`), mais la CI a été exécutée sur un commit
**antérieur** à l'introduction de cette correction. Le commit `01146ec`
(« corriger les residus J14 — … et importorskip sans torch ») est celui qui
introduit le `importorskip` — il est **postérieur** à la CI citée.

**Ce que la CI a réellement couvert.** La CI citée couvre `fe499ff`, où les
fichiers de test contenaient encore `import torch` en tête de module : la
régression N3 **n'était pas** vérifiable à ce commit. L'audit a donc rejoué la
CI sur le HEAD d'arrivée réel (`d6a2a74`, arbre de travail propre) :

| Étape | Résultat sur `d6a2a74` |
|---|---|
| `ruff check academiia/ modules/ tests/ tools/ infra/` | **All checks passed!** |
| `pytest modules/a2d/tests modules/dopd/tests tests infra/vllm/scripts/tests -q --ignore=…mutation.py` | **210 passed** |
| Collecte **sans torch** (venv dédié) | **100 passed, 2 skipped**, 0 erreur |
| `python tools/check_links.py` | **0 lien cassé** |
| `git diff 1966111..HEAD -- modules/a2d/eval/dream_a2d/` | **vide** |

**Correction requise.** Citer `d6a2a74` comme HEAD d'arrivée et y rattacher la
preuve CI ; corriger la ligne N2 (« deux commits » → quatre) et l'intervalle
`1966111..fe499ff` → `1966111..d6a2a74` dans l'en-tête du rapport.

---

## 4. Écarts mineurs et réserves de lecture

### M1 — La « commande exacte » et l'attribution de commit divergent dans `DOPD_SPEC.md`

`DOPD_SPEC.md` §6.9.9 affirme : « le commit `fe499ff` porte le code producteur
(`diagnostic_annexe_d.py` / `diagnostic.py`) **et le résultat versionné** ».
C'est **faux** : `git show --stat fe499ff` ne contient que 3 fichiers
(`diagnostic.py`, `prompts.json`, `diagnostic_annexe_d.py`) ; le
`results.json` a été commité dans `d6a2a74`. Correction : « `fe499ff` porte le
code producteur ; `d6a2a74` porte le résultat versionné et le rapport ».

### M2 — Le statut de porte est estampé par un script **non versionné**

`results.json` → `multiprompt.gate_j15_status = "GO conditionnel"` (et
`multiprompt.criteria`, `multiprompt.rationale`) est écrit par
`_artefact/j15_stamp_status.py` — script **hors versionnement** (dossier
`_artefact/` ignoré par git), **absent** du harnais et du noyau
(`grep gate_j15_status modules/dopd/harness/ modules/dopd/dopd/` → aucune
occurrence), et **non referenced** dans aucun document versionné. Un lecteur
ne peut donc pas reproduire ce champ : il dépend d'un script ad hoc. Le champ
`gate` (produit par le harnais) ne contient, lui, que `status: "passed"` et
`reserve_required: false` — sans la décision J15.

### M3 — La top-k est mesurée sur la distribution causale, mais le rapport parle de « posterior exact »

`J15.md` §3.4 (l. 176-178) : « distance entre **le posterior exact** et la cible
tronquée à k = 16 ». L'implémentation (`_full_causal_at_position`,
`diagnostic_annexe_d.py:462-479`) calcule la **distribution causale complète**
softmax du dernier logits — pas le posterior exact énuméré. Le pré-enregistrement
§6.9.8 est, lui, correct (« la distribution causale complète à sa troncade »).
Le rapport contredit donc son propre pré-enregistrement.

### M4 — Le §3.4 annonce une « borne 2 048 » non effective

§6.9.8 pré-enregistre « les `N` premiers marginaux vus (borne `2 048`) ». Le
compteur `topk_count` est **réinitialisé par configuration** (`harness:518`), et
chaque appel (prompt, configuration) ne contient qu'environ 1 019 marginaux :
la borne **ne lie jamais**, et les **61 120** marginaux sont tous mesurés. C'est
un **sur-ensemble** du plan pré-enregistré — acceptable, mais l'écart entre le
plan (« 2 048 ») et le réalisé (« 61 120 ») n'est explicité que dans le tableau
de résultats, pas dans la section protocole.

### M5 — La réserve « queue » est correcte mais incomplète

Le rapport note que 42,8 % des marginaux ont `ΔKL = 0` (sans futur visible) et
que la médiane est nulle. C'est **exact** (recalcul : 26 152/61 120 = 42,788 %
au seuil `|ΔKL| < 1e-9`). Décomposition précise : **26 080** marginaux sont
**structurellement** nuls (aucun futur visible — `ΔKL` exactement `0.0`), et
**72** sont des **artefacts numériques** (`|ΔKL| < 1e-9` mais `> 0`, bornés à
9,6e-10) ; le seuil `1e-12` retombe sur les 26 080 seuls. Mais
« moyenne portée par la queue » ne décrit que **la moitié** du problème : la
queue favorable est *et* une queue nuisible de 13,3 % (Écart A). La réserve
formulée est donc **euphémique**.

### M6 — Divergences d'échelle avec le papier non déclarées dans `J15.md`

`DOPD_SPEC.md` §6.9.5 (l. 753) déclare la réduction d'échelle pour J14
(« `Qwen3-0.6B` au lieu de 8B, 3 configurations au [lieu de] … »), mais
`J15.md` §3 ne déclare **ni** l'échelle (8B → 0.6B), **ni** le nombre de
configurations (10 → 3), **ni** le nombre de blocs (400 → 100). Conséquences
non documentées : les KL locales sont **2,10×** plus élevées que celles du
papier (0,9351 vs 0,4444 pour la cible causale), et la comparaison de la
réduction relative entre deux échelles et deux jeux de configurations **non
équivalents** n'a pas de base directe.

### M7 — Le papier ne fournit aucun élément pour étayer le « −71,9 % » comme repère comparable

La Table 4 du papier (`0,4444 → 0,1249`, **71,9 %**) est un **agrégat poolé**
sur 49 948 marginaux, 10 configurations, 400 blocs, 4 niveaux `σ`, avec
Qwen3-8B. Le papier ne publie **ni** ventilation par configuration, **ni**
test des signes, **ni** IC, **ni** médiane, **ni** analyse de queue, **ni**
exemple dégradé. Les quatre statistiques que J15 présente comme « renfort »
sont donc des **ajouts locaux**, sans référent dans le papier — ce qui est
légitime, mais doit être dit : le « −71,9 % vs −16,1 % » n'est pas un écart
« au référent », c'est un écart entre deux protocoles non comparables.

### M8 — Le libellé « test des signes » du critère §6.9.7-4 est mal Employers

Le critère pré-enregistré dit « test des signes **non paramétrique
(approximation normale à deux queues)** ». L'implémentation applique
effectivement cette approximation (`diagnostic.py:595-620`), avec correction de
continuité — correct. Mais le libellé « test des signes » évoque spontanément
un test exact (binomial), qui donnerait `p ≈ 0` également ici. Sans importance
numérique, mais le rapport devrait préciser « approximation normale du test des
signes ».

---

## 5. Verifications bloquantes — état réel

| # | Vérification (mission J15) | Verdict de l'audit | Note |
|---|---|---|---|
| a | Conclusion J14 reformulée « avec réserves de lecture » | **✓ conforme** | P2 |
| b | `pytest.importorskip` ; collecte OK sans torch | **✓ conforme** | P3 — 100 passed / 2 skipped sans torch |
| c | Critères pré-enregistrés avant le run GPU | **✓ conforme** | P1 — ascendance prouvée, diff des critères vide |
| d | Diagnostic multi-prompt exécuté + résultats versionnés | **✓ conforme** | P6 — tous chiffres recalculés |
| e | Mesure exacte vs top-k = 16 documentée | **⚠ conforme en présence, incorrect en fond** | Écart D — métrique = moitié de la TV réelle ; §3.4 vise le « posterior exact » au lieu de la causale (M3) |
| f | CI verte sur le HEAD d'arrivée, correctement citée | **✗ non conforme** | Écart E — CI sur `fe499ff`, arrivée `d6a2a74` ; la CI **rejouée** sur l'arrivée est verte |
| g | Aucun import interdit dans `modules/dopd/dopd/` | **✓ conforme** | P10 |
| h | `git diff 1966111..HEAD -- modules/a2d/eval/dream_a2d/` | **✓ conforme** | vide |
| i | Correctifs numériques `target.py` intacts | **✓ conforme** | P10 |
| j | Liens Markdown | **✓ conforme** | P5 — 0 cassé |

**Bilan des vérifications bloquantes : 9 / 10 conformes.** Seule la lettre
(f) est non conforme en l'état, mais **l'état réel du code est vert** : l'audit
a rejoué la CI sur `d6a2a74` et l'a trouvée verte. Le défaut est dans la
**citation**, pas dans le code.

---

## 6. Conformité à la gouvernance

| Règle | Verdict | Note |
|---|---|---|
| Règle 9 (écritures dans le dépôt) | **⚠ réserve** | Le run GPU a lu le modèle depuis `/home/sjteb/.cache/huggingface/…` et **non** depuis `_artefact/hf-cache` comme la commande citée l'affirme ; le cache est inventorié dans `OUT_OF_WORKSPACE.md` (l. 49, 55) mais la redirection affichée dans `J15.md` §3.2 est inexacte (Écart C) |
| Règle 11 (push interdit par défaut) | **✓ conforme** | Aucun push ; aucun tag créé ou déplacé |
| Règle 12 (pas de commit de pure consignation) | **✓ conforme sur le principe, ✗ sur la description** | Les 4 commits sont substantiels (gel des critères, code, correction J14, résultats+rapport) — aucun ne consigne le hash d'un précédent. **Mais** `J15.md` §2 décrit « **deux commits** » et `{4a74023, fe499ff}`, alors qu'il y en a **4** et que le rapport lui-même est commité dans le 4ᵉ (Écart E) |
| ADR-001 (paquet pur) | **✓ conforme** | P10 |
| Non-régression G1 | **✓ conforme** | P10 |
| Hors scope strict (pas de `TeacherScorer`, pas d'entraînement) | **✓ conforme** | P8, P12 |

---

## 7. Synthèse et recommendations

### 7.1 Verdict

**La porte scientifique est franchie au sens strict des critères pré-enregistrés,
et le « GO conditionnel » pour J16 est la décision correcte.** Le noyau
numérique est conforme au papier, la pré-enregistrement est intègre, et l'effet
future-aware est réel, significatif et stable en direction sur 17 prompts
positifs sur 20.

**En revanche, la communication de J15 est en places notable au-dessus des
résultats.** Trois affirmations versionnées sont factuellement fausses (A, B, C),
une métrique publiée est mal définie (D), et une preuve de CI vise le mauvais
commit (E). Aucune de ces erreurs ne remet en cause la **décision** ; toutes
remettent en cause la **traçabilité**, qui est précisément ce que la mission J15
voulait restaurer.

### 7.2 Corrections prioritaires

| Priorité | Correction | Fichiers |
|---|---|---|
| **P0** | Corriger « 20/20 prompts en direction positive (aucun ne dégrade la KL) » → 17/20 positifs, 3/20 négatifs, 1 significativement négatif (`p18`), 13,3 % de marginaux dégradés | `J15.md` §3.3, §4 ; `DOPD_SPEC.md` §6.9.9 ; `STATUS.md` |
| **P0** | Corriger « −16,1 % » → « −8,6 % » (valeur J15 ; 16,1 % est le chiffre J14) | `J15.md` §4, §7-2 ; `DOPD_SPEC.md` §6.9.9 |
| **P0** | Remplacer la « commande exacte » par la commande réellement exécutée (`--prompt-file`, `--top-k 16`) et corriger la mention `HF_HOME` | `J15.md` §3.2 |
| **P1** | Corriger `truncate_causal_gap` : renormaliser `truncated` avant la TV ; republier les métriques (une seule suffit alors) ; reformuler §3.4 en couverture de masse comparable à la Table 3 du papier | `diagnostic.py:645-652` + tests ; `J15.md` §3.4 ; `DOPD_SPEC.md` §6.9.8/6.9.9 |
| **P1** | Citer `d6a2a74` comme HEAD d'arrivée, y rattacher la preuve CI, corriger « deux commits » → quatre et l'intervalle `1966111..d6a2a74` | `J15.md` en-tête, §2, §5, §6 |
| **P1** | Verser le script de décision de porte dans le harnais ou le noyau versionné (champ `gate_j15_status` non reproductible) | `diagnostic.py` ou `harness` |
| **P2** | Remplacer « posterior exact » par « distribution causale complète » dans `J15.md` §3.4 | `J15.md` §3.4 |
| **P2** | Corriger l'attribution de commit dans `DOPD_SPEC.md` §6.9.9 (`fe499ff` porte le code ; `d6a2a74` porte le résultat) | `DOPD_SPEC.md` §6.9.9 |
| **P2** | Déclarer les divergences d'échelle dans `J15.md` §3 (8B → 0,6B ; 10 → 3 configurations ; 400 → 100 blocs) | `J15.md` §3 |
| **P2** | Compléter la réserve « queue » par la queue nuisible (13,3 %) | `J15.md` §3.3, §7-1 |

### 7.3 Ce qui ne doit pas être touché

- `modules/dopd/dopd/target.py` — correctifs numériques intacts (P10).
- `modules/dopd/dopd/diagnostic.py` **hors** `truncate_causal_gap` — l'implémentation
  des eq. 20–22/25–26/28 et des statistiques est correcte (P7, P9).
- Les critères pré-enregistrés §6.9.7–6.9.8 — **ne pas les modifier** (P1) :
  les corriger violerait la pré-enregistrement. Les corrections doivent porter
  sur §6.9.9 (résultats) et sur le rapport.
- `modules/a2d/eval/dream_a2d/` et le verdict G1 — intacts (P10).

### 7.4 Pour J16

Le « GO conditionnel » est **confirmé**. Le protocole pré-enregistré du
`TeacherScorer` réel doit reprendre les trois lessons de cet audit :

1. **Rendre reproductible tout champ publié** — le statut de porte ne doit
   pas dépendre d'un script hors versionnement.
2. **Publier la direction, pas seulement l'exclusion de zéro** — un critère
   « IC exclut zéro » doit être annoté par le signe.
3. **Citer le HEAD d'arrivée réel** dans toute preuve de CI, et la rejouer
   sur ce HEAD.

---

## 8. Preuves

Toutes les vérifications de cet audit sont reproductibles sans GPU.

```bash
# Environnement (venv DANS le dépôt, règle 9)
python3 -m venv --system-site-packages _artefact/venv
_artefact/venv/bin/python -m pip install -e . --no-deps --no-build-isolation
_artefact/venv/bin/python -m pip install ruff pytest numpy

# CI sur le HEAD d'arrivée
_artefact/venv/bin/python -m ruff check academiia/ modules/ tests/ tools/ infra/
_artefact/venv/bin/python -m pytest modules/a2d/tests modules/dopd/tests tests \
    infra/vllm/scripts/tests -q --ignore=modules/a2d/tests/test_a2d_mutation.py
_artefact/venv/bin/python tools/check_links.py

# Collecte sans torch (venv dédié)
python3 -m venv _artefact/venv_notorch
_artefact/venv_notorch/bin/python -m pip install numpy pytest setuptools
_artefact/venv_notorch/bin/python -m pip install -e . --no-deps --no-build-isolation
_artefact/venv_notorch/bin/python -m pytest modules/dopd/tests -q -rs
# -> 100 passed, 2 skipped

# Pré-enregistrement
git merge-base --is-ancestor 4a74023 fe499ff
git diff 4a74023..fe499ff -- modules/dopd/docs/DOPD_SPEC.md   # vide
git diff --stat 1966111..HEAD -- modules/a2d/eval/dream_a2d/    # vide
git log --oneline 5351bd4..HEAD -- modules/dopd/dopd/target.py # vide

# Recalcul des chiffres (voir §2 P6, §3 A–B)
_artefact/venv/bin/python - <<'PY'
import json
d = json.load(open("modules/dopd/eval/annexe_d_qwen3_0.6b_multiprompt/results.json"))
rows = d["rows"]; n = len(rows)
mp = d["multiprompt"]
causal = sum(r["causal_kl"] for r in rows) / n
future = sum(r["future_kl"] for r in rows) / n
print("delta", (causal - future), "rel%", 100 * (causal - future) / causal)
print("median", mp["median_delta_kl"], "prop_sig", mp["proportion_prompts_significant"])
print("neg prompts", [p for p, v in mp["per_prompt"].items() if v["mean_delta_kl"] < 0])
vis = [r for r in rows if r["visible_future"]]
print("degraded", sum(1 for r in vis if r["delta_kl"] < 0), "/", len(vis))
PY
```

**Environnement de l'audit** : `_artefact/venv/`, `_artefact/venv_notorch/`,
`_artefact/pkgroot/` (arborescence d'import de secours). Ces trois chemins sont
sous `_artefact/` — **ignorés par git**, conformément à la règle 9.

---

*Fin de l'audit J15. Aucun fichier de mission n'a été modifié par cet audit ;
seul ce journal d'audit est ajouté (règle 12 : les désaccords avec un audit
vont dans un journal d'audit séparé, pas dans le rapport de mission).*
