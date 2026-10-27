# Pré-enregistrement J18R — Mesure reconstruite de l'effet future-aware (post-audit J18)

**Date :** 2026-10-27
**Statut :** figé avant tout code producteur, tout run (GPU ou CPU) et tout résultat. Toute modification après un résultat vaut nouveau protocole, pas correction de rapport.
**Ancêtre du code producteur :** HEAD `f7f1fcc` (audit J18 corrigé, 2026-10-27).
**Aucun code J18R, aucun run, aucun commit de code ne sont autorisés tant que ce pré-enregistrement n'aura pas été accepté par accord écrit explicite de l'utilisateur.**

---

## 0. Objet et contexte

### 0.1 Ce qui est invalide (J18) — et ce qui l'est par décision, non par inference

1. **Écart H (critique, `AUDIT_J18.md` §0).** Dans la géométrie J18 (un seul masque,
   le même bloc servit d'état visible **et** de futur), la cible future-aware est la
   posterior exacte énumérée (écart mesuré 4,5e-10 sur Qwen3-0.6B) :
   `future_kl ≡ 0` et `ΔKL ≡ KL(exact‖causal)` **par construction**. Les critères
   C1–C4 de J18 sont des conséquences algébriques, pas des observations. **Le GO de
   J18 n'est pas certifié et n'est pas transférable à J19.**
2. **`results.json` de J18 reste immuable** (sha256 `4e891a3d9cd54b26f351e629e6dc38b13d46525911db275664b2a425d9fa225e`,
   cité dans `J18.md`). Il porte encore `decision.status = "GO"` sans pointeur
   (divergence **N7**) : le pointeur est livré par `modules/dopd/eval/j18_decision/SUPERSEDED.json`
   (forme figée en §7.12), jamais par une réécriture du fichier.
3. **Aucune hypothèse de signe positif n'est autorisée pour J18R.** Le contre-test
   « variante papier » de l'audit (ΔKL ≈ −3,49) **n'est pas réconcilié** avec J15
   (+0,08) ; l'attribution de l'amplitude J18 (≈ 3,05) au tirage uniforme (écart J)
   **n'est pas démontrée**. Le NO-GO est un résultat **explicitement légitime** de
   J18R. Le contraste « uniforme vs pondéré » n'est **pas** un critère J18R : il
   relève d'une ablation ultérieure (mission distincte, hors périmètre).
4. **Écart F de l'audit initial : retiré.** Le support construit sur le mélange
   pondéré des préfixes restreints (Annexe D.1, eq. 21) est **conforme** ;
   `_build_restricted_support` est réutilisé tel quel.
5. **J14–J17 restent valides** (porte Annexe D franchie avec réserves de lecture,
   `TeacherScorer` réel livré) et **ne sont pas retouchés**. Les réserves J15/J16
   doivent être rappelées dans le rapport J18R (effets non uniformes : médiane nulle,
   queue nuisible de 13,30 %, 3/20 prompts dégradés) — `DOPD_SPEC.md` §6.10.7.

### 0.2 Ce que J18R fait

1. **Reconstruit la mesure** sur la convention J15 (non dégénérée) : blocs de base
   tirés **proportionnellement au joint du professeur** (pas uniforme), rétention 0,45
   par position, **futur compatible tiré séparément** dans le joint perturbé
   `σ ∈ {0 ; 0,5 ; 1,0 ; 1,5}`, posterior exacte conditionnée **uniquement** sur
   l'état visible, **sans** ce futur (§1).
2. **Exerce la règle de production pour la première fois en mesure** (N1) :
   `compute_future_aware_target` (paquet, `target.py`) avec candidats = union
   top-k professeur ∪ top-k « élève », **k < V** pour que la troncature soit
   réellement exercée (N2, N3) (§2).
3. **Valide le harnais par ancrage** : le bras « support complet » doit reproduire
   J15 (mêmes prompts, mêmes graines, même géométrie) à l'intérieur de l'IC 95 %
   publié de J15 ; sinon **dérive du harnais → arrêt, aucune décision** (§4.3).
4. **Décide GO / NO-GO** pour **ouvrir** un `StudentPolicy` réduit (≤ 0,6B) en J19.
   La décision porte sur l'ouverture, pas sur la livraison d'un élève.

### 0.3 Ce que J18R ne fait pas

- Aucun entraînement, aucun `backward`, aucun optimizer, aucun poids modifié, aucun
  checkpoint écrit ;
- Aucun `StudentPolicy` avant la décision (J19) ;
- Aucune échelle > Qwen3-0.6B ;
- Aucun push vers `origin/main` (règle 11 d'AGENTS.md) ;
- Aucune modification de J14–J17, des résultats A2D ou du verdict G1 ;
- Aucune ablation « uniforme vs pondéré » (reportée), aucune comparaison
  d'amplitude avec le papier (non comparable, J16 M6/M7) ;
- **Aucun auto-audit** : l'audit de J18R est réalisé par OpenCode (N1) puis par
  l'utilisateur (N2). Les désaccords éventuels avec l'audit vont dans un **journal
  d'audit séparé**, jamais dans le rapport de mission (règle 12, ADR-005 §4).

### 0.4 Divergences traitées par ce pré-enregistrement

| Réf. | Divergence | Traitement | Section |
|---|---|---|---|
| **H** (audit) | cible future-aware ≡ posterior exacte ; ΔKL identité | géométrie non dégénérée + contrôle négatif figé | §1 |
| **J** (audit) | blocs tirés uniformément | tirage pondéré par le joint (`sample_compatible_index`) | §1.1 |
| **A** (audit) | « rejeu intégral » déclaré mais non exécuté | chemin de mesure **déclaré** (cache) + rejeu intégral FP32 sur ≥ 5 % des jobs comme référence | §5.1–5.2 |
| **E** (audit) | `mean_causal_kl`/`mean_future_kl` non publiés | dégélation **contrôlée** de `diagnostic.py` (2 lignes) + publication obligatoire | §4.4, §7.10 |
| **F** (audit) | support « mélange » | **retirée** — conforme (Annexe D.1) ; support J15 réutilisé tel quel | §0.1.4 |
| **G** (audit) | arithmétique float64 déclarée float32 | float32 **réellement appliquée** via `target.py` (App. B) + test de distinction | §5.3 |
| **I** (audit) | `correctness_gate: true` sans vérificateur | `correctness_gate = false` déclaré | §7.1 |
| **K** (audit) | empreinte d'artefact absente, C7 infalsifiable | `lineage.json` sidecar + hash de `results.json` ; critère falsifiable | §7.2 |
| **L** (audit) | garde-fou après chargement | garde-fou **avant** `load_teacher` | §7.7 |
| **M** (audit) | `torch_version` = version transformers | deux clés distinctes | §7.4 |
| **N** (audit) | champs `training.*` littéraux | dérivés du compteur de `backward` | §7.5 |
| **O** (audit) | tests F3/F4 non détectés, branches non testées | liste de tests figée | §8 |
| **P** (audit) | deux horloges | horloge unique `time.monotonic()` | §7.6 |
| **N1** | `compute_future_aware_target` jamais exercé en mesure | bras de décision passe par la fonction du paquet | §2.3 |
| **N2** | confusion V/k | définitions séparées figées ; `k < V` obligatoire | §1.7 |
| **N3** | top-k de J15 tronquait le mauvais objet (causale plein vocabulaire, `top_k = 16 ≥ V`) | troncature exercée **dans** le support de largeur V, `k < V` | §1.7, §2.5 |
| **N4** | C2 de J18 vide (sous-ensemble « futur visible » non défini) | deux médianes publiées, deux sous-ensembles **non vides** (assertion) | §4.1 C2 |
| **N5** | `bootstrap_seed == seed` (J18) | graines distinctes figées | §3.3 |
| **N6** | aucun template de chat (J14–J18) | bras secondaire non décisionnel | §6 |
| **N7** | `results.json` J18 `GO` sans pointeur | `SUPERSEDED.json` (pointeur audit + J18R) | §7.12 |

---

## 1. Géométrie figée (convention Annexe D / J15, non dégénérée)

Pour chaque prompt et chaque configuration `(N, V[, k])` :

### 1.1 Blocs de base — tirage pondéré (écart J)

- Support restreint construit **tel quel** par `_build_restricted_support`
  (Annexe D.1 : mélange pondéré par les préfixes restreints, top-`V` par position) —
  code J15 inchangé (écart F retiré).
- Joint énuméré `enumerate_joint()` → `V^N` séquences, log-probabilités
  normalisées (eq. 22), arithmétique float64 (J14/J15, inchangée).
- Chaque bloc de base est tiré par
  `sample_compatible_index(log_joint, compatibles, rng)` — **proportionnellement à la
  probabilité jointe du professeur**. Le tirage uniforme est **interdit** (écart J).
- **Rétention 0,45 par position** : une position est conservée (visible) avec
  probabilité 0,45 (`rng.random(N) < 0,45`) ; si toutes les positions sont
  conservées, une position est masquée uniformément au tirage (convention J14/J15).

### 1.2 Futur compatible tiré SÉPARÉMENT (écart H)

Pour chaque niveau `σ ∈ {0 ; 0,5 ; 1,0 ; 1,5}` :

- Perturbation `Z_X ~ N(0,1)` de longueur `|séquences|`, tirée **une seule fois**
  par (configuration, `σ`) (eq. 30, convention J15) et réutilisée pour tous les
  états de base de la configuration ; `perturbed_log = log_joint + σ·Z_X`.
- Pour chaque état de base : l'ensemble compatible = séquences énumérées
  compatibles avec l'état visible ; le futur est tiré **séparément** dans le joint
  perturbé : `sample_compatible_index(perturbed_log, compatibles, noise_rng)`.
- **Le futur tiré ne fait pas partie de l'état visible.** J18 a utilisé le même
  bloc pour les deux rôles : c'est l'écart H.

### 1.3 Posterior exacte de référence (écart H)

- `exact_token_posteriors(sequences, log_joint, visible, supports)` — joint
  **non perturbé**, conditionnée **uniquement** sur l'état visible (positions
  conservées du bloc de base), **sans** le futur tiré.
- C'est la référence contre laquelle chaque cible est mesurée :
  `KL(exact ‖ cible)` (eq. 28, axe **figé** — mutation F2 déjà pincée par la suite
  existante).

### 1.4 Marginaux et « futur visible »

- Marginal = position masquée `i` d'un bloc de base.
- « Futur visible » = il existe `p > i` conservée (non masquée) dans l'état de
  base. Sans futur visible (ni à droite, ni suffixe non vide) : la cible
  future-aware est inopérante — `cible := cible causale`, `ΔKL = 0` par
  construction (eq. 12, cas limite) ; le marginal est compté dans « tous » et
  exclu du sous-ensemble « futur visible » (§4.1 C2).

### 1.5 Contrôle négatif — la géométrie un-masque doit donner `future_kl ≈ 0`

C'est le test qui aurait détecté l'écart H :

- **Géométrie de contrôle** : un seul masque par bloc, toutes les autres positions
  visibles, et le futur := le bloc de base lui-même (équivalent : le futur tiré
  dans le compatible réduit à `{bloc}`). Cible future-aware = reconstruction
  inline eq. 18 sur le support complet (T1, §2.2).
- **Échantillon déterministe** : configurations `(N,V) = (4,12)` et `(4,24)`,
  prompts `p01`, `p02`, `p03`, blocs de base `0, 1, 2`, `σ = 0` → 18 marginaux de
  contrôle (3 × 3 × 2).
- **Seuil figé** : `max(future_kl) ≤ 1e-3` **et** `médiane(future_kl) ≤ 1e-5`.
  (Reproductions de l'audit : 5,6e-11 en moyenne, 4,5e-10 au pire écart — float64 ;
  modèle jouet 6,6e-15. Le seuil est ~4 ordres au-dessus du résidu d'identité
  attendu et ~5 ordres en dessous du plus petit effet réel attendu, ≈ 0,08.)
- **Échec du contrôle** = défaut de validité du harnais → **STOP, aucune décision**
  (ni GO ni NO-GO sur l'effet), diagnostic consigné. Ce n'est **pas** un NO-GO
  scientifique.
- Les 18 valeurs sont publiées (`arms.control`).

### 1.6 Définitions `V` et `k` — interdiction absolue de confusion (N2)

- **`V` (largeur de support)** : nombre de tokens du support restreint par
  position ; vocabulaire de travail du joint restreint et des cibles.
- **`k` (top-k de production)** : taille de chaque top-k de candidats de la règle
  de production (professeur et élève) ; `|candidats| ≤ 2k`.
- **`V` et `k` sont des grandeurs distinctes.** Une configuration ne doit
  **jamais** poser `k = V` (troncation nulle, cas J18) ni `k > V` (cas J15 :
  `top_k = 16 ≥ V`, troncature non exercée — N3). Dans le bras de décision,
  **`k < V` est obligatoire** pour que la troncature soit réellement exercée.

### 1.7 La troncature de J15 ne sert pas de référence de troncature (N3)

La statistique `top_k` de J15 (masse écartée de la **causale plein vocabulaire**
à 16 tokens) mesure un objet **différent** de la troncature de production
(union top-k **dans** le support de largeur `V`). J18R n'en reprend aucune valeur
comme borne ; la troncature de production est mesurée directement sur les marginaux
du bras de décision (distribution de `full_kl − future_kl` — la cible `future_kl`
dans le record du bras de décision **est** `prod_kl` — §4.4).

---

## 2. Trois cibles par marginal (figé)

Chaque marginal du bras de décision porte **trois** cibles ; chaque marginal du
bras d'ancrage porte les deux cibles de la référence J15 (la règle de production
n'est pas définie sans `k`).

### 2.1 T0 — cible causale (référence de mesure, eq. 25)

`r_i(v)` = prior causal du professeur, restreint au support de la position
(`support.causal_prior(prefix, position)`, `prefix = futur[:i]`). Identique dans
les deux bras. `causal_kl = KL(exact ‖ T0)`.

### 2.2 T1 — future-aware support complet (référence J15, eq. 18 inline)

- Scores `s_i(v) = log p_T(futur[i+1:] | q, z, futur[:i], v)` (eq. 26) pour
  **tous** les `V` tokens du support, via le `TeacherScorer` (un lot de branches,
  App. B) ; référence `s_i(x̂_i)` avec `x̂_i = futur[i]`.
- `T1(v) ∝ r_i(v)·exp(s_i(v) − s_i(x̂_i))`, normalisé sur le support ; si le
  normaliseur n'est pas fini ou ≤ 0 : `T1 := T0` (fallback causal, déclaré).
- Arithmétique **float64** : c'est la définition de la référence J15, inchangée.
- `full_kl = KL(exact ‖ T1)` ; `ΔKL_full = causal_kl − full_kl` (métrique J15).

### 2.3 T2 — règle de production (N1) via `compute_future_aware_target`

- **Fonction appelée** : `academiia.dopd.target.compute_future_aware_target`
  (`target.py`, **gelé** — zéro diff, hash §9.2). La reconstruction inline n'est
  **pas** autorisée pour T2 (N1 : la fonction du paquet doit être exercée en
  mesure).
- **Candidats** : `candidate_indices` = union triée sans doublon des :
  - top-`k` du **professeur** : les `k` plus grandes probabilités de `r_i` sur le
    support ;
  - top-`k` de l'**élève** : les `k` plus grandes probabilités de la **marginale
    du joint perturbé** à la position `i`, conditionnée sur l'état visible,
    c'est-à-dire `exact_token_posteriors(sequences, perturbed_log, visible,
    supports)[i]` (même énumération, même posterior, joint perturbé à la place du
    joint exact).
- **Tokens non scorés** : les tokens du support hors candidats reçoivent le
  **facteur 1** (correction nulle, poids `r_i(v)`) — la KL sur le support reste
  **finie** (§3.3 du papier, App. B).
- `reference_loglikelihood = s_i(x̂_i)`, **toujours calculée** (branche dédiée),
  même si `x̂_i ∉ candidats` (docstring de `target.py`).
- `visible_future=False` (pas de futur visible, §1.4) : la fonction rend le prior
  causal (`REASON_NO_VISIBLE_FUTURE`) — `T2 := T0`, `ΔKL_prod = 0`.
- `arithmetic_dtype = "float32"` **réellement appliqué** (App. B ; écart G) —
  vérifié par le test T6 (§8).
- `prod_kl = KL(exact ‖ T2)` ; `ΔKL_prod = causal_kl − prod_kl` — **cible de la
  décision** (§4.2).
- `full_kl` est **également** calculé sur chaque marginal du bras de décision
  (champ informatif `full_kl` du record) : il permet de publier la part de la
  correction perdue par la troncature (`full_kl − future_kl`).
- **Fallback (App. B)** : `TargetResult` porte `applied` et `reason` ; la
  publication des compteurs `reason` est **obligatoire** (`applied` /
  `no_visible_future` / `empty_candidate_set` / `non_finite_scores` /
  `non_finite_normalizer`) : un fallback `non_finite_normalizer` (normaliseur
  non fini) rend `T2 := T0` pour ce marginal (`ΔKL_prod = 0`, compté dans C1) —
  comportement canonique, pas un échec du run.

### 2.4 Records (schéma figé)

- Bras d'ancrage : schéma J15 + `arm = "anchor"` :
  `['arm', 'block_index', 'causal_kl', 'config_index', 'delta_kl', 'future_kl',
  'noise', 'position', 'prompt', 'visible_future']` avec `future_kl = full_kl`,
  `delta_kl = ΔKL_full`.
- Bras de décision : schéma ci-dessus + `full_kl` (informatif, §2.3) ;
  `future_kl = prod_kl`, `delta_kl = ΔKL_prod`.
- Bras de contrôle (§1.5) : `arm = "control"`, mêmes champs, `future_kl` = KL de la
  cible un-masque.
- `noise` ∈ {`0.0`, `0.5`, `1.0`, `1.5`} ; `config_index` = index dans la liste
  des configurations du bras (ordre figé §2.5) ; `block_index` ∈ [0, 100[.

### 2.5 Grille de configurations (figée)

| Bras | Configurations `(N, V[, k])` | `σ` | Rôle |
|---|---|---|---|
| **ancrage** | `(4,6)`, `(4,12)`, `(6,6)` | `0 ; 0,5 ; 1,0 ; 1,5` | **12 configurations** — grille et définition J15 ; T0 + T1 ; reproduit J15 (C9) |
| **décision** | `(4,12,4)`, `(4,24,8)` | `0 ; 0,5 ; 1,0 ; 1,5` | **8 configurations** — T0 + T1 + T2 ; `k < V` (§1.6) ; cible de la décision (C1) |
| **contrôle** | `(4,12)`, `(4,24)` | `0` | 18 marginaux — contrôle négatif (§1.5) |

**Justification des configurations de décision** :

- `k/V = 1/3` dans les deux cas (`4/12`, `8/24`) — même ratio de troncature,
  largeurs différentes : la troncature est exercée à deux échelles ;
- `(4,12,4)` : `V = 12` est dans la grille J15 (comparabilité avec l'ancrage) et
  `k = 4` est le niveau d'ablation de la Table 3 du papier (couverture
  97,87 % / 95,43 % professeur/élève) ;
- `(4,24,8)` : support plus large que toute la grille J14–J15, `k = 8` intermédiaire
  (Table 3 du papier : 99,45 % / 99,44 % à `k = 16`) ;
- `N = 4` partout : défaut du papier (§6.1 du spec), cohérent J14–J18.

Les 20 prompts de J15 (§3.1) × 100 blocs (§3.2) s'appliquent à chaque
configuration des trois bras.

---

## 3. Données et graines (figé)

### 3.1 Prompts

- Les **20 prompts de J15**, mêmes textes, mêmes indices `p01`–`p20`, même fichier
  : `modules/dopd/eval/annexe_d_qwen3_0.6b_multiprompt/prompts.json`
  (schema `dopd.prompts/1`, sha256 `87aa12c9b48a479b0709d7de01912cc00ce77b4ff679fc1f89cfeccb641ba3e5`).
- Aucune sélection, aucun ajout, aucun retrait a posteriori. Le sous-ensemble
  J18 (13 prompts) **n'est pas** repris : J18R mesure sur les 20 prompts, ce qui
  inclut `p10` et `p18` (négatifs en J15, fortement positifs en J18) — la
  contradiction des échelles doit se révéler ou se résoudre sur le jeu complet.

### 3.2 Effectifs

- **Exactement 100 blocs de base par (prompt, configuration)** — satisfait
  « ≥ 100 blocs de base par configuration » (20 × 100 = 2000 blocs échantillonnés
  par configuration).
- Publication **obligatoire** des deux compteurs (correction de l'écart D de J18) :
  `n_blocks_sampled` (blocs échantillonnés, = 2000 par configuration) et
  `n_distinct_block_indices` (= 100).
- Marginaux : comptés après tirage (positions masquées variables par bloc) ;
  ≥ 2000 marginaux au total par bras (C4).

### 3.3 Graines (N5)

| Bras | Graine diagnostic | Graine bootstrap | Origine |
|---|---|---|---|
| ancrage | **20261023** | **20261024** | valeurs J15 — **obligatoires** pour reproduire J15 (C9) |
| décision + contrôle + N6 | **20261027** | **20261028** | distinctes **entre elles** (J18 les posait égales — N5), distinctes de 20261023/20261024 (ancrage) et de 20261026 (J18) |

**Dérivation figée** (même schéma que J15, `diagnostic_annexe_d.py` — toute autre
dérivation = dérive de harnais détectable par C9) :

- rng blocs de base + masquage : `np.random.default_rng(seed + 1000·N + V)` ;
- rng perturbation `Z_X` : `np.random.default_rng(seed + 100_000·N + 10_000·V + int(σ·10))` ;
- rng futur compatible : `np.random.default_rng(seed + 1000·N + V + int(σ·10))` ;
- bootstrap : `bootstrap_delta_ci(records, bootstrap_seed, 10_000, group_fields=("prompt", "config_index", "block_index"))` (fonction du paquet, unités de rééchantillonnage = blocs de base).

### 3.4 Bootstrap

- 10 000 itérations, IC 95 % (percentiles 2,5/97,5), **par blocs de base**,
  clé `(prompt, config_index, block_index)` — la dépendance intra-bloc est
  préservée (unité figée ; mutation F3 pincée par T3, §8).
- IC per-prompt : `bootstrap_prompt_ci` (paquet), même graine bootstrap.
- Tests des signes : sur les marginaux (package, approximation normale, unité
  déclarée) **et** test binomial exact bilatéral sur les signes des moyennes
  per-prompt (unité déclarée) — leçon écarts B (J18) et M8 (J15) : l'unité est
  toujours publiée.

### 3.5 Échantillon de rejeu (figé)

- **Job** = une demande de scoring d'un marginal (prior causal + scores des
  candidats + score de référence).
- Rejeu intégral (sans cache) sur les jobs `job_index mod 20 == 0` — **exactement
  5 %**, sélection déterministe, indépendante des résultats (§5.2).

---

## 4. Critères et règle de décision (pré-enregistrés)

### 4.1 Critères de validité (C5–C10) — tout échec bloque la décision

| # | Critère | Seuil figé |
|---|---|---|
| **C4** | Effectif | 100 blocs de base par (prompt, config) ; ≥ 2000 marginaux par bras de mesure (ancrage et décision) ; `n_blocks_sampled` et `n_distinct_block_indices` publiés |
| **C5** | Intégrité numérique | (a) aucun NaN/Inf ; (b) somme de chaque distribution = 1 ± 1e-5 ; (c) rejeu §5.2 : ≥ 5 % des jobs, max \|Δ log-prob\| ≤ 1e-3 ; (d) float32 réellement appliqué sur T2 (T6) ; (e) contrôle négatif §1.5 : max ≤ 1e-3, médiane ≤ 1e-5 |
| **C6** | Traceabilité (ADR-002, **falsifiable**) | manifeste `academiia.lineage/1` + `lineage.json` sidecar ; `producer.module = "dopd"` ; commande exacte rejouable (§7.3) ; `lineage.json.artifact.hashes["results.json"]` = sha256 recalculé et égal ; `torch_version` et `transformers_version` distinctes et correctes (§7.4) ; `correctness_gate = false` (§7.1) ; `training.*` dérivés (§7.5) ; horloge unique monotone (§7.6) |
| **C7** | Professeur gelé | `requires_grad` faux partout, `model.training = False`, compteur `backward` = 0, aucun optimizer |
| **C8** | Provenance du modèle (règle 9) | garde-fou **avant** `load_teacher` (§7.7) ; snapshot dans `_artefact/hf-cache/` (dans le dépôt) ; `model_source_in_repo = true` ; sha256 de `model.safetensors` au manifeste ; violation → refus **avant** tout chargement |
| **C9** | **Ancrage J15** | `ΔKL_full` du bras d'ancrage ∈ **[0,06381534237632878 ; 0,09617038274240884]** — IC 95 % publié de J15 (agrégat `0,0801905853381537`). Hors intervalle → **dérive du harnais → STOP, aucune décision** (R1) |
| **C10** | Non-régression | `target.py` bit-identique (hash §9.2) ; `diagnostic.py` modifié **uniquement** par les 2 lignes de §7.10 ; `config.py`, `interfaces.py`, `__init__.py` inchangés ; artefacts J14–J17 immuables ; `results.json` J18 immuable (`4e891a3d…`) ; `modules/a2d/` inchangé sur la plage de commits |

### 4.2 Décision primaire (C1) — NO-GO explicitement légitime

**C1 — effet de la règle de production** : `ΔKL_prod` agrégé (moyenne
arithmétique des marginaux du bras de décision) `> 0` **et** l'IC 95 %
(bootstrap §3.4) **exclut** zéro.

Règle :

- C1 tenu **et** C4–C10 tenus → **GO** : ouvrir un `StudentPolicy` réduit (≤ 0,6B)
  en J19 (protocole d'entraînement pré-enregistré distinct — §6.10.7 du spec) ;
- C1 non tenu (`ΔKL_prod ≤ 0` **ou** IC contenant 0) → **NO-GO** : pas
  d'ouverture de `StudentPolicy`. **Le NO-GO est un résultat légitime, pas un
  échec de mission.**

**Aucune hypothèse de signe positif n'est autorisée** (§0.1.3) : le contre-test
« variante papier » de l'audit (ΔKL ≈ −3,49) n'est pas réconcilié avec J15
(+0,08) ; aucune des deux amplitudes n'est transférable à J18R par anticipation.

### 4.3 Statistiques de robustesse (C2, C3) — publication obligatoire, signalées

| # | Statistique | Définition figée |
|---|---|---|
| **C2** | Médianes (N4) | `médiane(ΔKL)` sur **tous** les marginaux **et** sur le sous-ensemble **futur visible** (champ `visible_future = true`) — les **deux** sous-ensembles doivent être non vides (`n_all > 0` et `n_visible_future > 0`, sinon le run est invalide). Une médiane ≤ 0 ou une proportion de dégradés ≥ 0,5 **alors que C1 est tenu** est signalée en rapport comme réserve « effet porté par une queue » (leçons J15/J16) ; elle **ne renverse pas** la décision figée en §4.2 |
| **C3** | Degradation | `frac(ΔKL < 0)` sur les marginaux du bras de décision, publiée avec `mean(ΔKL | ΔKL < 0)` et le pire cas (convention J15 : 13,30 %, −1,1863, −28,7094) |

### 4.4 Publications obligatoires (quel que soit l'issue)

Par bras (ancrage et décision), sur tous les marginaux du bras :

- `n` (tous), `n_visible_future`, `n_no_visible_future` ;
- **`mean_causal_kl`, `mean_future_kl`** (cible du bras), `mean_delta_kl`,
  **réduction relative** = `(mean_causal_kl − mean_future_kl) / mean_causal_kl`
  (références : J14 −16,12 % ; J15 **8,58 %** ; papier 71,9 % sur `0,4444 → 0,1249`) ;
- **médiane sur TOUS les marginaux et sur futur-visible seulement** (C2) ;
- **part de marginaux dégradés** (`ΔKL < 0`) + moyenne des dégradés + pire cas (C3) ;
- **statistiques par-prompt** : `n`, `mean_delta_kl`, `median_delta_kl`, IC 95 %
  per-prompt, signe de la moyenne, significativité ;
- **statistiques par-σ** (par configuration et par niveau) : `n`,
  `mean_causal_kl`, `mean_future_kl`, `mean_delta_kl`, `median_delta_kl` ;
- tests des signes (unités déclarées, §3.4) ;
- bras de décision : distribution de `full_kl − future_kl` (coût de la
  troncature, §1.7) et `|candidats|` par marginal (min/max/moyenne) ;
- bras d'ancrage : `max |ΔKL_J18R − ΔKL_J15|` sur les marginaux partagés
  (clé `prompt, config, block, position, σ`) — diagnostic de dérive (C9) ;
- bras de contrôle : les 18 valeurs `future_kl` (C5e) ;
- décision : statut, règle appliquée, réserves J15/J16 rappelées (§0.1.5),
  budget consommé, versions, graine(s).

### 4.5 Table de décision (priorité décroissante, figée)

| Priorité | Condition | Issue |
|---|---|---|
| 1 | C10 (non-régression) échoue | **NO-GO** + décision explicite écrite (R1) |
| 2 | C7 (professeur gelé) ou C8 (provenance) échoue | **NO-GO** + décision explicite écrite (R1) |
| 3 | **C9 (ancrage) échoue** | **STOP** — dérive du harnais, **aucune** GO/NO-GO, diagnostic de dérive publié |
| 4 | C5 (intégrité) ou C6 (traceabilité) échoue | **NO-GO** + décision explicite écrite (R1) |
| 5 | C1 tenu (et C4 tenu) | **GO** — ouvrir un `StudentPolicy` réduit (≤ 0,6B) en J19 |
| 6 | C1 non tenu (`ΔKL_prod ≤ 0` ou IC ∋ 0) | **NO-GO** — résultat légitime |

**R1** (`DOPD_SPEC.md` §6.10.8) : le silence n'est pas un GO. Toute issue est
consignée dans un rapport de mission avec la règle appliquée.

---

## 5. Exactitude numérique (figé)

### 5.1 Chemin de mesure — déclaré (écart A)

Le **chemin cache** de préfixe du `TeacherScorer` (App. B, §3.6) est le chemin de
mesure — **déclaré comme tel** dans le manifeste et le rapport. Le rejeu intégral
n'est plus présenté comme la référence globale (formulation J18 inexacte) : c'est
la référence **échantillonnée** de §5.2.

### 5.2 Rejeu intégral FP32 sur ≥ 5 % des jobs (écart A)

- Jobs `job_index mod 20 == 0` (§3.5) : re-score via `replay_scores` /
  `replay_job` (`use_cache=False`), professeur FP32.
- **Tolérance figée : max |Δ log-probabilité| ≤ 1e-3** (prior causal **et**
  scores candidats) entre les deux chemins, par job ; toute violation = C5 échoué.
- Référence de marge : J17 mesurait ≤ 6,2e-05 en FP32 sur états réels GPU —
  la tolérance garde un facteur ~16 de marge.
- Les écart(s) observés sont publiés (`numerics.replay` : n_jobs, max/min écart).

### 5.3 Arithmétique — réellement float32 via `target.py` (écart G, App. B)

- T2 : `arithmetic_dtype = "float32"` **appliqué** (deltas, facteurs `exp(Δs)`,
  poids, normaliseur en float32 dans `compute_future_aware_target`) ; le test T6
  (§8) distingue le résultat float32 du résultat float64 sur un fixture dédié.
- T1 (ancrage) : float64 — définition de la référence J15, inchangée ;
  la déclaration de dtype est **exacte** dans le manifeste par bras.
- KL et énumération : float64 (paquet, inchangé).
- Le champ `arithmetic_dtype` du manifeste ne peut plus être une description
  déconnectée du code (écart G).

### 5.4 Professeur

- Chargé en **FP32** (`load_teacher(dtype=float32)` — continuité J17 R1 / J18) ;
  dtype natif du checkpoint (bfloat16) déclaré séparément (`scorer_dtype`).
- Arithmétique du scorer float32 (`teacher_scorer_qwen3.py`, inchangée).
- Professeur gelé (C7) ; aucun `backward`.

### 5.5 `target.py` gelé

Zéro diff sur la plage de commits (hash §9.2). Toute modification de `target.py`
= violation C10 = NO-GO (priorité 1).

### 5.6 Intégrité des distributions

- `max_sum_error ≤ 1e-5` sur toutes les distributions (cibles, posteriors,
  marginales) ; `finite = true` (convention J14/J15, C5a–C5b).
- Vérification d'énumération (deux implémentations du posterior exact) conservée
  sur un échantillon déterministe (convention J14) — incluse dans C5.

---

## 6. Bras secondaire N6 — template de chat (non décisionnel, ≤ 30 min)

- **Objet** : quantifier la sensibilité à l'absence de template de chat
  (N6) — J14–J18 ont mesuré sur des prompts bruts `« Question: … Answer: »`,
  sans `apply_chat_template`.
- **Protocole** : mêmes 20 prompts (textes J15) tokenisés par
  `tokenizer.apply_chat_template([{"role": "user", "content": <texte>}],
  add_generation_prompt=True, enable_thinking=False)` ; même grille de décision
  (`(4,12,4)`, `(4,24,8)` × 4 σ), mêmes graines (20261027/20261028), 100 blocs de
  base par (prompt, config) ; cibles T0/T1/T2 comme le bras de décision.
- **Statut** : **non décisionnel** — ne participe **pas** à C1 ni à la table §4.5.
  Publication : `arms.n6_chat_template` avec `ΔKL_prod` agrégé + IC 95 % +
  per-prompt + comparaison chiffrée au bras de décision (même géométrie, même
  graines : écart attribuable au template).
- **Budget** : plafond dur **1800 s** (30 min) dans le budget global §7.8 ;
  au-delà, le bras s'arrête et les statistiques partielles sont publiées
  **explicitement partielles** (le bras de décision et l'ancrage ne sont
  jamais sacrifiés sur ce bras).

---

## 7. Gouvernance (figé)

### 7.1 Porte de correction (écart I)

`correctness_gate = false` **déclaré** au manifeste : J18R n'implémente **aucun**
`TaskVerifier`, aucune `R(x̂)` ; la correction future-aware est appliquée à tous
les blocs échantillonnés et le fallback causal est documenté (`DOPD_SPEC.md`
§3.4/§6.2 : « sans vérificateur, la porte doit être désactivée et le fallback
causal documenté »).

### 7.2 Manifeste ADR-002 + sidecar + hash de l'artefact (écart K)

- Manifeste `academiia.lineage/1` **dans** `results.json` (`producer.module =
  "dopd"`, `producer.version`, `producer.commit`, `producer.command`, `parent`,
  `data`, `config` complet, `training`, `verification`).
- **Sidecar `lineage.json`** écrit **à côté** de l'artefact
  (`modules/dopd/eval/j18r_decision/lineage.json`) — règle ADR-002 (« écrit à
  côté de l'artefact ») non appliquée en J18.
- **Hash de l'artefact produit** : `lineage.json.artifact.hashes["results.json"]`
  = sha256 de `results.json` (calculé **après** écriture de `results.json`, avant
  écriture du sidecar). C6 est **falsifiable** : un vérificateur recalcule le
  digest et compare (écart K : `hashes = {}` et C7 infalsifiable en J18).
- `lineage.json` est lui aussi commité ; son sha256 figure dans le rapport.

### 7.3 Commande exacte, rejouable (écart C)

`producer.command` doit être **exactement** la chaîne suivante (une commande
logique, racine du dépôt, venv du dépôt activé, sans variable d'environnement
spéciale — la redirection de cache HF est imposée **par le harnais** avant tout
chargement, §7.7) :

```
python modules/dopd/harness/run_j18r_decision.py --model Qwen/Qwen3-0.6B --device cuda --dtype float32 --prompt-file modules/dopd/eval/annexe_d_qwen3_0.6b_multiprompt/prompts.json --blocks 100 --seed 20261027 --bootstrap-seed 20261028 --bootstrap-iterations 10000 --anchor-seed 20261023 --anchor-bootstrap-seed 20261024 --replay-fraction 0.05 --replay-tolerance 1e-3 --n6-chat-template --n6-max-seconds 1800 --max-seconds 14400 --max-gpu-mib 16384 --output modules/dopd/eval/j18r_decision/results.json
```

Tout écart entre `producer.command` et la commande réellement exécutée =
violation C6. La rejeuabilité est testée en Phase 1 (parse de la commande +
`--help` + exécution sèche des drapeaux).

### 7.4 Versions séparées (écart M)

`protocol.torch_version = torch.__version__` **et**
`protocol.transformers_version = transformers.__version__` — deux clés
distinctes, toutes deux présentes et correctes (J18 avait écrit `5.10.2` dans
`torch_version`). `protocol.python_version` également.

### 7.5 Champs `training.*` dérivés (écart N)

`training.performed`, `training.backward`, `training.weights_updated` sont
**dérivés** du compteur de `backward` mesuré au runtime (hook/counter ; = 0 en
J18R) : `x = (backward_count > 0)` pour les trois champs — jamais de littéral.
`training.backward_count` publié. `training.requires_grad_all_false` : prédicat
mesuré sur le professeur (convention J18, conservée).

### 7.6 Horloge unique (écart P)

**Une seule** horloge : `time.monotonic()`, pour la durée murale, les
progressions **et** les garde-budgets. `wall_seconds` = différence monotone.
L'usage de `time.time()` dans le harnais J18R est interdit (test statique de
Phase 1 : aucune occurrence dans `run_j18r_decision.py`).

### 7.7 Garde-fou « modèle dans le dépôt » AVANT chargement (écart L, règle 9)

1. Résolution de la source (HF_HOME/HUGGINGFACE_HUB_CACHE redirigés **par le
   harnais** vers `_artefact/hf-cache/`, dans le dépôt) ;
2. Vérification d'appartenance au dépôt du snapshot résolu ;
3. Violation → **refus avant tout chargement** (sortie 2, message, aucun
   forward) ;
4. `model_source_in_repo`, chemin du snapshot, `model_revision`, sha256 de
   `model.safetensors` au manifeste (C8).

### 7.8 Budget (figé)

| Poste | Plafond | Effet du dépassement |
|---|---|---|
| Temps GPU cumulé (tous bras, hors N6) | **≤ 4 h** (14 400 s) | arrêt immédiat, **aucun** `results.json` écrit au-delà (convention J18) |
| Mémoire GPU pic | **≤ 16 GiB** (`torch.cuda.max_memory_allocated`) | arrêt immédiat, idem |
| Bras N6 | ≤ 1800 s | arrêt du bras N6 seul, publication partielle explicite (§6) |
| Échelle | Qwen3-0.6B uniquement | — |
| RAM process | consignée (`max_rss_mib`), sans plafond | — |

Avant tout run GPU : `nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free
--format=csv` ; besoin ≈ 16 GiB ; si un modèle LM Studio est chargé **et**
`memory.free < 17 GiB` : **STOPPER et demander la permission** (règle 6
d'AGENTS.md) ; jamais décharger un modèle LM Studio ni tuer de processus.

### 7.9 Push interdit (règle 11)

Aucun push `origin/main`, aucun tag, aucun upload HuggingFace. Commits locaux
atomiques uniquement. La mission J18R ne contient **pas** d'accord de push écrit
(`AGENTS.md` règle 11 : accord mission par mission) — la présente mission
l'interdit explicitement.

### 7.10 Dégélation contrôlée de `diagnostic.py` (écart E)

**Uniquement** l'exposition de `mean_causal_kl` / `mean_future_kl` dans le
dictionnaire retourné par `evaluate_multiprompt` (`diagnostic.py`) — deux lignes
ajoutées, les valeurs étant déjà calculées par `summarize_marginals`
(`diagnostic.py:360-365`) :

```python
"mean_causal_kl": summary["mean_causal_kl"],
"mean_future_kl": summary["mean_future_kl"],
```

**Aucune autre ligne** de `diagnostic.py` n'est modifiée (hash avant changement :
`e936ca5ada3f217ea3c10bcda751e685c60f285585bc59c010b0de9e232aa6a8`, §9.2).
`target.py`, `config.py`, `interfaces.py`, `__init__.py` : **gelés** (C10).

### 7.11 Immutabilités

- Artefacts J14–J17 (`eval/annexe_d_qwen3_0.6b*/`, `eval/teacher_scorer_qwen3_0.6b*/`,
  `eval/budget_rtx5090/`) : non modifiés, non régénérés.
- `modules/dopd/eval/j18_decision/results.json` : **immuable**
  (`4e891a3d9cd54b26f351e629e6dc38b13d46525911db275664b2a425d9fa225e`).
- `modules/a2d/` (verdict G1, résultats) : inchangé.

### 7.12 `SUPERSEDED.json` (N7) — forme figée

Fichier **nouveau**, livré dans `modules/dopd/eval/j18_decision/` (à côté du
`results.json` J18 qu'il **ne modifie pas**) :

```json
{
  "kind": "dopd_superseded_notice",
  "supersedes": {
    "path": "modules/dopd/eval/j18_decision/results.json",
    "sha256": "4e891a3d9cd54b26f351e629e6dc38b13d46525911db275664b2a425d9fa225e"
  },
  "reason": "Ecart H (critique) de l'audit J18 : dans la geometrie un-masque, la cible future-aware est la posterior exacte (4,5e-10 pres) ; future_kl = 0 et dKL = KL(exact||causal) par construction ; le GO de J18 n'est pas certifie et n'est pas transferable a J19.",
  "audit": {
    "path": "modules/dopd/docs/reports/AUDIT_J18.md",
    "sha256": "0e211490a2bb99650c3f7226f320206c4287c5a02fb6657f82c3cf45d302eaae",
    "sections": ["§0 ecart H", "§7.1 verdict corrige (2026-10-27)"]
  },
  "replacement": {
    "preregistration": "modules/dopd/docs/reports/J18R_PRE.md",
    "results": "modules/dopd/eval/j18r_decision/results.json (livre en phase 2 de J18R)"
  },
  "status": "superseded — ne pas s'appuyer sur le GO de J18",
  "date": "2026-10-27"
}
```

> Le champ `date` prend la date réelle de livraison (phase 3) ; la valeur
> ci-dessus est celle de ce pré-enregistrement.

### 7.13 Audit de J18R

- Réalisé par **OpenCode (N1)** puis par **l'utilisateur (N2)** — **pas
  d'auto-audit** par le producteur.
- Les désaccords éventuels avec l'audit vont dans un **journal d'audit séparé**
  (`AUDIT_J18R.md` ou équivalent), **jamais** dans le rapport de mission
  (règle 12, ADR-005 §4.3).
- Le rapport de mission J18R cite la plage de commits `HEAD_départ..HEAD_arrivée`
  et les preuves ; il ne liste pas ses propres commits de consignation
  (ADR-005 §4.1).

---

## 8. Tests — liste figée (code en phase 1, après acceptation)

| # | Test | Ce qui est vérifié | Avoir détecté |
|---|---|---|---|
| **T1** | **Identité / non-dégénérescence** | (a) géométrie J15 (futur tiré séparément) sur fixture déterministe : `médiane(future_kl) > 0` et `ΔKL` non identitaire ; (b) géométrie un-masque : `max(future_kl) ≤ 1e-3` et `médiane ≤ 1e-5` (contrôle négatif §1.5) | **écart H** (le test (b) aurait détecté H avant le run) |
| **T2** | Mutation **F4** (`future := causal`) | sur le fixture T1(a), le remplacement de T1/T2 par T0 donne `ΔKL ≡ 0` : la suite **doit échouer** (assertion `ΔKL > 0`) | faute non pincée en J18 (audit §O) |
| **T3** | Mutation **F3** (bootstrap par marginaux) | fixture à corrélation intra-bloc : l'IC par blocs ≠ l'IC par marginaux ; assertion sur l'unité de rééchantillonnage et la clé `(prompt, config_index, block_index)` | faute non pincée en J18 (audit §O) |
| **T4** | **Équivalence ligne-à-ligne** T1 inline (support complet, tous les `V` tokens scorés) ↔ `compute_future_aware_target` avec `candidate_indices` = tout le support, **dans le régime fini** (normaliseurs finis des deux côtés ; fixture choisi en conséquence) | comparaison étape par étape : deltas égaux, facteurs et normaliseur au relatif ≤ 1e-6, cible finale `max |p_inline − p_paquet| ≤ 1e-5` (float64 inline vs float32 paquet) | N1 : les deux chemins de construction doivent coïncider ; la règle de production est bien l'eq. 18 tronquée |
| **T5** | **Branches d'échec C4–C10** de la table §4.5 | chaque branche produit le statut figé (synthétiques) : `ΔKL ≤ 0` → NO-GO ; IC ∋ 0 → NO-GO ; ancrage hors IC → STOP ; rejeu > 1e-3 → C5 ; `hashes` vide → C6 ; professeur non gelé → C7 ; modèle hors dépôt → C8 ; hash `target.py` modifié → C10 → NO-GO priorité 1 ; effectif < 2000 marginaux → C4 | branches jamais testées en J18 (audit §O) |
| **T6** | **float32 réellement appliquée** | fixture où float32 et float64 donnent des cibles distinctes : `compute_future_aware_target(arithmetic_dtype="float32")` reproduit la référence float32 et **s'écarte** de la référence float64 au-delà du seuil de distinction | écart G (déclaration déconnectée du code) |
| **T7** | **Séparation tests purs / torch** | les tests T1–T6 s'exécutent **sans torch** (NumPy + fixtures figés) ; les tests qui nécessitent torch sont isolés dans des fichiers distincts (`pytest.importorskip("torch")` au niveau fichier, jamais module global de tests purs) — collecte sans torch verte (convention J15 P3) | audit §O (fichier entièrement sauté sans torch) |

**Régressions conservées** (déjà vertes) : F1 (signe de `delta_kl`) et F2 (axe
KL) par `test_dopd_diagnostic.py` — non supprimées, non affaiblies.

---

## 9. Environnement et sources figées

### 9.1 Environnement d'exécution (figé)

- Python **3.12.3** ; torch **2.12.0+cu130** ; transformers **5.10.2**
  (versions déclarées séparément, §7.4).
- Venv **dans le dépôt** : `_artefact/venv/` (règle 9),
  `pip install -e . --no-deps`.
- Caches HF redirigés **dans le dépôt** : `_artefact/hf-cache/` (règle 9) —
  imposé par le harnais (§7.7), pas par l'environnement d'invocation.
- Professeur : `Qwen/Qwen3-0.6B`, révision
  `c1899de289a04d12100db370d81485cdf75e47ca`, snapshot présent dans
  `_artefact/hf-cache/` (in-repo).
- GPU : RTX 5090 32 Go, WSL2 ; règle VRAM AGENTS.md §6 (§7.8).

### 9.2 Sources figées (hashes au moment du gel de ce PRE)

| Fichier | sha256 | Statut J18R |
|---|---|---|
| `modules/dopd/dopd/target.py` | `e9568bf3329c8c0a17d09019c966ef2f8919ed132ea9240d9cc57005097b4556` | **gelé** — zéro diff (C10) |
| `modules/dopd/dopd/diagnostic.py` | `e936ca5ada3f217ea3c10bcda751e685c60f285585bc59c010b0de9e232aa6a8` | dégélation contrôlée — 2 lignes §7.10 |
| `modules/dopd/dopd/config.py` | `2efeb22cf11f1250d7f09b72ec43d59727c901f2c9b9d23e1fb58cc544de3eed` | gelé |
| `modules/dopd/dopd/interfaces.py` | `1af4b1e441bf54e4b496da1034411b741ec7419ca8022c59c2a0fe4c27f40ce0` | gelé |
| `modules/dopd/dopd/__init__.py` | `8598b8c8bed0f46285ac36f5f3b63662d733c6d492e2d8ac5743393c84eafaf6` | gelé |
| `modules/dopd/harness/teacher_scorer_qwen3.py` | `091db9dc008abf0df2621be8e16e1cc9e79791f28a15392babb90378b783d0be` | réutilisé inchangé (J17) |
| `modules/dopd/harness/diagnostic_annexe_d.py` | `8132c775ca455e95b6443e47e3e46cf0de7955227ff06a0f656b2f4faacd1afd` | référence J15 — inchangé |
| `modules/dopd/eval/annexe_d_qwen3_0.6b_multiprompt/prompts.json` | `87aa12c9b48a479b0709d7de01912cc00ce77b4ff679fc1f89cfeccb641ba3e5` | source des 20 prompts (J15) |
| `modules/dopd/eval/annexe_d_qwen3_0.6b_multiprompt/results.json` | `ed41ed7e4aee15c66289ae0d3865efcdaf4e0327761d735a62da2f26e742c5f5` | **ancrage C9** (immuable) |
| `modules/dopd/eval/j18_decision/results.json` | `4e891a3d9cd54b26f351e629e6dc38b13d46525911db275664b2a425d9fa225e` | **immuable** (J18) + `SUPERSEDED.json` §7.12 |
| `modules/dopd/docs/reports/AUDIT_J18.md` | `0e211490a2bb99650c3f7226f320206c4287c5a02fb6657f82c3cf45d302eaae` | source des écarts (immuable) |

### 9.3 Chiffres de référence (ancrage et réserves)

| Référence | Valeur | Source |
|---|---|---|
| J15 — ΔKL agrégé / IC 95 % | `0,0801905853381537` / `[0,06381534237632878 ; 0,09617038274240884]` | `results.json` J15 (C9) |
| J15 — médiane / queue nuisible / prompts dégradés | `0` / `4661/35040 = 13,30 %` (moy. −1,1863, pire −28,7094) / `3/20` (`p06`, `p10`, `p18`) | J15/J16 (rappel obligatoire §4.2) |
| J15 — réduction relative | `8,58 %` | J16 (écart B) |
| J14 — réduction relative (prompt unique) | `16,12 %` | J14 (à ne pas confondre, J16 écart B) |
| Papier (Table 4, Qwen3-8B) | KL causale `0,4444` → `0,1249` (−71,9 %) | `DOPD_SPEC.md` §5.4 (non comparable — J16 M6/M7) |
| J17 — écart cache/rejeu (FP32, états réels) | ≤ `6,2e-05` | J17 (marge de la tolérance §5.2) |
| Audit — identité H (float64, Qwen3-0.6B réel) | `future_kl` moy. `5,6e-11`, pire écart `4,5e-10` | `AUDIT_J18.md` §0 (base des seuils §1.5) |

---

## 10. Livrables et phases (après acceptation écrite)

**Phase 0 (cette livraison)** : `J18R_PRE.md` — commité seul, puis **STOP**.

**Phase 1 (code — débloquée par l'acceptation écrite du PRE)** :

- `modules/dopd/harness/run_j18r_decision.py` (nouveau, hors paquet, ADR-001 :
  torch/transformers autorisés ici) — géométrie §1, cibles §2, bras §2.5,
  rejeu §5.2, gouvernance §7 ;
- les 2 lignes de `diagnostic.py` (§7.10) — **seule** modification de paquet ;
- tests T1–T7 (§8) + régressions F1/F2 conservées ;
- CI verte : `ruff check` + `pytest modules/a2d/tests modules/dopd/tests tests
  infra/vllm/scripts/tests -q` (hors mutation, convention existante).

**Phase 2 (run)** : commande canonique §7.3 ; budget §7.8 ; artefacts :
`modules/dopd/eval/j18r_decision/results.json` + `lineage.json`.

**Phase 3 (rapport)** : `modules/dopd/docs/reports/J18R.md` (statut, preuves,
publications §4.4, réserves J15/J16, budget, versions) +
`modules/dopd/eval/j18_decision/SUPERSEDED.json` (§7.12) + bandeau de statut sur
les documents concernés (J18.md porte déjà le bandeau de renvoi).

**Audit** : OpenCode (N1) → utilisateur (N2) (§7.13).

---

## 11. Acceptation

**Ce document est figé.** Toute modification après un résultat vaut nouveau
protocole, pas correction de rapport. Les phases 1 (code), 2 (run) et 3
(rapport) sont **bloquées** jusqu'à l'acceptation écrite explicite du
présent pré-enregistrement par l'utilisateur.

**Aucun code producteur, aucun run de mesure, aucun commit de code n'est
produit dans cette phase 0.**
