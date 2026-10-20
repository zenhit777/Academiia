# Rapport de mission J6bis — Clôture corrective de J6 : intégrité, bibliothèque, harnais, protocole d'évaluation

**Date** : 2026-10-20 · **Périmètre** : sections A à E de la mission J6bis
(A1–A3, B1–B7, C1–C7, D1–D3, E1–E4) · **J7, G2, G3 : non démarrés**
(le push est réservé à l'utilisateur : les 5 commits ci-dessous sont **locaux**).

> **Post-audit N1** (2026-10-20) : corrections **F1–F7** (documentation, garde-fous CLI,
> inventaire) — voir **§7**. Le rapport d'audit est versionné dans
> `docs/reports/AUDIT_J6BIS.md`.

| Commit | Contenu |
|---|---|
| `a1b2b46` | `docs(review)` : verdict N2 reproduit tel quel + ERRATUM N2 en tête de l'audit N1 |
| `1757bf7` | `docs(J6bis A2/A3)` : requalification factuelle de J6 + statut des portes |
| `b57f29e` | `feat(a2d)` : plan sur l'union τ_A/τ_D, vérification v2 signée, tests de mutation |
| `9ae02a1` | `feat(harness)` : harnais Dream versionné dans le dépôt + résultats à noms portables |
| `8b1132f` | `docs(design)` : protocole G1, budget d'évaluation, G2 adapter scaling, G1b Nemotron-3B |

---

## 1. Vérifié (artefact à l'appui)

### A — Intégrité et requalification documentaire

| Tâche | Fait | Artefact |
|---|---|---|
| **A1** | `docs/reports/REVIEW_N2_J6.md` créé : le verdict N2 est **reproduit tel quel** (bloc « Verdict N2 sur J6 » + table de correspondance exigence → livrable). Un **ERRATUM N2** est inséré **en tête** de `docs/reports/AUDIT_J6.md` ; **le corps de l'audit N1 est intact** (aucune ligne du corps n'a été modifiée). L'ERRATUM invalide MAJ-07 (le push a été fait **par l'utilisateur, après la livraison de J6** ; le reflog n'identifie pas l'auteur d'un push) et corrige le bilan en **2 bloquants, 7 majeurs, 8 mineurs**. | `docs/reports/REVIEW_N2_J6.md`, `docs/reports/AUDIT_J6.md` (lignes 3-25), commit `a1b2b46` |
| **A2** | **État git factuel** dans `docs/reports/J6.md` : bloc d'état en tête (14 commits **locaux** à la rédaction ; push ultérieur par l'utilisateur, 2026-10-20 11:24:02 pour `284fb62` ; aucune attribution). La mention « les 3 commits sont locaux » est corrigée (le tableau liste 14 commits, la ligne `284fb62` ajoutée). **C0.5 = NON ATTEINT** avec les trois raisons (Base MATH-175 **+3,2** hors ±3 ; IFEval n=8 non mesuré ; « la cible est dans l'IC » est une non-rejection, pas une équivalence — il faudrait un IC *contenu* dans ±3, cf. TOST). **« MATH-500 » → « MATH-175 (sous-ensemble minerva_math, 175/5000) »** partout, y compris `summarize.py:benchmark_of` (avec `PAPER_TARGET_KEY` / `TARGET_SCOPE` pour marquer le périmètre différent) et dans `SUMMARY.{json,md}` régénérés. **MAJ-06 reformulé** : l'évaluation établit le harnais et les checkpoints de référence ; la bibliothèque de fusion n'est validée que **mécaniquement**. **MIN-01** (α sans plafond / λ borné à [0,1] par l'eq. 2, dans `docs/A2D_SPEC.md` et le README), **MIN-02** (routeur/experts = équivalents AR, distincts du self-conditioning), **MIN-03** (graines réelles : `random_seed 0`, `numpy/torch/fewshot 1234`, relues dans les 6 `results_*.json`), **MIN-05** (22 tests A2D à la livraison J6bis / 65 au total, `KNOWN_ISSUES.md`), **MIN-06** (`GUIDE_RESTANT.md` marqué obsolète, docstrings de `measure_effect.py` et `compute_tau_a.py` corrigées), **MIN-07** (arbre propre : les 2 `.jsonl` signalés ont été commités en `31ea9df`, `git status` est vide). | `docs/reports/J6.md`, `README.md`, `CHECKLIST.md`, `docs/A2D_SPEC.md`, `KNOWN_ISSUES.md`, `GUIDE_RESTANT.md`, `scripts/measure_effect.py`, `scripts/compute_tau_a.py`, `harness/dream/summarize.py`, `eval/dream_a2d/SUMMARY.{json,md}`, commit `1757bf7` |
| **A3** | Table de statut des portes dans `CHECKLIST.md` : **G1 ouverte** (fait et mesuré / non atteint, avec la définition de franchissement), **G1b non démarrée**, **G2 bloquée (B7)**, **G3 non démarrée** ; J4Bis et J5 inchangés. | `CHECKLIST.md` (table « État des portes »), `README.md` (table « Trajectoire par portes ») |

### B — Bibliothèque : correction et durcissement (CPU, aucun GPU)

| Tâche | Fait | Artefact |
|---|---|---|
| **B1** | Le plan de fusion est construit sur l'**UNION** des modules de τ_A et de τ_D ; chaque module est classé `both` / `A-only` / `D-only`. Politique **`--coverage {strict,zero-fill}`** : `strict` (défaut quand les deux sources sont des paires de checkpoints complets) **refuse** une couverture inégale en **listant les modules fautifs** ; `zero-fill` traite un vecteur absent comme nul et **déclare** la liste exacte au manifeste. Le choix est **explicite obligatoire** dès qu'une source est un adaptateur LoRA (message d'erreur explicite). Le manifeste publie la politique, les compteurs et les effectifs **par famille**. | `scripts/a2d/merge.py` (`build_plan`, `_module_candidates`, `_coverage_summary`), `scripts/merge_a2d.py` (`--coverage`), `scripts/verify_merge.py` (`--coverage`) |
| **B2** | `ablation-dp-minus-tau-d` construit son plan **depuis τ_D seul** (plus de dépendance à τ_A) : le mode n'est plus un no-op certifié. Un plan **vide** ⇒ `MergeError` (« plan de fusion VIDE ….Refus d'écrire un checkpoint identique à la référence ») sauf `--allow-empty` explicite. | `scripts/a2d/merge.py` (`build_plan`, `merge_checkpoint`), `scripts/merge_a2d.py` (`--allow-empty`) |
| **B3** | **Une seule** fonction `expected_tensor(ref, tau_a, tau_d, c_a, c_d, family, policy)` définit la valeur attendue ; elle est utilisée par le **writer** (`merge_checkpoint`) **et** par le vérificateur (`verify_checkpoint`, donc par `verify_merge`). L'alignement des embeddings (intersection des lignes) est fait **à l'intérieur** ⇒ les deux chemins ne peuvent plus diverger (cause de MAJ-03). `verify_written` et `compute_effective_coefficients` partagent désormais `verify_checkpoint`. | `scripts/a2d/merge.py` (`expected_tensor`, `_align_delta`, `verify_checkpoint`) |
| **B4** | **Vérification v2** : régression des **moindres carrés signés** de (out − ref) sur [τ_A, τ_D] (Gram **float64**), par famille et globalement ; les coefficients **signés** `c_a_eff` / `c_d_eff` sont comparés à ceux **attendus du mode** (transfert (α,0) ; merge (λ,−λ) ; ablations). Les bornes de tolérance sont **dérivées du résidu BF16 mesuré** (matrice de la moindres carrées appliquée au résidu), et **non** de l'écart réel du tenseur écrit — une corruption ne peut donc pas élargir la tolérance. Repli sur des **projections simples** si \|cos(τ_A,τ_D)\| ≥ 0,99 (mal conditionné), avec signalement. Deux contrôles supplémentaires : les tenseurs **non fusionnés sont bit-identiques** à la référence (comparaison des motifs d'octets) ; l'écart maximal à `round_bf16(expected)` recalculé indépendamment est **≤ 1 ULP** du dtype cible. | `scripts/a2d/merge.py` (`verify_checkpoint`, `_solve`, `_acc_add`, `_ulp`, `_bit_identical`) |
| **B5** | `verify_merge` appelle `validate_config` : `--mode merge` **exige** τ_D (sortie **2**, message explicite) ; `--mode ablation-dp-minus-tau-d` n'exige plus τ_A. Le mode `merge` sans τ_D était **accepté** avant (MAJ-01) : c'est corrigé. | `scripts/verify_merge.py`, `scripts/a2d/merge.py` (`compute_effective_coefficients` → `validate_config`) |
| **B6** | **20 tests de mutation** (21 aujourd'hui ; `scripts/tests/test_a2d_mutation.py`, pytest paramétré) : inversion de signe, permutation α↔λ, permutation τ_A↔τ_D, coefficient faux, τ_D partiel sous `strict` (refusé) et sous `zero-fill` (déclaré), τ_A absent, plan vide (refusé / `--allow-empty`), tenseur non fusionné altéré, mode erroné, lignes d'embedding intersectées (vérifiables **et** mutables), seuils ULP BF16 (1 ULP passe, 2 ULP échoue), choix explicite de couverture exigé pour une source LoRA, couverture déclarée au manifeste. Les sondes **BLK-01** et **BLK-02** sont des tests de non-régression. La **CI** les exécute : étape `Tests adversariaux (mutations)` + 4 étapes CLI (ablation-minus, merge sans τ_D refusé, τ_D partiel refusé en `strict`, `zero-fill` déclaré). | `scripts/tests/test_a2d_mutation.py`, `.github/workflows/ci.yml` |
| **B7** | **Conception seule** (`docs/design/G2_adapter_scaling.md`) : l'App. A.2 (les adaptateurs de diffusion DG **restent non fusionnés** et sont **mis à l'échelle**), la réécriture `D_Merge(λ) = [D_b + λτ_A] + (1−λ)·adaptateurs`, le mode « base + adaptateur remis à l'échelle » (3 options analysées, 2 retenues), le traitement des **poids partagés** encodeur/décodeur (vérifié sur l'index du checkpoint : **une seule copie** des poids ; les 413 modules de l'adaptateur = 205 décodeur + 205 encodeur + 3 self-conditioning ⇒ le delta effectif est la **somme** des deux jeux), le plan de vérification V1–V8, et **10 hypothèses dont 5 non vérifiées** contre le PDF. **Aucun code écrit.** | `docs/design/G2_adapter_scaling.md` |

### C — Harnais : conformité au workspace

| Tâche | Fait | Artefact |
|---|---|---|
| **C1** | `harness/dream/` **versionné** : `README.md`, `setup.sh` (idempotent), `requirements-dream.txt` (déplacé depuis la racine), `tasks/` (déplacé), `run_evals.sh` (déplacé et réécrit), `summarize.py` (déplacé). `setup.sh` clone le fork `DreamLM/Dream` **épinglé sur le commit `31f94a60d187e3fd481fee3bbc2c732eb94a879c`** (celui qui a produit les résultats J6), crée `.venv/`, installe torch 2.8.0+cu128 / transformers 4.46.2 / datasets 2.21.0 puis le fork en editable, et **redirige `HF_HOME`, `HF_DATASETS_CACHE`, `HUGGINGFACE_HUB_CACHE`, `PIP_CACHE_DIR`, `XDG_CACHE_HOME` et `TMPDIR`** vers `harness/dream/.cache/` **avant** toute opération réseau. `run_evals.sh` utilise des **chemins relatifs** (racine déduite de `BASH_SOURCE`) et les paramètres `MODEL_DIR`, `TASK`, `LIMIT`, `SEED`, `OUTDIR`, `BATCH` + la logique de skip. `harness/README.md` fixe la convention `harness/<famille>/` pour les harnais futurs (nemotron, diffusiongemma). | `harness/README.md`, `harness/dream/*` |
| **C2** | `.gitignore` : `harness/**/.venv/`, `harness/**/.fork/`, `harness/**/.cache/` (vérifié par `git check-ignore -v`). | `.gitignore` |
| **C3** | La logique **réellement utilisée** (le script `run_dream_evals2.sh` hors dépôt : entrelacement **par tâche** base/instruct, garde `SKIP`, `BATCH`/`LIMIT_*` en variables d'environnement) est récupérée et **reparamétrée** dans `harness/dream/run_evals.sh`. La copie V1 qui était dans le dépôt (sans entrelacement, sans skip) est remplacée. La **provenance de chaque jeu de résultats** (commande d'origine, version du fork, graines, horaires, effectif) est documentée dans `harness/dream/README.md`. | `harness/dream/run_evals.sh`, `harness/dream/README.md` |
| **C4** | L'environnement a été **reconstruit de zéro** par `setup.sh` (venv de 7 537 409 029 o dans le dépôt, fork au commit épinglé) puis **rejoué** : smoke **GSM8K n=8** sur `dream_7b_base`, **3 min 46 s**. **Contrôle d'identité des items** : les 8 items sont **identiques** (mêmes `doc_id` **et** mêmes questions) aux 8 premiers de la passe J6 à 300 ; mêmes graines (`0,1234,1234,1234`) ; même commit de fork (`31f94a6`). Scores sur ces 8 items : `flexible-extract` 0,625 / `strict-match` 0,375. **Temps d'import torch mesuré sur DrvFs** : 15,86 / 15,78 / 15,77 s contre 2,39 / 1,08 / 1,05 s sur le disque local (`/usr`, ext4) ⇒ **×15**. Le `du` du venv sur DrvFs prend **269 s**. | `harness/dream/setup.sh`, `_artefact/harness_setup.log`, `_artefact/smoke_c4/`, `harness/dream/README.md` |
| **C5** | **Inventaire en lecture seule** des écritures hors dépôt : `/home/sjteb/dream-harness` (7 588 593 361 o), `~/.cache/huggingface` (92 586 706 003 o), `~/.cache/pip` (8 869 011 825 o), `~/.cache/uv` (40 329 711 647 o), `~/.cache/vllm` (3 641 214 554 o), `~/.cache/electron`, `~/.cache/node-gyp`, `~/nltk_data`, `/tmp` (dont `tau_ref` 4 013 949 312 o), volumes et images Docker (`docker system df` : 258,6 Go d'images, 81,28 Go de cache de build, volume `qdrant_qdrant_data` 2,249 Go). **Tailles, propriétaires et dates** pour chaque entrée. **Commandes de suppression PROPOSÉES** (jamais exécutées), avec la liste de ce qu'il ne faut **pas** supprimer. | `docs/reports/OUT_OF_WORKSPACE.md` |
| **C6** | Les 6 dossiers de résultats à noms de chemin absolu (`__mnt__d__…`) sont renommés en `eval/dream_a2d/<alias>/<tâche>/` par un **script** (`scripts/restructure_eval_dirs.py`, avec `--dry-run`), qui **vérifie l'empreinte SHA-256 de chaque fichier avant et après** (contenu non modifié). Table de correspondance écrite dans `eval/dream_a2d/RESTRUCTURE_MAP.json`. `summarize.py` adapté à la nouvelle structure (clé de run `<modèle>/<tâche>`, plus de collision entre tâches). Les 6 scores sont **inchangés** après régénération. | `scripts/restructure_eval_dirs.py`, `eval/dream_a2d/RESTRUCTURE_MAP.json`, `harness/dream/summarize.py` |
| **C7** | `AGENTS.md` : règle **« Aucune écriture hors du dépôt »** (règle 9), convention **harness/ `<famille>/`** (règle 10), **« `git push` est réservé à l'utilisateur »** (rècle 11) ; structure du dépôt mise à jour. | `AGENTS.md` |

### D — Protocole d'évaluation rapide

| Tâche | Fait | Artefact |
|---|---|---|
| **D1** | `docs/design/EVAL_BUDGET.md` : **coûts mesurés**, jamais supposés — latence par tâche et par modèle (GSM8K 10,12 / 10,14 s·prompt ; MATH-175 34,45 / 34,22 ; IFEval 325,97 / 327,61), surcoûts fixes mesurés (amorçage ≈ 96 s par ajustement affine sur 3 points, import torch 15,8 s, chargement ≈ 25 s), projection vers les jeux complets (GSM8K 3 h 42 par modèle ; MATH 47 h 51 ; IFEval 48 h 59 — donc hors budget), tableau des leviers de vitesse, et la liste des leviers **non** disponibles avec la raison mesurée/lue dans le code. | `docs/design/EVAL_BUDGET.md` |
| **D2** | `docs/design/G1_PROTOCOL.md` : définition de G1 (décision N2-4), les 4 modèles (D_b, D_p, Transfer α=0,9, Merge λ=0,6 — coefficients de la Table 6), **réutilisation** des mesures D_b/D_p existantes avec **preuve mécanique** de l'identité des items (empreintes SHA-256 de `(doc_id, doc_hash)` et des `prompt_hash`, versionnées dans `eval/dream_a2d/ITEM_FINGERPRINTS.json` : **M0 et M1 ont déjà les mêmes empreintes** par tâche), protocole de décodage figé, **critères** : Transfer − Base ≥ **+10** (MATH-175) et ≥ **+8** (GSM8K) ; Merge − Instruct ≥ **−2** sur les deux ; **IC bootstrap apparié** rapportés (10 000 ré-échantillonnages, graine fixée) ; **IFEval complet exclu** du critère (sous-ensemble fixe de 150 prompts en tâche de nuit, non bloquant) ; **aucune revendication « ±3 reproduit »** ; budget **≤ 2 h par évaluation**, on réduit `n` et **jamais** les paramètres de décodage. | `docs/design/G1_PROTOCOL.md`, `scripts/eval_item_fingerprint.py`, `scripts/paired_bootstrap.py` |
| **D3** | **Mesuré** sur 16 prompts GSM8K, protocole figé : batch 1 → 291,7 s ; **2** → 273,7 ; **4** → 252,7 ; **8** → 244,2 ; **16** → 352,8 s. Gain maximal **×1,32** (batch 8) ; **batch 16 est plus lent** (×0,76). SDPA est **déjà actif** (`_supports_sdpa = True`) et `torch.compile` est **inaccessible** par le chargeur du fork : ces deux leviers ne sont pas disponibles, et le dire est plus utile que de les supposer. **Résultat le plus important de la mesure : à graines identiques et protocole identique, les sorties CHANGENT avec la taille de lot** (0 à 5 réponses identiques sur 16 entre le lot 1 et les lots 2/4/8/16) ⇒ **le batch fait partie du protocole effectif** (inscrit comme tel dans G1_PROTOCOL). | `docs/design/EVAL_BUDGET.md` §4, `_artefact/d3_sweep/` |

### E — Faisabilité G1b : Nemotron-Labs-Diffusion-3B (lecture seule)

| Tâche | Fait | Artefact |
|---|---|---|
| **E1** | Identifiants HF **vérifiés** : `mistralai/Ministral-3-3B-Base-2512` (apache-2.0, 7 698 241 056 o), `mistralai/Ministral-3-3B-Instruct-2512` (apache-2.0, 4 672 099 184 o), `nvidia/Nemotron-Labs-Diffusion-3B-Base` (licence Nemotron, 7 663 347 000 o), `nvidia/Nemotron-Labs-Diffusion-3B` (7 663 347 000 o + `linear_spec_lora/` 95 427 600 o). **`nvidia/Nemotron-Labs-Diffusion-3B-Instruct` N'EXISTE PAS** (HTTP 401, identique à un dépôt témoin inexistant) : le checkpoint « Instruct » est `nvidia/Nemotron-Labs-Diffusion-3B` **sans suffixe** (qualification étayée par `eos_token_id` 11 vs 2, `max_position_embeddings` 262 144 vs 4 096, RoPE factor 16 vs 0,25, gabarit ChatML). Aucun poids téléchargé. | `docs/design/G1B_NEMOTRON3B.md` §1 |
| **E2** | Inventaire des **237** tenseurs de diffusion (total 3 831 659 520 = `safetensors.total` de l'API, donc **validé**) et des **458** tenseurs AR ; **isométrie** vérifiée (squelette 3 429 006 336 des deux côtés) ; familles (embed 1, head 1, norms 53, attn 104, mlp 78, router 0, experts 0) ; brouillon de mapping App. A.2 en **14 règles** (236 paires, tête `diffusion_head` **conservée**, 222 tenseurs multimodaux AR **sans équivalent**) ; **lignes protégées 16 / 17 vérifiées sur les tokenizers** (14 ids divergents + ids 34-35 + ligne 100 = token de masque) ; point d'attention sur la **différence positionnelle** du Base (4 096 / 0,25 vs 262 144 / 16). | `docs/design/G1B_NEMOTRON3B.md` §2 |
| **E3** | Pile NeMo-Skills : installation `pip` (licence Apache-2.0), **Docker non requis** pour le chemin local (`--cluster=local`), **requis** pour le chemin SLURM (`eval.sh` exige enroot/pyxis) ; taille du dépôt ≈ 390 MiB ; image SGLang 17,1 Go ; **le `ns eval` stock ne supporte pas la diffusion** ⇒ voie officielle = client patché `xp/nemo-skills/eval_dlm.py` + `dlm_batch_server.py` (3 voies comparées). **Décodage AR avec cache : faisable et natif** (`ar_generate()`, `DynamicCache`, `cache_position` explicite, `use_cache=True` forcé) ; le modèle 3B publié est en `bidirectional` ⇒ chemin `sdpa`, pas de flex-attention. Coût AR vs diffusion : bande passante ÷ 5,9 ; 104 KiB/token de cache KV ; 1,1-1,9 GPU-h (médian) à **30 GPU-h** (pire cas 8 192 tokens) par modèle et par jeu en AR, **5,1 GPU-h** en diffusion. | `docs/design/G1B_NEMOTRON3B.md` §3-§4 |
| **E4** | Cibles chiffrées des Tableaux 1c, 17, 21 reportées (‖τ_A‖ 43,1 ; ‖τ_D‖ 17,1 ; ‖Δ_b‖ 280,9 ; ‖Δ_p‖ 271,4 ; cos(Δ_b,Δ_p) 0,987 ; cos(τ_A,τ_D) 0,075) avec la mention explicite : **non vérifiables** sur les configs lues (ce sont des propriétés de tenseurs), mais leur **protocole** est vérifiable et applicable (6 points listés). **Conclusion motivée** : **GO** pour la partie mécanique (23 Go + ≈ 10 min de calcul, plan de fusion isométrique et non ambigu) ; **NO-GO** pour l'évaluation complète sur cette machine (≈ 315 h AR / ≈ 53 h diffusion par benchmark de 541 prompts, × 10 benchmarks ; `ns eval` sans support diffusion ; dépendance `transformers>=5.0.0`) ; **conditions de réouverture** listées. | `docs/design/G1B_NEMOTRON3B.md` §5-§6 |

---

## 2. Hypothèses non prouvées

* **La « experimental formula » de l'App. A.2** (que les adaptateurs DG restent
  non fusionnés et sont mis à l'échelle) : l'App. A.2 **renvoie** à cette
  formule **sans la=redonner**. La reconstruction `D_Merge(λ) = [D_b + λτ_A] +
  (1−λ)·adaptateurs` est **cohérente** avec l'équation (2) et avec
  `D_p = D_b + adaptateurs`, mais elle n'est **pas** citée mot pour mot dans le
  PDF. → `docs/design/G2_adapter_scaling.md` H1.
* **Le facteur `(1−λ)` s'applique-t-il à l'ensemble des adaptateurs** (décodeur +
  encodeur + self-conditioning) ? L'App. A.2 ne le détaille pas. → H2.
* **Le papier met-il à l'échelle `lora_alpha` ou les poids `B` ?** Non précisé ;
  sans impact numérique au-delà de l'arrondi BF16. → H3.
* **Les deux jeux de l'adaptateur DG (encodeur et décodeur) sont-ils appliqués
  simultanément à l'inférence ?** Déduit de la structure du modèle (poids
  partagés) et de l'adaptateur publié (413 modules) ; non stated dans le PDF. → H4.
* **Le TPF (tokens par forward) du Nemotron-LD-3B** : le 5,89 rapporté par
  l'outil `evaluate.py` de NVlabs est mesuré sur **8B**, pas sur 3B. Tous les
  coûts de la section E3 sont des **projections** `[INFERENCE]`.
* **La taille de l'image Docker NeMo-Skills** n'est pas publiée par NVIDIA.
* **Les formes exactes** des tenseurs de diffusion et du `vision_tower` /
  `multi_modal_projector` : dérivées de `config.json` (`[INFERENCE]`), jamais
  lues dans un en-tête safetensors (interdit par la consigne « lecture seule »).
* **La reproductibilité des sorties** : à graine et protocole identiques, les
  sorties de Dream-7B **dépendent de la taille de lot** (mesuré). La cause
  (ordre de consommation du germe, padding, longueur commune) est une
  `[INFERENCE]`.
* **L'amorçage de ≈ 96 s** est déduit d'un ajustement affine sur trois points
  (8, 16 et 300 prompts) et non mesuré directement ; les temps bruts sont, eux,
  mesurés.
* **L'effet du batching sur le score** : sur 16 prompts, les scores varient
  (0,4375 → 0,5000) ; l'amplitude réelle sur 300 items n'est pas mesurée.

---

## 3. Non exécuté

* **J7** (transfert α=0,9 et merge λ=0,6 réels sur Dream-7B, puis évaluation) :
  **non exécuté** — explicitement hors périmètre de J6bis (« la review N2 de
  J6bis le précède »). Aucun checkpoint fusionné n'a été produit ni évalué.
* **G2** (DiffusionGemma médical) : **non démarré**. Le mode « base + adaptateur
  remis à l'échelle » n'est pas implémenté : B7 est une **conception**.
  Les adaptateurs publiés (2,68 Go) n'ont **pas** été téléchargés.
* **G3** : **non démarré**.
* **G1b** : **non démarré** (étude de faisabilité seulement ; aucun poids
  téléchargé).
* **IFEval complet** (541 prompts par modèle) : non exécuté — coût mesuré
  ≈ 49 h par modèle.
* **MATH-500 / test MATH complet** : non exécuté — coût mesuré 47 h 51 par
  modèle.
* **Suppressions hors dépôt** : **aucune**. Les commandes de la section 3 de
  `docs/reports/OUT_OF_WORKSPACE.md` sont des **propositions** en attente
  d'accord écrit.
* **`git push`** : **jamais effectué** par l'agent. Les 5 commits de la
  mission sont **locaux** (`origin/main` est resté sur `284fb62`).
* **Évaluation d'un modèle fusionné** : aucune (le seul artefact de fusion de
  J6 — transfert α=0,9 sur Dream-7B-Base — reste **mécanique**, sans
  évaluation ; les nouveaux modèles n'ont pas été fusionnés).
* **Adaptateur DG appliqué à un modèle** : non fait (aucun téléchargement).

---

## 4. Écarts vs le prompt J6bis

| # | Demande | Réalisé | Écart / motif |
|---|---|---|---|
| E1 | A1 : « le corps de l'audit N1 reste intact » | L'ERRATUM est inséré **entre le titre et le bloc « Date »** ; aucune ligne du corps n'est modifiée | conforme |
| E2 | B4 : « régression des moindres carrés de (out − ref) sur [τ_A, τ_D] » | Réalisé **avec un seul régresseur** dans les modes à une seule source (transfert : [τ_A] ; ablation-minus : [τ_D]) : un régresseur absent n'a pas de projection signée à calculer | Le mode merge (2 régresseurs) est le cas complet ; les modes à source unique sont traités par la projection simple, qui est la même formule |
| E3 | B4 : « Compare (c_a_eff, c_d_eff) SIGNÉS aux (c_a, c_d) attendus du mode » | Le coefficient **sans régresseur** est rapporté `n/a` (il n'est pas estimable) ; les deux coefficients sont estimés dès que les deux sources existent | conforme pour le cas à deux sources ; l'impossibilité d'estimer un coefficient absent est explicite dans la sortie |
| E4 | B1 : « défaut strict pour deux sources de checkpoints complets » | `strict` est le défaut **quand le mode combine deux sources** (c'est le seul cas où l'union a un sens). Pour un mode à source unique, la couverture est trivialement complète | Choix explicité dans `validate_config` et dans `.cursorrules` |
| E5 | C4 : « Mesure et rapporte le temps d'import de torch sur DrvFs » | 15,86 / 15,78 / 15,77 s (DrvFs) contre 2,39 / 1,08 / 1,05 s (ext4) ⇒ **×15** | conforme |
| E6 | C6 : « via un script et un fichier de correspondance » | `scripts/restructure_eval_dirs.py` + `eval/dream_a2d/RESTRUCTURE_MAP.json` (empreintes SHA-256 de chaque fichier) | conforme ; le contenu des `.json`/`.jsonl` n'est **pas** modifié (les chemins absolus qu'ils contiennent sont la provenance du run) |
| E7 | C5 : « tailles et propriétaire » | Tailles `du -sb` et propriétaires `stat` pour chaque entrée | conforme |
| E8 | D3 : « Rapporte l'accélération et si les sorties à graine égale sont identiques » | Accélération mesurée (×1,32 max) **et** identité des sorties mesurée (**non identiques** : 0-5/16) | conforme ; le résultat est défavorable et est reporté tel quel |
| E9 | E2 : « brouillon de mapping selon A.2 » | Brouillon en 14 règles, + le **masque des 16/17 lignes protégées** (extension de `MergePolicy` à prévoir) | conforme |
| E10 | E4 : « Conclusion GO / NO-GO motivée avec coûts » | GO (mécanique) / NO-GO (évaluation) avec les coûts projetés | conforme ; la conclusion est **partielle** (GO sur un volet, NO-GO sur l'autre), ce qui est la lecture honnête des données |
| E11 | R5 : « aucun job GPU de plus de 60 minutes » | 6 jobs GPU : reconstruction + smoke (3 min 46), balayage D3 (5 runs de 4 min 42 à 6 min 31) | conforme ; le plus long job dure **6 min 31 s** |
| E12 | B6 : « La CI les exécute » | Étape `Tests adversariaux (mutations)` + étape `Modes A2D non couverts par le test unitaire (CLI)` (4 cas) | conforme ; les **étapes ajoutées ont été rejouées localement avec les commandes identiques** (ablation-minus, refus sans τ_D, refus τ_D partiel en `strict`, `zero-fill` déclaré) ; le workflow reste un YAML valide (12 étapes) |
| E13 | A2 : « MAJ-06 reformulé (évaluation = harnais et références ; bibliothèque = vérifiée mécaniquement seulement) » | Formulation retenue dans `README.md`, `docs/reports/J6.md` §1.1 et le tableau des portes | conforme |
| E14 | A2 : « MIN-01 à MIN-07 traités » | MIN-01, 02, 03, 05, 06, 07 traités ; **MIN-04** traité par le harnais (commande exacte et graine consignées dans `harness/dream/README.md` ; le YAML `ifeval_a2d.yaml` reste tel quel, l'écart `do_sample: false` vs `temperature=0.1` étant signalé dans le rapport) | MIN-04 partiellement traité : le YAML n'a pas été aligné (il décrit la tâche, le run surcharge par `--model_args`) |
| E15 | A1 : « bilan corrigé = 2 bloquants, 7 majeurs, 8 mineurs » | Écrit tel quel dans l'ERRATUM | conforme |

---

## 5. Désaccords

* **Avec l'audit N1 (`docs/reports/AUDIT_J6.md`) sur MAJ-07** : la review N2
  **invalide** ce constat. Le rapport J6 affirmait « aucun push git effectué »
  alors que `HEAD == origin/main` ; l'affirmation était **exacte à la date de
  rédaction** (les 14 commits étaient locaux) et le push a été effectué **par
  l'utilisateur après la livraison**. Le reflog n'identifiant pas l'auteur d'un
  push, aucune attribution n'est faite. Le corps de l'audit N1 n'a **pas** été
  réécrit : la correction passe par l'ERRATUM N2 en tête et par la note factuelle
  de `docs/reports/J6.md`.
* **Avec le rapport J6 sur la formulation « C0.5 atteint »** : la version
  initiale affirmait « *C0.5 est donc atteint sur GSM8K et MATH-500* » et
  « *la porte G1 reste ouverte sur ce point* » (§4 E2) — les deux énoncés ne
  peuvent pas être vrais simultanément, et l'argument « la cible est dans l'IC »
  est une non-rejection. La version corrigée dit **NON ATTEINT** et renvoie à
  `docs/design/G1_PROTOCOL.md`.
* **Avec le rapport J6 sur « MATH-500 »** : la version initiale annonçait
  « *MATH-500* » pour un jeu de **175** problèmes ; la métrique est renommée
  **MATH-175** dans tout le dépôt (la cible de la Table 16 reste MATH-500 et
  l'écart est signalé comme « périmètre différent »).
* **Avec le README de J6 sur « la bibliothèque de fusion est validée »** :
  remplacé par « validée **mécaniquement** seulement ».
* **Avec le critère C0.5 du prompt J6 (±3 pts, IFEval complet)** : la review N2
  l'a **remplacé** (N2-4) par le protocole `docs/design/G1_PROTOCOL.md`, parce
  qu'il est irréaliste en coût sur cette machine (coûts mesurés : IFEval
  complet ≈ 49 h par modèle) et parce que « ±3 » n'est pas démontrable par un
  IC qui *contient* la cible.
* **Avec le « guideline » de `GUIDE_RESTANT.md`** : le guide est conservé pour
  l'historique mais **marqué obsolète** en bannière (il décrivait un pipeline
  hors papier et une publication de checkpoint).
* **Avec la bibliothèque J6 sur la couverture des sources** : la J6 fusionnait
  τ_D **conditionnellement** (`if cfg.tau_d.has(module)`), ce qui transformait
  un merge en transfert sans alerte. C'est maintenant une **erreur** en
  politique `strict` — y compris pour le cas réel G2 (413 modules d'adaptateur
  contre 205 côté AR), qui sera donc **refusé** tant que le mode « adaptateur à
  l'échelle » n'existe pas (c'est la raison du blocage de G2).

---

## 6. Preuves d'état

### 6.1 `git status --short`

```
$ git status --short
```

*(sortie vide : arbre de travail propre, aucun fichier non suivi)*

### 6.2 `git log origin/main..HEAD`

```
$ git log --oneline origin/main..HEAD
8b1132f docs(design): protocole G1, budget eval, G2 adapter scaling, G1b Nemotron-3B
9ae02a1 feat(harness): harnais Dream versionne dans le depot + resultats a noms portables
b57f29e feat(a2d): plan sur l'union tau_A/tau_D, verification v2 signee, tests de mutation
1757bf7 docs(J6bis A2/A3): requalification factuelle de J6 + statut des portes
a1b2b46 docs(review): verdict N2 sur J6 (REFUSÉ, G1 ouverte) + ERRATUM N2 en tête de l'audit N1
```

→ **5 commits locaux, non poussés.** `origin/main` est resté sur `284fb62`.

### 6.3 Lignes « update by push » du reflog `origin/main`

```
$ git reflog show origin/main --date=iso | grep "update by push"
284fb62 refs/remotes/origin/main@{2026-10-20 11:24:02 +0200}: update by push
c8b4e1c refs/remotes/origin/main@{2026-10-20 04:26:16 +0200}: update by push
e4d6074 refs/remotes/origin/main@{2026-10-20 02:08:59 +0200}: update by push
5a4fcf3 refs/remotes/origin/main@{2026-10-19 21:43:50 +0200}: update by push
7799460 refs/remotes/origin/main@{2026-10-19 19:59:31 +0200}: update by push
1656263 refs/remotes/origin/main@{2026-10-19 12:11:40 +0200}: update by push
a0c9d7c refs/remotes/origin/main@{2026-10-19 02:59:18 +0200}: update by push
af4f939 refs/remotes/origin/main@{2026-10-18 21:16:47 +0200}: update by push
5a8577b refs/remotes/origin/main@{2026-10-18 07:42:31 +0200}: update by push
16b6efa refs/remotes/origin/main@{2026-10-16 21:14:50 +0200}: update by push
c7d084f refs/remotes/origin/main@{2026-10-16 19:11:56 +0200}: update by push
12c3d47 refs/remotes/origin/main@{2026-10-16 17:36:04 +0200}: update by push
4a91967 refs/remotes/origin/main@{2026-10-16 16:52:29 +0200}: update by push
4942857 refs/remotes/origin/main@{2026-10-16 16:41:30 +0200}: update by push
9d90a41 refs/remotes/origin/main@{2026-10-16 16:20:03 +0200}: update by push
5f4a0ae refs/remotes/origin/main@{2026-10-16 16:16:47 +0200}: update by push
```

→ **Aucun push** de cette mission (les 5 commits J6bis sont absents du reflog).
Les 16 lignes ci-dessus sont **antérieures** à J6bis ; la dernière (`284fb62`,
11:24:02) est le push du rapport J6, effectué **par l'utilisateur** après la
livraison de J6. Le reflog **n'identifie pas l'auteur** d'un push : aucune
attribution n'en est déduite.

### 6.4 Sortie complète de `pytest`

```
$ python3 -m pytest scripts/tests -q
.................................................................        [100%]
65 passed in 31.04s

$ python3 -m pytest vLLM/scripts/tests -q
.................                                                        [100%]
17 passed in 2.47s
```

* 65 = 29 (A2D, `test_a2d_library.py`) + 21 (mutation, `test_a2d_mutation.py`)
  + 9 (`test_pipeline.py`) + 6 (`test_eval_compare.py`).
* Les 21 tests de mutation couvrent : inversion de signe · permutation α↔λ ·
  permutation τ_A↔τ_D · coefficient faux · τ_D partiel (`strict` refusé /
  `zero-fill` déclaré) · τ_A absent · plan vide (refusé / `--allow-empty`) ·
  tenseur non fusionné altéré · mode erroné · lignes d'embedding intersectées ·
  seuils ULP BF16 (1 ULP passe, 2 ULP échoue) · couverture LoRA explicite ·
  couverture au manifeste ; famille couverte par une seule source
  (`zero-fill`) — vérifiée et non ignorée · garde-fou « coefficient nul »
  (refus par défaut, contournement `--allow-empty-coefficient` tracé au
  manifeste).

### 6.5 Ce qui existe hors du dépôt

Inventaire complet et **commandes de suppression proposées** (jamais
exécutées) : `docs/reports/OUT_OF_WORKSPACE.md`. Résumé des tailles mesurées :

| Chemin | Taille (o) |
|---|---:|
| `/home/sjteb/dream-harness/` (harnais J6 hors dépôt) | 7 588 593 361 |
| `/home/sjteb/.cache/huggingface/` | 92 586 706 003 |
| `/home/sjteb/.cache/uv/` | 40 329 711 647 |
| `/home/sjteb/.cache/pip/` | 8 869 011 825 |
| `/home/sjteb/.cache/vllm/` | 3 641 214 554 |
| `/home/sjteb/nltk_data/` | 15 204 406 |
| `/tmp/tau_ref/` | 4 013 949 312 |
| `/tmp/pytest-of-sjteb/`, `/tmp/a2d/`, `/tmp/dummy_*`, `/tmp/dgsrc` | ≈ 673 000 |
| Docker : images 258,6 Go · cache de build 81,28 Go · volume `qdrant_qdrant_data` 2,249 Go | — |

**Depuis J6bis**, le harnais n'écrit **plus** hors du dépôt : `HF_HOME`,
`HF_DATASETS_CACHE`, `HUGGINGFACE_HUB_CACHE`, `PIP_CACHE_DIR`, `XDG_CACHE_HOME`
et `TMPDIR` pointent vers `harness/dream/.cache/` (1 241 003 141 o), le venv
(7 537 409 029 o) et le fork (51 106 809 o) sont dans le dépôt.

### 6.6 État des portes après J6bis

| Porte | État | Raison |
|---|---|---|
| **G1** | **OUVERTE** | mécanique vérifiée (v2), baseline ré-exécutée dans un harnais reconstruit, protocole de franchissement écrit ; **critère de reproduction ±3 non atteint** ; aucun modèle fusionné évalué |
| **G1b** | **NON DÉMARRÉE** | faisabilité instruite (GO mécanique / NO-GO évaluation) |
| **G2** | **BLOQUÉE (B7)** | mode « base + adaptateur remis à l'échelle » à implémenter |
| **G3** | **NON DÉMARRÉE** | — |

---

## 7. Post-audit N1 (2026-10-20) — corrections F1–F7

**Audit** : `docs/reports/AUDIT_J6BIS.md` (N1, agent OpenCode, lecture seule, 2026-10-20).
Verdict : **aucune déviation silencieuse** des équations, coefficients, mappings ni du schéma
numérique A2D n'a été trouvée sur le chemin Qwen2.5-7B → Dream-7B. Les constats F1–F9 sont
des défauts de **documentation**, de **garde-fous CLI** et d'**inventaire** ; **aucun artefact de
fusion déjà produit n'est affecté** (le run réel de J6 passait `α=0,9` explicite).

| Constat | Gravité | État | Correction |
|---|---|---|---|
| **F1** | modérée | **corrigé** | `docs/design/G1_PROTOCOL.md` §4.1 : **+19,4** = Dream-7B MATH-500 **transfert** (20,8 → 40,2 ; Figure 2a, Tableaux 1a/16) ; **+15,2** = Dream-7B GSM8K **transfert** (66,5 → 81,7 ; Figure 2a) ; plus grand gain de **merge** Dream = **+9,8** (IFEval 63,0 → 72,8 ; MATH-500 +6,6 ; GSM8K +0,7) ; gains de transfert Nemotron (Table 8) ≈ **+29 à +37** (IFEval). Seuils **C1/C2 inchangés**. Idem la mention équivalente de `docs/design/G1B_NEMOTRON3B.md`. |
| **F2** | **élevée** | **corrigé** | `scripts/a2d/merge.py` (`evaluate_coefficient_guard`, `first_nonzero_delta`) : un transfert à `α=0` ou un merge à `λ=0` est **refusé** (sortie 2) tant que le plan contient des vecteurs non nuls — le no-op écrivait une copie de `D_b` / `D_p`, indiscernable d'un modèle non fusionné (la vérification compare les coefficients effectifs aux coefficients **demandés**, 0). `--allow-empty-coefficient` assume le no-op (avertissement au manifeste, état dans `coefficient_guard`). Wrapper legacy `merge_jainius.py` : comportement observable inchangé (opt-in uniquement à `α=0`). |
| **F3** | modérée | **corrigé** | `--no-verify` (conservé) inscrit `"post_write_verification": {"skipped": true, "reason": "--no-verify"}` au manifeste. |
| **F4** | faible | **corrigé** | Compteurs de tests : **65** au total (29 A2D + 21 mutation + 9 pipeline + 6 eval-compare) — `KNOWN_ISSUES.md`, `CHECKLIST.md`, `J6bis.md` (ligne A2, B6, §6.4). |
| **F5** | faible | **corrigé** | Latence IFEval harmonisée sur le taux **batch 4** mesuré (325,97 / 327,61 s/prompt ⇒ ≈ **49 h** pour le jeu complet par modèle) : `J6.md` (§1.1, §3, E1, §5), `harness/dream/summarize.py` (+ `eval/dream_a2d/SUMMARY.{md,json}` régénérés), `eval/dream_a2d/SUMMARY.md`. `docs/design/EVAL_BUDGET.md` : « quasi linéaire » retiré (les facteurs ×3,4 / ×32 comparent des **tâches différentes**) ; MATH-500 : 5 000 → 500 = **×10** ⇒ **4 h 47** par modèle (**9 h 34** pour les deux). |
| **F6** | faible | **corrigé** | `docs/reports/OUT_OF_WORKSPACE.md` : `/tmp/SUMMARY_test.{json,md}` (48 867 / 2 814 o, 2026-10-20 13:03), `/tmp/a2d_smoke_transfer` (checkpoint de transfert α=0,9 de J6, ≈ 15 Go, **absent du disque**) et la taille à jour de `/tmp/pytest-of-sjteb` (**8 483 981** o). **Aucune suppression.** |
| **F7** | faible | **corrigé** | Aide CLI de `--lambda` : seuil d'avertissement **1.0** (`LAMBDA_MAX`), conforme à l'eq. (2) bornée dans [0,1]. |
| **F8** | info | **déclaré** | `eval/dream_a2d/ITEM_FINGERPRINTS.json` : `git_hash` = commit du **fork Dream** (`31f94a6`), pas un commit d'Academiia. Non corrigé (renommage optionnel). |
| **F9** | info | **déclaré** | `do_sample: false` / `temperature: 0.0` dans les YAML de tâche vs `temperature=0.1` imposé par `--model_args` : déjà signalé (J6bis E14, `docs/reports/AUDIT_J6.md` MIN-04). |

**Suite de tests après corrections** : `python -m pytest scripts/tests -q` → **65 passed**
(29 `test_a2d_library.py` + 21 `test_a2d_mutation.py` + 9 `test_pipeline.py` + 6
`test_eval_compare.py`).

**Commits** (locaux ; le push est réservé à l'utilisateur) :

| Commit | Contenu |
|---|---|
| `c80670f` | `docs(audit)` : versionner `AUDIT_J6BIS` (N1 OpenCode, 2026-10-20) |
| `42124bc` | `fix(cli)` : seuil d'avertissement `--lambda` = 1.0 (F7) |
| `ca3deef` | `fix(a2d)` : refuser transfert/merge à coefficient nul par défaut (F2) |
| `a61d298` | `fix(a2d)` : tracer `--no-verify` dans le manifeste (F3) |
| `88faf77` | `docs(design)` : corriger les ancrages +19,4 / +15,2 (F1) |
| `9f342ab` | `docs` : corriger les compteurs de tests (65 / 21 mutations) (F4) |
| `dc64315` | `docs` : harmoniser latences IFEval et projections (F5) |
| `619a6c6` | `docs` : compléter l'inventaire C5 (F6) |
