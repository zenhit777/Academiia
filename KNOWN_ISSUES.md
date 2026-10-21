 # KNOWN_ISSUES.md — Anomalies du checkpoint Jainius (rebuild v2 en place, 2026-10-17)

## 🆕 Mission J8 (2026-10-21) — clôture scientifique de G1, bibliothèque durcie, legacy déplacé

- **Bilan G1** : `docs/reports/G1_BILAN.md` — verdict **G1 NON FRANCHIE** inchangé (aucun seuil
  rejoué). Nouvelle mesure : M2 épuise le budget MATH de 512 tokens sur **82/175** générations
  (67 sans réponse finale) ; sur ces 67 items M2−M0 = **−2,99** [−11,94 ; +5,97], contre
  **+17,59** [+9,26 ; +26,85] sur les 108 autres ⇒ l'effet C1 est **dilué par le plafond de
  génération**, pas absent. Diagnostic 1 024 tokens : **annulé sur coût mesuré** (314 s/item,
  5,8 h/modèle), aucun résultat — `docs/reports/G1_BILAN_1024.md`, `eval/dream_a2d_trunc/`.
- **Bibliothèque (phase B)** : mode `merge-dg-scaled` (alias `adapter-scale`) pour DiffusionGemma
  (adaptateur τ_D **non fusionné**, `lora_B × (1−λ)`, vérification V5/V6) ; `max_ulp = 0` par
  défaut (1 ULP = échec) ; merge sans τ_D = **erreur dure** (sortie 2) ; libellés de manifeste
  désambiguïsés.
- **Legacy déplacé (phase C)** : `merge_jainius.py`, `compute_tau_a.py`, `measure_effect.py`
  sont dans **`legacy/scripts/`** ; `AUDIT.md` et `GUIDE_RESTANT.md` dans **`legacy/docs/`**.
  Les entrées **historiques** ci-dessous conservent les anciens chemins (`scripts/...`) : ce sont
  des traces datées, pas des commandes à rejouer. Les commandes actives sont dans `README.md`,
  `AGENTS.md` et `CHECKLIST.md`.

## 🆕 Mission J7 (2026-10-21) — porte G1 : **NON FRANCHIE**

- **Verdict G1 : la porte reste OUVERTE** (règle de franchissement §1 de
  `docs/design/G1_PROTOCOL.md` : un seul point en échec ⇒ G1 non franchie).
  Sur les quatre critères amendés (M1) : **C2 atteint** (transfert − base GSM8K : **+14,33** pts,
  IC [+9,33 ; +19,67] ; le papier rapporte **+15,2** sur la même paire) ; **C1 non atteint**
  (transfert − base MATH-175 : **+9,71** pts, IC [+2,86 ; +16,57] — **0,29 pt sous le seuil de +10**,
  IC excluant 0) ; **C3/C4 non établis** (merge − instruct : −2,29 [−9,14 ; +4,57] et
  +1,67 [−2,33 ; +5,67] : **la dégradation de 2 points ne peut pas être exclue**).
  Aucun seuil n'a été rejoué après les résultats. Rapport : `docs/reports/J7.md`.
- **Défaut corrigé — les checkpoints fusionnés n'étaient PAS chargeables** : les shards écrits
  par la bibliothèque n'avaient pas de bloc `__metadata__` (safetensors), alors que
  `transformers.load_state_dict` l'exige (`format == "pt"`) → `AttributeError: 'NoneType' object
  has no attribute 'get'` au chargement. La **vérification mécanique passait** (`post_write_verification`
  v2, `verify_merge` VERDICT OK) : le défaut n'a été trouvé que par le **smoke de chargement**
  (`from_pretrained` + génération). Corrigé le 2026-10-21 (`4f1b7d6`) : le bloc `__metadata__` est
  **repris du shard source** ; test de non-regression dans `scripts/tests/test_a2d_library.py`.
  **Leçon** : une fusion doit être validée par un *smoke de chargement*, pas seulement par la
  vérification mécanique.
- **Bruit de décodage : nul à lot 4** (mesuré, M2 de la revue N2). M1 rejoué sur GSM8K n=300
  avec une **graine torch différente** (0,1234,**1234**,1234 → 0,1234,**4321**,1234) : les
  **300 réponses brutes sont byte-identiques** (SHA-256 global identique), précision 84,00 % dans
  les deux runs, écart apparié **+0,00 pt**, IC [+0,00 ; +0,00]. À lot 4 et protocole figé, le
  décodage de Dream-7B est **déterministe** ; la variabilité réelle documentée par le balayage D3
  (`docs/design/EVAL_BUDGET.md` §4.1) vient de la **composition du lot**, pas du RNG. Les
  incertitudes des verdicts G1 sont donc **purement d'échantillonnage**.
- **Outil de verdict : mélange d'unités corrigé** (`484b152`). Les écarts et bornes de l'IC sont
  des proportions (0,1433 = +14,33 pts) alors que les seuils sont en points : la comparaison
  proportion contre points classait à tort C2 « non atteint » et C3/C4 « non réfutés ». Verrouillé
  par `scripts/tests/test_g1_verdict.py` (7 tests). Le verdict affiché **avant** ce correctif était
  faux — il n'a jamais été publié (le rapport et le manifeste n'existent qu'après).
- **Audit J7 (2026-10-21)** : rapport `docs/reports/AUDIT_J7.md` — directives suivies,
  **aucune déviation silencieuse A2D**, verdict `G1 NON FRANCHIE` reproduit. Cinq écarts de
  forme/traçabilité : quatre corrigés dans `28e8410` (libellé C3/C4, portée des écritures hors
  dépôt, docstring du plancher de bruit, renvoi vers la reconstitution N2) ; le cinquième
  (sortie `verify_merge` M3 non archivée) est documenté sans changement.

## 🆕 Mission J6 (2026-10-20) — bibliothèque A2D conforme + baseline Dream-7B

- **Bibliothèque A2D** (`scripts/a2d/`, `scripts/merge_a2d.py`, `scripts/verify_merge.py`) :
  transfert (eq. 1), merge (eq. 2), ablations `D_p + ατ_A` (Table 9) et `D_p − λτ_D` (Table 10) ;
  τ **calculé à la volée** (paire de checkpoints ou adaptateur LoRA), **jamais persisté** ;
  FP32 + un seul cast BF16 ; écriture atomique ; **vérification post-écriture obligatoire**
  (`α_eff`/`λ_eff`, échec dur hors tolérance). 22 tests ajoutés (`scripts/tests/test_a2d_library.py` —
  29 aujourd'hui) ; 65 tests au total dans `scripts/tests` (rectifié J6bis : MIN-05 — la valeur « 19 »
  était fausse ; rectifié J6ter : la valeur « 37 » était également fausse — 29 A2D + 21 mutation
  + 9 pipeline + 6 eval-compare).
- **Corrections documentaires (A2)** :
  - « τ_D inexistant publiquement » est **faux** : `τ_D = D_p − D_b` est défini (§2.1) et, pour
    DiffusionGemma, c'est le delta LoRA de post-training **publié**
    (`gevaertlab/diffusiongemma-radiology-vqa`, r=64/α=128, experts/routeur/vision gelés —
    vérifié dans `docs/VP_ADAPTERS.md`). Ce qui manque est la chaîne complète, pas la définition.
  - « routeur inchangé » était **faux** : le build legacy fusionne 90 tenseurs de routeur
    (355 = attn 115 + mlp 90 + experts 60 + routeur 90). Le routeur est une famille à part,
    fusionnée par défaut et **déclarée** dans le manifeste.
  - « Tab. 9 » : l'ablation `D_p + ατ_A` est bien la Table 9 du papier, mais le setup du
    papier y suppose un `τ_A` **médical** (IT → médical) ; notre build utilise un `τ_A`
    **général** (base → IT) ⇒ **ablation hors papier**, pas une reproduction.
  - `D_b` pour DiffusionGemma = le checkpoint **IT** (aucun base diffusion public).
  - J5 est marqué **non conforme** au protocole du papier (FP8, `temperature=0`, 128 tokens,
    103 prompts de texte, sans juge) : c'est un contrôle de non-régression, pas une mesure.
- **Géométrie des vecteurs (Table 21) reproduite** sur Qwen2.5-7B → Dream-7B (337 tenseurs du
  backbone) : ‖τ_A‖ 23.01 (23.0), ‖τ_D‖ 31.47 (31.5), ‖Δ_b‖ 729.04 (729), ‖Δ_p‖ 732.14 (732),
  cos(Δ_b,Δ_p) +0.9986 (+0.999), cos(τ_A,τ_D) +0.0130 (+0.013) — écarts ≤ 0,11 %.
  Outil : `scripts/measure_geometry.py`.
- **Baseline Dream-7B** : checkpoints téléchargés (`Qwen/Qwen2.5-7B[-Instruct]`,
  `Dream-org/Dream-v0-{Base,Instruct}-7B`, identifiants HF vérifiés le 2026-10-19) ; harnais
  d'évaluation = fork lm-eval de Dream (`DreamLM/Dream`, `eval_instruct/lm_eval`) dans un venv
  **séparé** (`requirements-dream.txt`). Résultats partiels et écarts : `docs/reports/J6.md`.

## ✅ Suite audit Sonnet 2026-10-16 (appliquée, en attente du balayage α)

- **Méthode renommée (formulation 2026-10-16, révisée le 2026-10-20) :** ce n'est PAS le
  A2D-Merge (`D_p + λ(τ_A − τ_D)`) mais l'ablation `D_p + ατ_A` — **hors papier**, cf. la
  section J6 ci-dessus (τ_A général au lieu d'un vecteur de domaine, α non sélectionné sur
  ancre). README/AGENTS corrigés, défaut **α=0.3** (0.8 déprécié : −3.8 pts SLAKE dans le papier).
 - **Parseur maison en retrait :** `compute_tau_a.py` + `merge_jainius.py` lisent via `safe_open`, helpers partagés `scripts/_common.py`. Le merge sauvegarde déjà en BF16 canonique.
- **Merge durci :** shape mismatch → erreur par défaut, manifest `merge_config.json`, écriture atomique, nettoyage shards obsolètes (`--overwrite` exigé), recopie de TOUS les sidecars (LICENSE...), `total_size` dans l'index. `verify_checkpoint.py` : égalité index/headers exigée, contrôle bon-fichier par tenseur, `--check-values` NaN/Inf, HF files requis.
- **Serving :** compose en forme shell (multi-flags `VLLM_EXTRA_ARGS` OK), image épinglée `vllm/vllm-openai:gemma`, `ipc: host`, `VLLM_MAX_MODEL_LEN=4096`, `MAX_NUM_SEQS=1`, `GPU_MEMORY_UTILIZATION=0.90`. `.gitignore` : `__pycache__/`, `*.pyc`, `.venv/`, `*.npy`.
 - **Mesure poids (résultats 2026-10-17, v2) :** `scripts/measure_effect.py` sur les 656 tenseurs (25.2B éléments) — % reellement modifies apres arrondi BF16 : α=0.05→94.8%, 0.1→97.3%, 0.2→98.5%, **0.3→99.1%**, 0.5→99.4%. Variante sans norms/embed (355 tenseurs) quasi identique. Par famille a 0.3 : attn ratio_rel med=2.98%, cos=0.998 ; mlp/experts ratio_rel med=2.82%, cos=0.998 ; embed ratio_rel med=2.66%, cos=0.998 ; norms ratio_rel med=0.044%, cos=0.61 (arrondi BF16 detruit la direction) ; routeur τ≈0. Le % d'elements modifies NE mesure PAS l'ampleur du changement : un ULP BF16 minuscule suffit a faire "bouger" des bits sans effet numerique significatif. La quantification n'absorbe presque rien — α pilote reellement le merge, d'où l'importance du balayage et de l'éval vs `D_p`. JSON : `_artefact/measure_effect_v2*.json`.
 - **2026-10-17** : `merge_jainius.py` — défaut exclut norms/embeddings (opt-in `--include-norms-embed`) ; suppression index/manifest perimes au démarrage du merge ; contrôle manifest τ (exit 2) + `--allow-missing-tau-manifest` ; README.md de destination préservé. `compute_tau_a.py` — défaut `--filter decoder`. `verify_checkpoint.py` — option `--reference <source>` vérifie noms/shapes/dtypes/sidecars. `scripts/tests/test_pipeline.py` : 9 tests (pytest).
- **Reste :** **test d'inférence GPU → fait le 2026-10-19** (voir section « Inférence GPU vLLM (2026-10-19) ») ; **pas de fichier `LICENSE`** — les repos sources (`google/gemma-4-26B-A4B`, `google/gemma-4-26B-A4B-it`, `google/diffusiongemma-26B-A4B-it`) n'en publient aucun (vérifié le 2026-10-19) : rien à recopier, la model card déclare `apache-2.0` + `license_link` vers les conditions Gemma ; **`_artefact/tau_a` est vide depuis le 2026-10-19 (~18:34)** — les `.npy` du calcul du 2026-10-16 ont été consommés par la refusion α=0.3 puis nettoyés ; un recalcul sur la VM actuelle (30 GiB) échoue en ENOMEM (voir section « Blocage mmap/RAM »).

## ✅ Resolu dans les scripts (2026-10-16)


### 2. Decodage BF16 faux + offset de lecture faux (corriges)
- **Cause decode :** `np.frombuffer(raw, uint16).astype(float32)` donne la valeur entiere du bit-pattern (ex. `0x3F80` → `16256.0` au lieu de `1.0`).
- **Cause offset (critique, ratee au premier passage) :** `f.seek(offset)` utilisait l'offset relatif aux donnees comme position absolue, sans ajouter `8 + header_len`. Toutes les lectures etaient decalees (~8 Ko), y compris le tenseur a offset 0 (header lu comme donnees). Le rebuild v1 (`_artefact/Jainius_rebuild` initial) etait donc invalide (ex. layernorm 17.32 → 0.00019 au lieu de ~17.3246) ; refait en rebuild v2 (voir ci-dessous).
- **Fix :** `bf16_bits_to_fp32` (`u16.astype(uint32)<<16` puis `view(float32)`), `data_start = 8 + header_len` stocke par fichier, `f.seek(data_start + offset)`, support `BF16`/`U16`/`F32`/`F16`, defaut portable `_artefact/tau_a`.
- **Verification :** test dummy dans `_artefact/` + comparaison D_p vs Jainius apres rebuild.

### 3. Bugs mineurs (corriges)
- `merge_jainius.py` : stats `len(dg_by_file)-len(mapping)` → `len(weight_map)-len(mapping)` ; mismatch de shape → D_p inchange + warning ; sauvegarde directe `torch.bfloat16`.
- `verify_checkpoint.py` : crash `UnicodeEncodeError` Windows (emoji/cp1252) → sortie ASCII ; parse strict.
- `download_models.py` : defaut `./checkpoints` → `checkpoints`.
 - **2026-10-17** : `merge_jainius.py` — défaut exclut norms/embeddings (`--include-norms-embed`) ; suppression index/manifest perimes au démarrage du merge ; contrôle manifest τ (exit 2) + `--allow-missing-tau-manifest` ; README.md de destination préservé. `compute_tau_a.py` — défaut `--filter decoder`. `verify_checkpoint.py` — option `--reference <source>` vérifie noms/shapes/dtypes/sidecars. `scripts/tests/test_pipeline.py` : 9 tests (pytest).
- Docs : `CHECKLIST.md` (hash de commit en dur supprime), `AUDIT.md` (chemin `checkpoints/`, `/tmp` → `_artefact/tau_a`), `.env.template` documente, `requirements.txt` ajoute, `.gitignore` (`/tmp` supprime).

## ✅ Reconstruit et vérifié (2026-10-16, rebuild v2)

- **Rebuild v1 invalide :** valeurs fausses (bug offset, ex. layernorm 17.32 → 0.00019). Supprimé, refait.
- **Rebuild v2 :** `compute_tau_a.py` (172 s, 1013 tenseurs) + `merge_jainius.py` (169 s, 656 mappés / 391 inchangés) → `checkpoints/Jainius` remplacé (ancien corrompu supprimé, `README.md` conservé).
- **Validation numérique :** layernorm L2 `Dp=17.32424 + 0.8·τ(0.00376)` → attendu `17.32725`, obtenu `17.33191` (écart = quantification BF16, max 0.1 au voisinage de 17) ; `q_proj` écart max 0.0019 (BF16) ; 3 tenseurs non-mappés bit-identiques ; `torch`/`safe_open` strict OK ; `verify_checkpoint.py` : 11/11 `[OK]`, 1047 tenseurs BF16.
 - **Reste (à cette date) :** pas de test d'inférence GPU (fait depuis : voir « Inférence GPU vLLM (2026-10-19) ») ; A2D partiel architectural (391 vision inchangés). `smoke_int8.py` : max_memory_allocated ajouté, docstring clarifié. **Ce build (α=0.8) a été remplacé le 2026-10-19 par le build α=0.3** (l'ancien produisait un texte dégénéré) ; `checkpoints/Jainius/merge_config.json` : α=0.3, 355 fusionnés / 692 inchangés.

## ℹ️ Limites non-bugs (inchangées)
- A2D partiel : par défaut 355 tenseurs decoder fusionnés (embeddings/norms/`layer_scalar`/encoder vision inchangés ; **le routeur EST fusionné** — 90 tenseurs, famille déclarée ; correction du 2026-10-20) ; variante `--include-norms-embed` : 656 fusionnés / 391 inchangés.

## ✅ Inférence GPU vLLM (2026-10-19) — validée, cause racine de la dégénérescence trouvée

- **Config qui produit une génération** sur cette machine (WSL2 / Ubuntu 24.04, RTX 5090 32 Go) : image `vllm/vllm-openai@sha256:251eba5cc7c1…` (0.24.0) épinglée par digest, runner V2 (défaut), `--quantization fp8` (quantification à la volée des poids BF16, 25,85 GiB en VRAM), `--cpu-offload-gb 0`, `--max-model-len 2048`, `--max-num-seqs 1`, `--gpu-memory-utilization 0.92`, `--enforce-eager`. Chargement ≈ 6 min, RAM hôte stable à 4-6 GiB.
- **Runner V1 (image `gemma` 0.22.1rc1)** : crash de génération expliqué (index `-1` dans `_prepare_input_ids`, le canevas de diffusion lu comme une liste de draft tokens) ; cette image est de plus inutilisable en runner V2 sous WSL2 (UVA indisponible, `is_pin_memory_available()` = False inconditionnel). Elle ne sert plus.
- **Cause racine du « texte dégénéré »** : le checkpoint alors livré était fusionné à **α=0.8** (mesuré : `α_eff = ||Jainius − D_p|| / ||τ_A|| = 0,799 ± 0,002`, norms et embeddings inclus) et non à α=0.3 ; il ne contenait aucun `merge_config.json`. Refusion à α=0.3 → génération cohérente (4/4 puis 103/103 prompts, 0 % dégénéré). Le témoin `D_p` servit avec la config identique est lui aussi cohérent : la pile de service n'était pas en cause.
- **Comparaison par génération (103 prompts, FP8)** : Jainius α=0.3 → 88,1 % de justesse, 75,0 % de format, 21,1 tok/s ; D_p → 100 %, 93,8 %, 116,4 tok/s. **Aucun gain** : le merge dégrade le modèle de diffusion. Détails : `eval/RESULTS.md`.
- Reste : tout est mesuré en **FP8** (le BF16, 48,1 GiB, ne tient pas dans 31,84 GiB de VRAM) ; le mécanisme du ralentissement ×5 (arrêt anticipé du sampler sur `confidence_threshold`) est plausible mais non instrumenté.

## 🔴 Blocage mmap/RAM : `compute_tau_a.py` échoue en ENOMEM sur la VM actuelle (30 GiB)

- **Symptôme** : `RuntimeError: unable to mmap 49907246508 bytes from file checkpoints/gemma4_base/model-00001-of-00002.safetensors: Cannot allocate memory (12)`.
- **Cause mesurée** : `safetensors.safe_open` pré-fault la projection du fichier (MAP_POPULATE) ; le shard AR fait 46,5 GiB, la VM WSL2 n'a que 30 GiB de RAM → ENOMEM. Un `mmap` simple du même fichier réussit (Python, `PROT_READ`) : c'est bien le pré-fault de safetensors qui bloque. Les shards de 4,4 GiB de D_p/Jainius passent sans problème.
- **Provenance et état de τ_A (faits mesurés)** :
  - τ_A a bien été **calculé sur cette machine le 2026-10-16** : `_artefact/logs/rebuild_tau.log` (171,5 s) et `rebuild2_tau.log` (172,4 s), 1013 tenseurs, sortie `_artefact/tau_a`. Ce calcul est **antérieur** au passage à `safe_open` (commit `16b6efa`, le même jour à 21:14) et à l'exigence de `tau_manifest.json` (2026-10-17) : les `.npy` n'ont donc jamais eu de manifest.
  - Le **2026-10-19**, ces `.npy` étaient présents et ont servi : mesure `alpha_eff = 0,799` (17:25-17:29) puis **les deux refusions** de la journée (17:30-17:43, α=0.3 variante par défaut → remplace `checkpoints/Jainius` ; 17:51-18:05, variante `--include-norms-embed`). Les deux ont tourné avec `--allow-missing-tau-manifest` : mapping par existence des `.npy`, **provenance non vérifiée par manifest** (annoncé explicitement dans `_artefact/vllm_test/merge_a03*.log`).
  - **État actuel (2026-10-19 18:34)** : `_artefact/tau_a` est **vide** (0 `.npy`, dossier modifié à 18:34:15) — nettoyé après la refusion ; le calcul du 16/10 n'est plus réutilisable.
- **Conséquence** : le checkpoint livré est intact et vérifié (`merge_config.json` : α=0.3, 355/692 ; `verify_checkpoint.py` 11/11 [OK]) ; mais **tout nouvel α ou refusion par le chemin LEGACY exige de reconstituer τ_A avec son manifest**, impossible en l'état sur la VM à 30 GiB avec la version `safe_open` actuelle.
- **Contournement propre (2026-10-20) : la bibliothèque conforme n'est pas concernée.** `scripts/merge_a2d.py` recalcule τ_A **à la volée**, tenseur par tenseur, en FP32, sans jamais matérialiser le vecteur : la RAM reste bornée par la taille d'un tenseur, pas par celle du checkpoint. Le blocage mmap/RAM est donc propre au chemin legacy (`compute_tau_a.py` → `.npy`). La mission J6 l'a vérifié sur la famille Qwen2.5-7B → Dream-7B (shards ≤ 5 Go) : `scripts/measure_geometry.py` a recalculé ‖τ_A‖, ‖τ_D‖, ‖Δ_b‖, ‖Δ_p‖ et les cosinus **sans** écrire un seul `.npy` (résultat : reproduction de la Table 21 à ≤ 0,11 %).
- **Remède (voie legacy)** : augmenter la RAM de la VM WSL2 (`%UserProfile%\.wslconfig` → `[wsl2] memory=64GB`, puis `wsl --shutdown`), ou re-découper les checkpoints AR en shards ≤ 20 GiB. Reconstruire ensuite `tau_a` par `scripts/compute_tau_a.py` (il écrit `tau_manifest.json`, exigé par le merge legacy).
- **Interdit** : contourner par un parseur safetensors maison (`.cursorrules`). Ce contournement n'a pas été employé.
