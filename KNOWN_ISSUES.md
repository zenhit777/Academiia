 # KNOWN_ISSUES.md — Anomalies du checkpoint Jainius (rebuild v2 en place, 2026-10-17)

## ✅ Suite audit Sonnet 2026-10-16 (appliquée, en attente du balayage α)

- **Méthode renommée :** ce n'est PAS le A2D-Merge (`D_p + λ(τ_A − τ_D)`, τ_D inexistant publiquement) mais l'ablation `D_p + ατ_A` (Tab. 9). README/AGENTS corrigés, défaut **α=0.3** (0.8 déprécié : −3.8 pts SLAKE dans le papier).
 - **Parseur maison en retrait :** `compute_tau_a.py` + `merge_jainius.py` lisent via `safe_open`, helpers partagés `scripts/_common.py`. Le merge sauvegarde déjà en BF16 canonique.
- **Merge durci :** shape mismatch → erreur par défaut, manifest `merge_config.json`, écriture atomique, nettoyage shards obsolètes (`--overwrite` exigé), recopie de TOUS les sidecars (LICENSE...), `total_size` dans l'index. `verify_checkpoint.py` : égalité index/headers exigée, contrôle bon-fichier par tenseur, `--check-values` NaN/Inf, HF files requis.
- **Serving :** compose en forme shell (multi-flags `VLLM_EXTRA_ARGS` OK), image épinglée `vllm/vllm-openai:gemma`, `ipc: host`, `VLLM_MAX_MODEL_LEN=4096`, `MAX_NUM_SEQS=1`, `GPU_MEMORY_UTILIZATION=0.90`. `.gitignore` : `__pycache__/`, `*.pyc`, `.venv/`, `*.npy`.
 - **Mesure poids (résultats 2026-10-17, v2) :** `scripts/measure_effect.py` sur les 656 tenseurs (25.2B éléments) — % reellement modifies apres arrondi BF16 : α=0.05→94.8%, 0.1→97.3%, 0.2→98.5%, **0.3→99.1%**, 0.5→99.4%. Variante sans norms/embed (355 tenseurs) quasi identique. Par famille a 0.3 : attn ratio_rel med=2.98%, cos=0.998 ; mlp/experts ratio_rel med=2.82%, cos=0.998 ; embed ratio_rel med=2.66%, cos=0.998 ; norms ratio_rel med=0.044%, cos=0.61 (arrondi BF16 detruit la direction) ; routeur τ≈0. Le % d'elements modifies NE mesure PAS l'ampleur du changement : un ULP BF16 minuscule suffit a faire "bouger" des bits sans effet numerique significatif. La quantification n'absorbe presque rien — α pilote reellement le merge, d'où l'importance du balayage et de l'éval vs `D_p`. JSON : `_artefact/measure_effect_v2*.json`.
 - **2026-10-17** : `merge_jainius.py` — défaut exclut norms/embeddings (opt-in `--include-norms-embed`) ; suppression index/manifest perimes au démarrage du merge ; contrôle manifest τ (exit 2) + `--allow-missing-tau-manifest` ; README.md de destination préservé. `compute_tau_a.py` — défaut `--filter decoder`. `verify_checkpoint.py` — option `--reference <source>` vérifie noms/shapes/dtypes/sidecars. `scripts/tests/test_pipeline.py` : 9 tests (pytest).
- **Reste :** **test d'inférence GPU → fait le 2026-10-19** (voir section « Inférence GPU vLLM (2026-10-19) ») ; **pas de fichier `LICENSE`** — les repos sources (`google/gemma-4-26B-A4B`, `google/gemma-4-26B-A4B-it`, `google/diffusiongemma-26B-A4B-it`) n'en publient aucun (vérifié le 2026-10-19) : rien à recopier, la model card déclare `apache-2.0` + `license_link` vers les conditions Gemma ; **`_artefact/tau_a` est absent** et ne peut pas être recalculé sur cette machine (voir section « Blocage mmap/RAM »).

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
- A2D partiel : par défaut 355 tenseurs decoder fusionnés (embeddings/norms/routeur/encoder vision inchangés) ; variante `--include-norms-embed` : 656 fusionnés / 391 inchangés.

## ✅ Inférence GPU vLLM (2026-10-19) — validée, cause racine de la dégénérescence trouvée

- **Config qui produit une génération** sur cette machine (WSL2 / Ubuntu 24.04, RTX 5090 32 Go) : image `vllm/vllm-openai@sha256:251eba5cc7c1…` (0.24.0) épinglée par digest, runner V2 (défaut), `--quantization fp8` (quantification à la volée des poids BF16, 25,85 GiB en VRAM), `--cpu-offload-gb 0`, `--max-model-len 2048`, `--max-num-seqs 1`, `--gpu-memory-utilization 0.92`, `--enforce-eager`. Chargement ≈ 6 min, RAM hôte stable à 4-6 GiB.
- **Runner V1 (image `gemma` 0.22.1rc1)** : crash de génération expliqué (index `-1` dans `_prepare_input_ids`, le canevas de diffusion lu comme une liste de draft tokens) ; cette image est de plus inutilisable en runner V2 sous WSL2 (UVA indisponible, `is_pin_memory_available()` = False inconditionnel). Elle ne sert plus.
- **Cause racine du « texte dégénéré »** : le checkpoint alors livré était fusionné à **α=0.8** (mesuré : `α_eff = ||Jainius − D_p|| / ||τ_A|| = 0,799 ± 0,002`, norms et embeddings inclus) et non à α=0.3 ; il ne contenait aucun `merge_config.json`. Refusion à α=0.3 → génération cohérente (4/4 puis 103/103 prompts, 0 % dégénéré). Le témoin `D_p` servit avec la config identique est lui aussi cohérent : la pile de service n'était pas en cause.
- **Comparaison par génération (103 prompts, FP8)** : Jainius α=0.3 → 88,1 % de justesse, 75,0 % de format, 21,1 tok/s ; D_p → 100 %, 93,8 %, 116,4 tok/s. **Aucun gain** : le merge dégrade le modèle de diffusion. Détails : `eval/RESULTS.md`.
- Reste : tout est mesuré en **FP8** (le BF16, 48,1 GiB, ne tient pas dans 31,84 GiB de VRAM) ; le mécanisme du ralentissement ×5 (arrêt anticipé du sampler sur `confidence_threshold`) est plausible mais non instrumenté.

## 🔴 Blocage mmap/RAM : `compute_tau_a.py` ne peut pas s'exécuter sur cette machine

- **Symptôme** : `RuntimeError: unable to mmap 49907246508 bytes from file checkpoints/gemma4_base/model-00001-of-00002.safetensors: Cannot allocate memory (12)`.
- **Cause mesurée** : `safetensors.safe_open` pré-fault la projection du fichier (MAP_POPULATE) ; le shard AR fait 46,5 GiB, la VM WSL2 n'a que 30 GiB de RAM → ENOMEM. Un `mmap` simple du même fichier réussit (Python, `PROT_READ`) : c'est bien le pré-fault de safetensors qui bloque. Les shards de 4,4 GiB de D_p/Jainius passent sans problème.
- **Conséquence** : `_artefact/tau_a` (τ_A) est **absent** et ne peut pas être recalculé ici ; un refusion nécessite de l'être ailleurs. Le checkpoint livré est intact et vérifié (`merge_config.json` : α=0.3, 355/692).
- **Remède** : augmenter la RAM de la VM WSL2 (`%UserProfile%\.wslconfig` → `[wsl2] memory=64GB`, puis `wsl --shutdown`), ou re-découper les checkpoints AR en shards ≤ 20 GiB. Reconstruire ensuite `tau_a` par `scripts/compute_tau_a.py` (il écrit `tau_manifest.json`, exigé par le merge).
- **Interdit** : contourner par un parseur safetensors maison (`.cursorrules`). Ce contournement n'a pas été employé.
