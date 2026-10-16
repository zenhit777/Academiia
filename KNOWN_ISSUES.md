# KNOWN_ISSUES.md — Anomalies du checkpoint Jainius (rebuild v2 en place, 2026-10-16)

## ✅ Resolu dans les scripts (2026-10-16)

### 1. `fix_dtype.py` cassé (corrige)
- **Cause :** `np` sans `import numpy` + patch binaire en place (`U16`→`BF16` allonge le header sans decaler les donnees) → double prefix, padding `\x00` trailing, JSON illisible en mode strict.
- **Fix :** reecriture complete via tmp + `os.replace` (`scripts/fix_dtype.py:repair_one_file`) : header canonique `json.dumps(separators=(",",":"))`, `U16`→`BF16` dans le dict, detection double-prefix et truncation.
- **Verification :** `python scripts/verify_checkpoint.py` (strict, ASCII, detecte padding/double-prefix/truncation).

### 2. Decodage BF16 faux + offset de lecture faux (corriges)
- **Cause decode :** `np.frombuffer(raw, uint16).astype(float32)` donne la valeur entiere du bit-pattern (ex. `0x3F80` → `16256.0` au lieu de `1.0`).
- **Cause offset (critique, ratee au premier passage) :** `f.seek(offset)` utilisait l'offset relatif aux donnees comme position absolue, sans ajouter `8 + header_len`. Toutes les lectures etaient decalees (~8 Ko), y compris le tenseur a offset 0 (header lu comme donnees). Le rebuild v1 (`_artefact/Jainius_rebuild` initial) etait donc invalide (ex. layernorm 17.32 → 0.00019 au lieu de ~17.3246) ; refait en rebuild v2 (voir ci-dessous).
- **Fix :** `bf16_bits_to_fp32` (`u16.astype(uint32)<<16` puis `view(float32)`), `data_start = 8 + header_len` stocke par fichier, `f.seek(data_start + offset)`, support `BF16`/`U16`/`F32`/`F16`, defaut portable `_artefact/tau_a`.
- **Verification :** test dummy dans `_artefact/` + comparaison D_p vs Jainius apres rebuild.

### 3. Bugs mineurs (corriges)
- `merge_jainius.py` : stats `len(dg_by_file)-len(mapping)` → `len(weight_map)-len(mapping)` ; mismatch de shape → D_p inchange + warning ; sauvegarde directe `torch.bfloat16`.
- `verify_checkpoint.py` : crash `UnicodeEncodeError` Windows (emoji/cp1252) → sortie ASCII ; parse strict.
- `download_models.py` : defaut `./checkpoints` → `checkpoints`.
- Docs : `CHECKLIST.md` (hash de commit en dur supprime), `AUDIT.md` (chemin `checkpoints/`, `/tmp` → `_artefact/tau_a`), `.env.template` documente, `requirements.txt` ajoute, `.gitignore` (`/tmp` supprime).

## ✅ Reconstruit et vérifié (2026-10-16, rebuild v2)

- **Rebuild v1 invalide :** valeurs fausses (bug offset, ex. layernorm 17.32 → 0.00019). Supprimé, refait.
- **Rebuild v2 :** `compute_tau_a.py` (172 s, 1013 tenseurs) + `merge_jainius.py` (169 s, 656 mappés / 391 inchangés) → `checkpoints/Jainius` remplacé (ancien corrompu supprimé, `README.md` conservé).
- **Validation numérique :** layernorm L2 `Dp=17.32424 + 0.8·τ(0.00376)` → attendu `17.32725`, obtenu `17.33191` (écart = quantification BF16, max 0.1 au voisinage de 17) ; `q_proj` écart max 0.0019 (BF16) ; 3 tenseurs non-mappés bit-identiques ; `torch`/`safe_open` strict OK ; `verify_checkpoint.py` : 11/11 `[OK]`, 1047 tenseurs BF16.
- **Reste :** pas de test d'inférence GPU (recommandé avant publication) ; A2D partiel architectural (391 vision inchangés).

## ℹ️ Limites non-bugs (inchangées)
- A2D partiel : 656 tenseurs decoder fusionnes, 391 vision inchanges (architectural).
- Pas de test d'inference GPU (recommande avant publication).
