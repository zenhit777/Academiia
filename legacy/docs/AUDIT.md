# AUDIT — Faiblesses du workspace Academiia (suivi)

**Date initiale :** 2026-10-16
**Derniere mise a jour :** 2026-10-16 (suite audit Sonnet : rename Dp+ατA, α=0.3, merge durci + manifest, safe_open, compose epingle, measure_effect)
**Statut du repo :** clean
**Checkpoints :** ~245 Go dans `checkpoints/` (racine, ignoré par git)

Convention : chaque point porte son statut — `[CORRIGE]`, `[PARTIEL]`, `[OUVERT]`.

---

## 🔴 Critiques (bloquantes)

### 1. Bug critique du checkpoint [CORRIGE]
Ancien etat : header JSON corrompu (`model-00001`) + dtype U16 + valeurs fusionnees fausses (decode BF16 + offset `seek`).
Correctif : rebuild v2 (`compute_tau_a.py` 172 s + `merge_jainius.py` 169 s) → 1047 tenseurs BF16 strict, `verify_checkpoint.py` 11/11 OK, validation numerique (ecarts = quantification BF16, non-mappes bit-identiques). `checkpoints/Jainius` remplace (model card conservee).

### 2. Aucune infrastructure d'évaluation [PARTIEL]
README + KNOWN_ISSUES : modele jamais teste en inference GPU. Ajouts : `scripts/measure_effect.py` (effet poids post-BF16, CPU, balayage α en cours), `scripts/smoke_int8.py` (test rapide 31 Go VRAM), compose vLLM BF16 offloadé documenté (`VLLM_MAX_MODEL_LEN=4096`). Reste : perplexité/bench vs `D_p` + génération A/B.

 ### 3. `fix_dtype.py` fragile [SUPPRIMÉ]

 Le script a été supprimé (commit git conserve l'historique). Le merge sauvegarde
 déjà en BF16 canonique via `torch.bfloat16`. Aucun risque de double-prefix/padding.

---

## 🟡 Moyennes (impact réel)

### 4. Aucune gestion des dépendances [CORRIGE]
`requirements.txt` ajoute (versions pinees : `huggingface_hub==1.18.0`, `transformers==5.10.2`, `safetensors`, `torch`, `numpy`).

### 5. Redondance documentaire [OUVERT]
Toujours 3 points d'entree (`AGENTS.md`, `.cursorrules`, `CLAUDE.md` redirecteur). Fardeau de maintenance inchange.

### 6. Pas de stratégie de backup [OUVERT]
~245 Go binaires sans sauvegarde documentee ; `checkpoints/` + `_artefact/` ignores par git. Perte seche si disque HS.

 ### 7. `alpha=0.8` arbitraire [CORRIGE (défaut 0.3 + balayage)]

 Ancien défaut en dur 0.8 sans ablation. Nouveau : défaut 0.3 (`merge_jainius.py`),
 `measure_effect.py` balaie 0.05–0.5 (+ variante `--include-norms-embed`).
 Résultats v2 (656 tenseurs, α=0.3) : ratio_rel médian ~3 %, cosinus attn/mlp >99.8 %,
 cosinus norms = 0.61 (arrondi BF16 détruit la direction normative). Le % d'éléments
 modifiés (~99 %) ne mesure PAS l'ampleur du changement — un ULP BF16 minuscule suffit
 à faire "bouger" des bits sans effet numérique significatif. Trancher α sur chiffres
 (perplexité/bench vs `D_p`) reste à faire.

---

## 🟢 Mineures (qualité)

### 8. `.env.template` vide [CORRIGE]
Documente : aucune variable requise, `HF_TOKEN` optionnel.

### 9. Pas de CI/CD ni validation automatique [CORRIGE]
`.github/workflows/ci.yml` : syntaxe + merge/verify/measure sur mini-safetensors factices.

### 10. "A2D partiel" minimise [PARTIEL]
Documente comme limite dans `KNOWN_ISSUES.md` + tableau README (391 inchanges, bit-identiques verifies), mais pas mis en avant scientifiquement.

### 11. Chemin tau_A non portable [CORRIGE]
Defaut `_artefact/tau_a` (portable, ignore par git). Docs (`AGENTS.md`, `README.md`) alignees.

---
 ## Résumé actualise

 > Le pipeline est **techniquement assaini** (checkpoint valide, scripts corriges et verifies). Suite audit Sonnet : methode renamee Dp+ατA (α=0.3), merge durci + manifest, safe_open, compose epingle, mesure poids v2 avec indicateurs d'ampleur (ratio_rel, cosinus). Reste le **chantier scientifique** : trancher α sur chiffres (perplexité/bench vs `D_p`), backup (6), consolidation docs (5).

