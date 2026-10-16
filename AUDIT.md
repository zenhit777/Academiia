# AUDIT — Faiblesses du workspace Academiia (suivi)

**Date initiale :** 2026-10-16
**Derniere mise a jour :** 2026-10-16 (commit `9621662 fix: corrige pipeline A2D + rebuild Jainius v2`)
**Statut du repo :** clean
**Checkpoints :** ~245 Go dans `checkpoints/` (racine, ignoré par git)

Convention : chaque point porte son statut — `[CORRIGE]`, `[PARTIEL]`, `[OUVERT]`.

---

## 🔴 Critiques (bloquantes)

### 1. Bug critique du checkpoint [CORRIGE]
Ancien etat : header JSON corrompu (`model-00001`) + dtype U16 + valeurs fusionnees fausses (decode BF16 + offset `seek`).
Correctif : rebuild v2 (`compute_tau_a.py` 172 s + `merge_jainius.py` 169 s) → 1047 tenseurs BF16 strict, `verify_checkpoint.py` 11/11 OK, validation numerique (ecarts = quantification BF16, non-mappes bit-identiques). `checkpoints/Jainius` remplace (model card conservee).

### 2. Aucune infrastructure d'évaluation [OUVERT]
README + KNOWN_ISSUES : modele jamais teste en inference GPU. Pas de benchmark, metriques ni script de generation. Artefact scientifiquement incomplet (amelioration A2D non demontree).

### 3. `fix_dtype.py` fragile [CORRIGE]
Ancien etat : remplacement binaire en place + `np` sans import → double prefix, padding, JSON illisible.
Correctif : reecriture complete via tmp + `os.replace`, header canonique, detection double-prefix/padding/truncation. Valide sur cas factices + `verify` strict.

---

## 🟡 Moyennes (impact réel)

### 4. Aucune gestion des dépendances [CORRIGE]
`requirements.txt` ajoute (versions pinees : `huggingface_hub==1.18.0`, `transformers==5.10.2`, `safetensors`, `torch`, `numpy`).

### 5. Redondance documentaire [OUVERT]
Toujours 3 points d'entree (`AGENTS.md`, `.cursorrules`, `CLAUDE.md` redirecteur). Fardeau de maintenance inchange.

### 6. Pas de stratégie de backup [OUVERT]
~245 Go binaires sans sauvegarde documentee ; `checkpoints/` + `_artefact/` ignores par git. Perte seche si disque HS.

### 7. `alpha=0.8` arbitraire [OUVERT]
Toujours defaut en dur dans `merge_jainius.py`, sans ablation ni sensibilite.

---

## 🟢 Mineures (qualité)

### 8. `.env.template` vide [CORRIGE]
Documente : aucune variable requise, `HF_TOKEN` optionnel.

### 9. Pas de CI/CD ni validation automatique [OUVERT]
`verify_checkpoint.py` existe et passe, mais jamais execute automatiquement. Rien ne bloque un commit KO.

### 10. "A2D partiel" minimise [PARTIEL]
Documente comme limite dans `KNOWN_ISSUES.md` + tableau README (391 inchanges, bit-identiques verifies), mais pas mis en avant scientifiquement.

### 11. Chemin tau_A non portable [CORRIGE]
Defaut `_artefact/tau_a` (portable, ignore par git). Docs (`AGENTS.md`, `README.md`) alignees.

---

## Résumé actualise

> Le pipeline est **techniquement assaini** (checkpoint valide, scripts corriges et verifies). Reste le **chantier scientifique** : preuve d'amelioration (2), robustesse process (5, 6, 9) et justification d'`alpha` (7).
