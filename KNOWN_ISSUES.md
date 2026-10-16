# KNOWN_ISSUES.md — Anomalies connues dans le checkpoint Jainius

## ⚠️ Header JSON corrompu (critique)

**Fichier concerné :** `checkpoints/Jainius/model-00001-of-00011.safetensors`
**Symptôme :** `JSONDecodeError: Expecting value: line 2 column 1 (char 1)` au parsing du header.
**Cause :** Le padding binaire ajouté lors de la conversion U16→BF16 a décalé le header JSON, rendant certains fichiers illisibles par `json.loads()`.
**Impact :** Les tenseurs dans ce fichier ne seront pas chargés correctement par `transformers` → génération corrompue.
**Solution :** Relancer `python scripts/fix_dtype.py --checkpoint checkpoints/Jainius` (refait le fix avec padding correct).

## ⚠️ Dtype "U16" vs "BF16" dans les headers

**Fichier concerné :** Tous les fichiers `.safetensors` de Jainius.
**Symptôme :** Le header JSON indique `"dtype": "U16"` au lieu de `"dtype": "BF16"`.
**Cause :** `safetensors.torch.save_file()` écrit le dtype numpy (uint16 → "U16") au lieu du nom standard HF ("BF16").
**Impact :** Le binaire est correct (2 bytes/élément), mais certains outils HF peuvent mal interpréter.
**Solution :** `scripts/fix_dtype.py` fait un replace binaire `b'"U16"' → b'"BF16"'` + ajustement du header length.

## ℹ️ A2D partiel (non critique)

**Fichier concerné :** Tout le checkpoint Jainius.
**Description :** Seuls 656 tenseurs decoder ont été fusionnés par A2D. Les 391 tenseurs encoder vision restent inchangés depuis D_p.
**Impact attendu :** Le modèle conserve les capacités de compréhension visuelle mais n'a pas bénéficié du post-training AR pour l'encoder.
**Ce n'est pas un bug** — c'est une limitation architecturale (Gemma 4 et DiffusionGemma ont des structures différentes).

## ℹ️ Pas de tests d'inference

**Description :** Le checkpoint Jainius n'a jamais été testé en génération réelle.
**Impact :** On ne peut pas garantir la qualité du texte généré.
**Recommandation :** Tester avec `AutoModelForCausalLM.from_pretrained()` sur un GPU dédié avant de publier.
