# AUDIT — Faiblesses du workspace Academiia

**Date :** 2026-10-16  
**Statut du repo :** clean (dernier commit `9d90a41 docs: integrate .cursorrules...`)  
**Checkpoints :** ~242 Go dans `checkpoints/` (racine, ignoré par git)

---

## 🔴 Critiques (bloquantes)

### 1. Bug critique non corrigé
Le checkpoint Jainius est **dans un état inconsistent** : header JSON corrompu dans `model-00001-of-00011.safetensors` + dtype U16 au lieu de BF16 partout. Le KNOWN_ISSUES.md le documente mais rien n'a été fait pour corriger. C'est comme publier un modèle avec des weights illisibles.

### 2. Aucune infrastructure d'évaluation
Le README dit *"Le modèle n'a pas été testé en inference"* et le KNOWN_ISSUES confirme : **on ne sait pas si Jainius est meilleur que DiffusionGemma**. Pas de benchmark, pas de métriques, pas de script de génération pour valider la qualité du texte. Un checkpoint A2D sans preuve d'amélioration, c'est un artefact scientifique incomplet.

### 3. `fix_dtype.py` fragile
Le fix est un **remplacement binaire** (`b'"U16"' → b'"BF16"'`) avec ajustement de header length. C'est du hacking binaire : si le format safetensors évolue, ça plante silencieusement. Pas de validation post-fix (sauf le script verify qui n'a pas été exécuté).

---

## 🟡 Moyennes (impact réel)

### 4. Aucune gestion des dépendances
Pas de `requirements.txt`, `pyproject.toml`, ni Dockerfile. Les versions sont implicites (`huggingface_hub` 1.18.0, `transformers` 5.10.2…). Un cloneur ne peut pas reproduire l'environnement sans deviner.

### 5. Redondance documentaire
3 points d'entrée pour la même info : `AGENTS.md`, `.cursorrules`, `CLAUDE.md`. Le CLAUDE.md n'est qu'un redirecteur vers AGENTS.md. Ça crée un **fardeau de maintenance** (modifier les 3 à chaque changement) sans valeur ajoutée.

### 6. Pas de stratégie de backup
242 Go de checkpoints binaires, aucun mention de sauvegarde. Si le disque tombe en panne, tout est perdu. Le `.gitignore` exclut `_artefact/`, donc pas de versioning non plus.

### 7. `alpha=0.8` arbitraire
Le poids d'interpolation A2D est hardcoded à 0.8 sans ablation study ni analyse de sensibilité. Pourquoi 0.8 ? Est-ce optimal ? C'est un paramètre scientifique, pas une constante magique.

---

## 🟢 Mineures (qualité)

### 8. `.env.template` vide
Crée de la confusion : "est-ce qu'il y a des variables d'environnement à configurer ou non ?" Si non, pourquoi le fichier existe-t-il ?

### 9. Pas de CI/CD ni validation automatique
Le `verify_checkpoint.py` existe mais n'est **jamais exécuté automatiquement**. Rien ne bloque un commit si la checklist échoue.

### 10. "A2D partiel" minimisé
Les 391 tenseurs encoder vision inchangés sont documentés comme *"ce n'est pas un bug"*. C'est une **limitation scientifique majeure** : le modèle garde les capacités visuelles mais sans post-training AR sur l'encoder. Ça devrait être plus visible, pas relégué en note de bas de page.

### 11. Chemin tau_A non portable (corrige)
Ancien defaut `/tmp/a2d_jainius` (chemin Linux). Corrige : defaut `_artefact/tau_a` (portable, ignore par git).

---

## Résumé en une phrase

> Le projet a un **pipeline technique solide** mais est **scientifiquement incomplet** : pas de preuve d'amélioration, pas de validation automatique, et le checkpoint final est dans un état corrompu non corrigé.
