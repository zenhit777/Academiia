# Mission J11bis — Clôture des écarts d'audit (CI + contrat d'installation)

**Date** : 2026-10-22 · **Architecte** : Grok (niveau 2) · **Développeur** : OhMyPi
**HEAD de départ** : `9365979` (fin J11, rapport inclus) · **HEAD d'arrivée** : `25f6e60`
**Mode** : CPU uniquement. Aucun GPU, aucune inférence, aucun checkpoint touché.

---

## 1. Objectif

Corriger les deux écarts retenus par l'audit externe post-J11 :

- **P1** : la commande générale `pytest` de la CI n'inclut pas `infra/vllm/scripts/tests`.
- **P2** : `requirements.txt` ne contient que des commentaires → `pip install -r requirements.txt` n'installe rien.

---

## 2. Phase 1 — P1 : couverture CI

### Fichier modifié : `.github/workflows/ci.yml`

#### Avant (ligne 135)
```yaml
- name: Suite de tests (hors mutations)
  run: pytest modules/a2d/tests tests -q --ignore=modules/a2d/tests/test_a2d_mutation.py
```

La commande ne couvrait que `modules/a2d/tests` et `tests`. Les 17 tests infra (`infra/vllm/scripts/tests`) étaient exécutés dans un step séparé (ligne 201), mais le total « 118 » du rapport J11 ne correspondait pas à ce que la CI exécute en une passe.

#### Après
```yaml
- name: Suite de tests (hors mutations)
  run: pytest modules/a2d/tests tests infra/vllm/scripts/tests -q --ignore=modules/a2d/tests/test_a2d_mutation.py
```

La ligne 201 (`pytest infra/vllm/scripts/tests -q`) a été remplacée par un commentaire :
```yaml
# Tests vLLM : syntaxe des scripts shell + compilation smoke_chat.py (la suite pytest generale inclut deja infra/vllm/scripts/tests).
```

Les steps `bash -n` et `py_compile` (lignes 202-203) ont été conservés car ils apportent une vérification supplémentaire (syntaxe des scripts shell, compilation de `smoke_chat.py`) que pytest ne couvre pas.

#### Résultat d'exécution
```
pytest modules/a2d/tests tests infra/vllm/scripts/tests -q --ignore=... → 94 passed in 63.11s
pytest modules/a2d/tests/test_a2d_mutation.py -q                    → 24 passed in 21.03s
Total hors mutations + mutations                                   → 118 (cohérent avec le rapport J11)
```

---

## 3. Phase 2 — P2 : contrat d'installation

### Fichier modifié : `requirements.txt`

#### Avant
Fichier purement commenté (pointeur vers pyproject.toml), aucune instruction d'installation.

#### Après
```text
# Source de vérité des versions : pyproject.toml
# Ce fichier assure la compatibilité avec : pip install -r requirements.txt
-e .

# Développement / évaluation / PEFT (recommandé) :
#   pip install -e ".[dev,eval,peft]"
#
# Harnais Dream (versions propres) :
#   modules/a2d/harness/dream/requirements-dream.txt
#
# Python >= 3.12 requis.
```

### Fichier modifié : `README.md` (section Installation)

Ajout d'une note PEP 668 avec procédure venv dans le dépôt :
```bash
python3 -m venv _artefact/venv-dev --system-site-packages
source _artefact/venv-dev/bin/activate
pip install -e .
```

### Vérification dans un venv propre (`_artefact/venv-j11bis`)

| Commande | Résultat |
|---|---|
| `python3 -m pip install --upgrade pip` | OK (pip 26.0) |
| `pip install -r requirements.txt` | OK (editable + entry-points) |
| `import academiia.core, academiia.a2d; print('OK')` | **OK** |
| `merge-a2d --help` | **OK** (5 entry-points fonctionnels) |

---

## 4. Phase 3 — Vérifications bloquantes

| # | Vérification | Résultat | Statut |
|---|---|---|---|
| a | `pytest modules/a2d/tests tests infra/vllm/scripts/tests -q --ignore=...` | **94 passed** | ✓ |
| b | `pytest modules/a2d/tests/test_a2d_mutation.py -q` | **24 passed** | ✓ |
| c | Total a+b ≥ 118 | **118** | ✓ |
| d | `import academiia.core, academiia.a2d; print('OK')` | **OK** | ✓ |
| e | `merge-a2d --help` après install via `requirements.txt` | **OK** | ✓ |
| f | `python tools/check_links.py` | 47 fichiers, tous liens OK | ✓ |
| g | `git diff 9365979..HEAD -- modules/a2d/eval/dream_a2d/` | **vide** | ✓ |
| h | Aucun fichier > 1 Mo ajouté ; checkpoints intact (332 Go, 10 entrées) | OK | ✓ |
| i | `git status` propre (hors artefacts ignorés) | vide | ✓ |

---

## 5. Commits

| Commit | Message |
|---|---|
| `96f705b` | fix(ci): inclure infra/vllm/scripts/tests dans la suite generale ; fix(install): requirements.txt operationnel via -e . ; README aligne |
| `25f6e60` | docs(rapport): addendum J11bis — P1 CI infra tests, P2 requirements.txt operationnel |

---

## 6. Écarts persistants (non résolus par J11bis)

- **Transport SSH** : l'environnement WSL n'a aucune clé SSH ; le transport se fait via `ssh.exe` Windows (`C:\Users\sjteb\.ssh\id_ed25519`, authentifié `zenhit777`).
- **PEP 668** : installation sur le Python système refusée ; un venv dans le dépôt est requis.

---

## 7. Conclusion

Les deux écarts d'audit (P1, P2) sont corrigés et vérifiés. La CI est maintenant cohérente avec le rapport J11 (118 tests). Le contrat d'installation (`requirements.txt`) est opérationnel. J11bis est close ; **J12** (brief + squelette du module d-OPD, Qwen3-0.6B en dev) peut démarrer.
