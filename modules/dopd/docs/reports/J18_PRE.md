# Pré-enregistrement J18 — Expérience de décision à budget borné

**Date :** 2026-10-26  
**Statut :** figé avant tout code producteur. Toute modification après un résultat vaut nouveau protocole, pas correction de rapport.  
**Ancêtre du code producteur :** `c1d6eaf` (HEAD actuel).  
**Aucun code J18 ne sera autorisé tant que ce pré-enregistrement n'aura pas été revu et accepté.**

---

## 0. Objectif

Expérience de **décision à budget borné**, pas campagne d'entraînement. Comparer une cible future-aware vs. cible causale sur un protocole minimal, avec le professeur en FP32 comme précision de production et le rejeu intégral FP32 comme référence d'exactitude.

**Ce que J18 fait :**
- Un seul passage sur un sous-ensemble réduit de prompts (≤ 50) ;
- Scoring du `TeacherScorer` réel en FP32 avec rejeu intégral FP32 comme référence ;
- Mesure de l'effet future-aware (ΔKL, médiane, queue nuisible) sur la géométrie d'entraînement réelle (préfixes courts 12–21 tokens) ;
- Décision GO / NO-GO pour un `StudentPolicy` réduit.

**Ce que J18 ne fait pas :**
- Aucun `StudentPolicy` complet ;
- Aucun entraînement multi-epoch ;
- Aucune échelle > 0,6B (Qwen3-0.6B uniquement) ;
- Aucun push vers `origin/main` (règle 11 d'AGENTS.md) ;
- Aucune modification de G1 ou des résultats A2D ;
- Aucun benchmark reproduit.

---

## 1. Précision et référence

| Élément | Valeur figée | Justification |
|---|---|---|
| Professeur | `Qwen/Qwen3-0.6B` en **FP32** (poids chargés FP32) | R1 : écart C2 imputé à BF16 ; FP32 est la précision de production |
| Référence d'exactitude | **Rejeu intégral FP32** (contexte complet rejoué par branche, sans cache) | Plus de comparaison cache/rejeu en BF16 ; le rejeu FP32 est la seule référence valide |
| Arithmétique cible | `float32` (§3.6 du papier) | Inchangée : logits, log-probabilités, deltas, normalisation en float32 |
| Géométrie de scoring | Préfixes **12 à 21 tokens inclus** (états réels J17) ; suffixe variable (1–3 tokens) | Ne pas hériter du gain cache de la cellule de référence longue (préfixe 1024). Le cache est plus lent que le rejeu aux préfixes courts (§6.11.4) : mesurer l'effet **sans** le biais du cache |
| `k` candidats | **16** (union top-16 professeur / top-16 élève) | §3.4, §6.9.8 ; couverture 99,45 % / 99,44 % à k=16 |
| Branches maximales par job | **33** (≤ 32 candidats + référence) | App. B du papier |
| `N` (taille de bloc) | **4** | §6.1 (défaut du papier) ; cohérent avec J17 |
| Seed | **20261026** | Date de la mission |

---

## 2. Géométrie préfixée

La géométrie réellement utilisée est **explicitement préfixée** :

- `prefix_ids = (q, z, Â_i)` — contexte prompt + partie du bloc avant `i` ; longueur **12 à 21 tokens inclus** (mesurée par job) ;
- `suffix_ids = (x̂_{i+1}, …, x̂_N)` — suffixe visible de longueur **1 à 3 tokens** ;
- `candidate_ids = V_i` — union des top-16 professeur et top-16 élève (§3.4), sans doublon ;
- `reference_id = x̂_i` — token réellement produit à la position `i`.

**Règle explicite :** ne pas utiliser de cellule de référence longue (préfixe 1024 tokens) pour mesurer l'effet future-aware. Le gain cache de J17 (3,28× à 512 tokens) est un artefact de géométrie long ; aux préfixes courts 12–21, le cache est **plus lent** que le rejeu (§6.11.4). L'effet future-aware doit être mesuré sur la géométrie d'entraînement réelle, sans hériter du gain cache.

---

## 3. Critères pré-enregistrés (C1…Cn)

### C1 — Effet future-aware significatif (critère primaire)
- **Seuil :** `ΔKL` agrégé > 0 et IC 95 % (bootstrap de blocs, clé `(prompt, config_index, block_index)`) excluant zéro.
- **Interprétation :** l'effet est réel sur le sous-ensemble réduit.

### C2 — Médiane de ΔKL > 0
- **Seuil :** médiane per-marginal `> 0`.
- **Rappel J15/J16 :** la médiane était nulle (42,788 % de marginaux structurellement nuls). Ce critère est conservé pour vérifier que le sous-ensemble réduit ne reproduit pas l'artefact.

### C3 — Proportion de prompts significatifs ≥ 0,5
- **Seuil :** fraction des prompts dont l'IC 95 % per-prompt exclut zéro `≥ 0,5`.
- **Rappel J16 :** 14/20 = 0,70 en lecture stricte.

### C4 — Test des signes par prompt, p < 0,05
- **Seuil :** test binomial exact bilatéral `p < 0,05`.
- **Rappel J16 :** 17/20 positifs → `p = 0,00258`.

### C5 — Effectif minimal
- **Seuil :** ≥ 100 blocs de base par configuration et ≥ 2 000 marginaux token admissibles au total.
- Un effectif inférieur rend la porte **NON CONCLUANTE**, même si le critère primaire est favorable.

### C6 — Intégrité numérique
- Aucun NaN/Inf dans les cibles, KL ou probabilités ; somme de chaque distribution `= 1` à `1e-5` près.

### C7 — Traceabilité ADR-002
- Manifeste `academiia.lineage/1` présent, `producer.module = "dopd"`, commit, commande exacte, versions, seeds, empreintes des sources, `training.performed = false`.

### C8 — Gelé (non-régression)
- `requires_grad` de tous les paramètres du professeur = `False` ; `model.training = False` ; aucun appel `backward`, `optimizer`, `step` dans le harnais.

### C9 — Provenance du modèle (règle 9)
- Snapshot lu **dans le dépôt** (`model_source_in_repo = true`) ; sinon avertissement émis et `model_source_in_repo = false` consigné.

### C10 — Non-régression
- `target.py`, `diagnostic.py` (correctifs J16) et `modules/a2d/eval/dream_a2d/` **inchangés** sur la plage de commits.

---

## 4. Règle de décision explicite

| Condition | Décision |
|---|---|
| C1–C5 tenus, C6–C10 tenus | **GO** pour un `StudentPolicy` réduit (≤ 0,6B) |
| C1 non tenu | **NO-GO** : l'effet future-aware n'est pas significatif sur le sous-ensemble ; J18 s'arrête à la documentation |
| C2 ou C3 fragiles (médiane ≤ 0 ou proportion < 0,5) mais C1 tenu | **GO conditionnel** — réserve explicite : l'effet est porté par une queue ; un `StudentPolicy` doit confirmer sur plus de prompts/échelles avant implémentation |
| Tout critère hors liste NO-GO (C6–C10) échoue | **Décision explicite écrite** (ADR ou STATUS) requise ; le silence n'est pas un GO. Cette règle est l'amendement R1 (§6.10.8 de DOPD_SPEC.md). |

---

## 5. Budget strict

| Poste | Plafond | Arrêt automatique |
|---|---|---|
| Temps GPU cumulé | **≤ 4 h** (RTX 5090, FP32) | Le script interrompt le protocole au-delà ; aucun résultat n'est écrit si le plafond est dépassé |
| Mémoire GPU pic | **≤ 16 GiB** (sur 32 Go total) | `torch.cuda.max_memory_allocated()` vérifié après chaque batch ; dépassement → arrêt immédiat |
| Nombre de prompts | **≤ 50** (sous-ensemble réduit du jeu J15) | Aucun prompt supplémentaire n'est ajouté a posteriori |
| Branches par job | **≤ 33** (union ≤ 32 + référence) | §3.6 du papier : le lot est réduit au-delà de 16 GiB de cache étendu |

---

## 6. Hors périmètre explicite

- Aucun `StudentPolicy` complet ;
- Aucun entraînement multi-epoch ;
- Aucune échelle > 0,6B (Qwen3-0.6B uniquement) ;
- Aucun push vers `origin/main` (règle 11 d'AGENTS.md) ;
- Aucune modification de G1 ou des résultats A2D ;
- Aucun benchmark reproduit ;
- Aucun chargement de modèle hors dépôt (`_artefact/hf-cache/`) ;
- Aucun artefact d'évaluation régénéré (les JSON versionnés J14–J17 sont immuables).

---

## 7. Condition préalable recommandée (non bloquante)

**Diagnostic Annexe D rejoué sur Qwen3-1.7B (voire 4B) en inférence seule**, avant tout entraînement ultérieur, pour expliquer l'écart d'échelle avec le papier (§5.1 : gain 2,6–4,0 pts par échelle). Ce diagnostic est hors J18 mais recommandé comme pré-requis de lecture pour la décision J19.

---

## 8. Livrables attendus

- `modules/dopd/eval/j18_decision/results.json` — résultats versionnés (sha256 immuable) ;
- `modules/dopd/docs/reports/J18.md` — rapport de mission avec verdict, IC, et décision ;
- Manifeste ADR-002 dans le JSON (`lineage` bloc).

---

**Ce document est figé. Toute modification après un résultat vaut nouveau protocole.**
