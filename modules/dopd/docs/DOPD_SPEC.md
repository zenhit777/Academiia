# d-OPD — brief de conception (mission J12)

- **Statut** : document de référence du module `modules/dopd/` — **squelette J12**, aucun entraînement.
- **Source primaire** : Liu, Hu, Han, « *d-OPD: Future-Aware On-Policy Distillation for
  Block Diffusion Language Models* », arXiv:2609.35362v1, 28 septembre 2026
  (<https://arxiv.org/abs/2609.35362>). Code annoncé : <https://github.com/mit-han-lab/d-OPD>.
- **Public** : développeur qui n'a pas lu le papier. Chaque affirmation chiffrée porte sa
  section ou son tableau d'origine.
- **Règle de lecture** : les chiffres de ce document sont ceux du **papier**. Aucun n'est
  reproduit localement en J12 ; voir §7.

> **Avertissement de périmètre.** J12 livre un **brief** et un **squelette importable**. Il
> n'y a ni téléchargement de modèle, ni rollout, ni entraînement, ni évaluation. Le module
> ne touche ni aux résultats, ni aux seuils, ni aux verdicts de la trajectoire A2D
> (`docs/CHECKLIST.md`).

---

## 0. En une page

Un modèle de langage à diffusion par blocs (*block diffusion language model*, dLLM) génère
des blocs de `N` tokens de gauche à droite ; à l'intérieur d'un bloc, les tokens se démasquent
en parallèle et se conditionnent mutuellement. Pour convertir un modèle autoregressif (AR)
déjà entraîné en dLLM, la distillation sur politique (*on-policy distillation*, OPD) fait
générer des trajectoires par l'élève, puis supervise ces états avec un professeur AR gelé.

**Le problème.** À un même état de débruitage, l'élève voit le bloc partiellement démasqué,
donc éventuellement des tokens **à droite** de la position à prédire. Le professeur AR, lui,
ne conditionne que sur le **préfixe causal**. Même si l'état est bien celui visité par
l'élève (ce que fait déjà OPDLM), la distribution du professeur ne reflète pas l'information
disponible pour l'élève : c'est le *target-conditioning mismatch* (§3.1).

**L'idée.** Le professeur AR définit une probabilité jointe sur le bloc. Conditionner cette
jointe sur les tokens visibles du bloc revient à **repondérer** la distribution causale par la
vraisemblance du futur visible sous chaque candidat (eq. 10). La cible exacte est
intraitable (marginalisation sur les préfixes et les futurs non observés). d-OPD l'approxime
avec le **bloc complété du même rollout** : le préfixe complété fournit la probabilité
causale `r_i(v)`, le suffixe complété fournit un score de compatibilité futur `s_i(v)`, et la
cible est `p̃_T,i(v) ∝ r_i(v)·exp(s_i(v) − s_i(x̂_i))` (eq. 15–18). La correction n'est
appliquée qu'aux `k` tokens les plus probables du professeur et de l'élève (union), et
seulement si le rollout passe le vérificateur de la tâche (§4.2).

**Résultats annoncés.** Sur Qwen3 de 0,6B à 8B, d-OPD dépasse OPDLM de **2,6 à 4,0 points**
de moyenne sur six benchmarks et atteint la performance d'OPDLM **1,35 à 1,58× plus vite**
en temps mural (§5). À taille de modèle fixée (`N = 4`), la plage est **2,6–3,9 pts** ; le
maximum de 4,0 pts est atteint par Qwen3-8B à `N = 8` (§5.1). Ces chiffres sont des
**cibles de lecture**, pas des résultats locaux.

**Ce que fait Academiia en J12.** Un module `modules/dopd/` (paquet `academiia.dopd`) qui
contient : ce brief, une configuration conforme aux valeurs par défaut du papier, et une
implémentation de référence **NumPy pur, CPU** de la correction future-aware. Aucun import
de `academiia.a2d` (ADR-001). Aucun entraînement.

---

## 1. Résumé du papier (une page maximum)

### 1.1 Contexte

Les LLM autoregressifs génèrent un token à la fois ; les modèles de diffusion textuelle
génèrent plusieurs tokens en parallèle. Les dLLM **par blocs** conservent une structure
causale entre blocs et autorisent des interactions bidirectionnelles **à l'intérieur** du
bloc courant. Cette structure est un bon compromis : elle préserve la qualité d'un modèle AR
tout en accélérant la génération.

Plutôt que d'entraîner un dLLM depuis zéro, la voie étudiée consiste à **convertir** un
checkpoint AR pré-entraîné en dLLM par distillation : le modèle AR sert de professeur gelé,
et l'élève apprend à générer par blocs. La distillation sur politique (OPD) supervise
l'élève sur les états qu'il visite lui-même, ce qui réduit le décalage entre entraînement et
génération. **OPDLM** (Su et al., 2026, arXiv:2606.06712) applique cette idée à la conversion
AR → dLLM par blocs.

### 1.2 Le manque identifié

OPDLM aligne les **états** (l'élève est supervisé sur ses propres trajectoires) mais pas les
**informations conditionnantes**. À une position masquée `i` du bloc actif, l'élève conditionne
sur tout le bloc partiellement visible `X_t`, y compris des tokens à droite de `i`. Le
professeur AR ne conditionne que sur le préfixe causal. La cible de distillation est donc
définie avec **moins d'information** que n'en a l'élève.

### 1.3 La méthode

d-OPD construit une cible « future-aware » en deux temps :

1. **Cible exacte** : posterior du professeur AR conditionné sur l'état visible complet du
   bloc (eq. 7–8). Pour un préfixe compatible fixé, le posterior s'écrit comme la
   distribution causale multipliée par un terme de **compatibilité du futur** (eq. 10).
2. **Approximation tractable** : le rollout complété fournit un préfixe `Â_i` et un suffixe
   `Ĉ_i`. On remplace la marginalisation par cette réalisation unique :
   `p̃_T,i(v) ∝ r_i(v)·exp(Δs_i(v))`, avec
   `r_i(v) = p_T(v | q, z, Â_i)`,
   `s_i(v) = log p_T(Ĉ_i | q, z, Â_i, v)` et
   `Δs_i(v) = s_i(v) − s_i(x̂_i)` (eq. 11–18).

La normalisation par le token de référence `x̂_i` est essentielle : un facteur de correction
de 1 doit être **neutre** pour les tokens non scorés, alors qu'une vraisemblance absolue de
séquence est généralement très inférieure à 1. Le facteur de référence s'annule dans la
normalisation si l'on corrige tout le vocabulaire.

En pratique, seuls les `k` tokens les plus probables du professeur et de l'élève (union) sont
scorés ; les autres gardent un facteur 1. Un **vérificateur de tâche** décide si la
correction s'applique : un rollout incorrect reçoit la supervision causale standard, mais
n'est **pas** jeté de l'entraînement. Le reste de la procédure (rollout, échantillonnage d'un
état de débruitage par bloc, KL avant, optimiseur) est **identique à OPDLM**.

### 1.4 Résultats annoncés

Sur Qwen3-0.6B, 1,7B, 4B et 8B, avec des blocs de 4 tokens (et 8/16 pour le 8B), d-OPD
améliore la moyenne de six benchmarks de **2,6 à 4,0 points** par rapport à OPDLM, et réduit
le temps mural pour atteindre la performance d'OPDLM de **1,35 à 1,58×**. À `N = 4`, la
plage par échelle est **2,6–3,9 pts** ; les **4,0 pts** correspondent à Qwen3-8B à `N = 8`,
et le gain n'est pas monotone en `N` (3,8 pts à `N = 16`). Dans un diagnostic
contrôlé où la cible exacte est énumérable (49 948 marginaux), la KL moyenne au posterior
exact passe de **0,4444** (cible causale) à **0,1249** (cible future-aware), soit **−71,9 %**
(§3.3, App. D). Ces chiffres sont ceux du papier ; voir §5 pour les tableaux et §7 pour ce
qui n'est pas reproduit.

---

## 2. Problème adressé : le *target-conditioning mismatch*

### 2.1 Notation

| Symbole | Définition | Source |
|---|---|---|
| `q` | question / prompt | §3.1 |
| `z` | contexte causal complété précédant le bloc actif | §3.1 |
| `N` | taille de bloc | §3.1 |
| `X = (x_1, …, x_N)` | bloc actif complet | eq. (1) |
| `X_t` | état partiellement masqué du bloc au pas de débruitage `t` | eq. (2) |
| `p_θ,i` | distribution de l'élève à la position `i` | §3.1 |
| `p_T` | distribution du professeur AR gelé | §3.1 |
| `x̂ = (x̂_1, …, x̂_N)` | bloc complété par le rollout de l'élève | eq. (4) |
| `r_i(v)` | cible causale `p_T(v | q, z, x̂_<i)` | eq. (5) |
| `O_i` | tokens visibles à droite de `i` dans `X_t` | eq. (10) |
| `Ĉ_i` | suffixe complété `(x̂_{i+1}, …, x̂_N)` | eq. (13) |
| `s_i(v)` | log-vraisemblance du suffixe complété sous le candidat `v` | eq. (14) |
| `V_i` | ensemble de candidats (union des top-k professeur/élève) | §4.2 |

### 2.2 Le décalage

À l'état `X_t`, l'élève calcule

```
p_θ,i(· | X_t, q, z)                                      (eq. 3)
```

et peut donc utiliser les tokens visibles à gauche **et à droite** de `i`. La cible OPDLM
standard est

```
r_i(v) = p_T(v | q, z, x̂_<i)                              (eq. 5)
```

Elle ne dépend que du préfixe causal. Si `x_i` est masqué alors que des tokens à sa droite
sont déjà visibles, l'élève dispose d'une information que la cible ignore. Le problème n'est
donc **pas** résolu par l'alignement d'états d'OPD : même sur l'état exact visité par
l'élève, la cible du professeur est conditionnée sur une information différente.

### 2.3 Ce que dit le diagnostic du papier

Le papier isole ce décalage dans un cadre où le posterior exact est calculable par
énumération (App. D) : le support du professeur est restreint à un vocabulaire fini par
position, puis tous les blocs complets sont énumérés. Sur 49 948 marginaux masqués, la cible
causale a une KL moyenne de **0,4444** au posterior exact, contre **0,1249** pour la cible
future-aware. Ce diagnostic mesure la **cible**, pas la performance aval ; il est cohérent
avec l'hypothèse que le futur visible apporte une information pertinente.

---

## 3. Idée centrale : correction future-aware

### 3.1 Cible exacte

Le professeur AR définit une distribution jointe sur le bloc :

```
p_T(X | q, z) = ∏_{i=1..N} p_T(x_i | q, z, x_<i)          (eq. 6)
```

Conditionner cette jointe sur les tokens visibles de `X_t` donne le posterior

```
π(X | X_t, q, z) = p_T(X | q, z)·1{X ⪰ X_t}
                   / Σ_{X'} p_T(X' | q, z)·1{X' ⪰ X_t}     (eq. 7)
```

et, au niveau token,

```
π_i(v | X_t, q, z) = Σ_{X : x_i = v} π(X | X_t, q, z)       (eq. 8)
```

Pour un préfixe compatible `A_i` fixé, la règle de Bayes sur la factorisation AR donne le
point clé :

```
π_i(v | X_t, q, z, A_i)
  = p_T(v | q, z, A_i) · p_T(O_i | q, z, A_i, v)
    / Σ_{v'} p_T(v' | q, z, A_i) · p_T(O_i | q, z, A_i, v')   (eq. 10)
```

Le premier facteur est le **prior causal** ; le second est le **terme de compatibilité du
futur**. Si aucun futur n'est visible (`O_i = ∅`), ce terme est constant en `v` et la cible
se réduit au prior causal. Si tout le préfixe gauche est visible, la marginalisation
extérieure disparaît. En général, la cible exacte doit intégrer l'incertitude sur le préfixe
**et** sur le futur non observé — d'où l'approximation.

### 3.2 Approximation par le rollout complété

d-OPD utilise le bloc `x̂` déjà produit par le rollout en cours :

- préfixe complété : `Â_i = x̂_<i` ;
- prior causal : `r_i(v) = p_T(v | q, z, Â_i)` (eq. 11) ;
- suffixe complété : `Ĉ_i = (x̂_{i+1}, …, x̂_N)` (eq. 13) ;
- score futur : `s_i(v) = log p_T(Ĉ_i | q, z, Â_i, v)` (eq. 14).

La cible approximée est

```
p̃_T,i(v) = r_i(v)·exp(s_i(v)) / Σ_{v'} r_i(v')·exp(s_i(v'))   (eq. 15)
```

### 3.3 Normalisation par le token de référence

Scorer tout le vocabulaire est trop coûteux. On ne score qu'un ensemble `V_i` de candidats.
Laisser les tokens non scorés avec un facteur 1 revient à leur appliquer une correction
**absolue** de 1, alors que les tokens scorés reçoivent `exp(s_i(v))`, généralement très
petit : les tokens non scorés seraient artificiellement favorisés. On exprime donc chaque
score relativement au token réellement produit par le rollout :

```
Δs_i(v) = s_i(v) − s_i(x̂_i)                                (eq. 16)
exp(Δs_i(v)) = p_T(Ĉ_i | q, z, Â_i, v)
               / p_T(Ĉ_i | q, z, Â_i, x̂_i)                 (eq. 17)
```

et

```
p̃_T,i(v) = r_i(v)·exp(Δs_i(v))
           / Σ_{v'} r_i(v')·exp(Δs_i(v'))                  (eq. 18)
```

Le facteur de référence s'annule si l'on normalise sur tout le vocabulaire ; il rend le
facteur 1 **neutre** pour les tokens non scorés. Concrètement : les candidats de `V_i` sont
repondérés par `exp(Δs_i(v))`, les autres gardent un facteur 1, puis on renormalise.

### 3.4 Ensemble de candidats et vérificateur

- **Candidats** : union des top-`k` du professeur et des top-`k` de l'élève. Le papier prend
  `k = 16` par défaut. La Table 3 mesure la masse couverte **sous chaque distribution**
  pour un ensemble de `k` candidats : 99,45 % (professeur) et 99,44 % (élève) à `k = 16`,
  97,87 % / 95,43 % à `k = 4`. Le surcoût d'entraînement mesuré est de **2,63 %** à
  `k = 16` (Table 3, App. E).
- **Vérificateur** : la correction n'est appliquée que si le rollout complété passe le
  vérificateur de la tâche. Un rollout incorrect reçoit la cible causale standard ; il reste
  dans l'entraînement. Le papier rapporte 58,8 → 59,8 de moyenne (8B, N = 4) en activant la
  porte (Table 3, droite).

### 3.5 Algorithme (App. A, Algorithm 1)

1. Générer des réponses avec l'élève et enregistrer les états de débruitage visités.
2. Obtenir le bloc complété `x̂` et l'étiquette de correction `R(x̂)`.
3. Échantillonner **un** état de débruitage par bloc généré.
4. Pour chaque position masquée `i` de l'état échantillonné :
   - calculer `r_i(v)` avec le préfixe complété `Â_i` ;
   - si `R(x̂) = 1` **et** qu'un futur est visible à droite de `i` :
     - former `V_i` (union des top-k) et le suffixe `Ĉ_i` ;
     - calculer `s_i(x̂_i)` puis `s_i(v)` pour `v ∈ V_i` ;
     - calculer `Δs_i(v)` et repondérer/renormaliser ;
   - sinon, garder `p̃_T,i = r_i`.
5. Moyenner les pertes token par token sur les positions masquées et mettre à jour l'élève
   avec la KL avant `D_KL(p̃_T,i ‖ p_θ,i)` (eq. 19).

### 3.6 Détails d'implémentation utiles (App. B)

- Le préfixe causal `(q, z, Â_i)` est **partagé** par toutes les branches de candidats : il
  peut être mis en cache (KV cache) et réutilisé ; seul le suffixe dépend du candidat.
- Pour `k = 16`, un job de scoring contient au plus **33 branches** : la branche de référence
  `x̂_i` plus les candidats distincts de l'union (≤ 32).
- Le scorer est chargé dans le dtype natif du checkpoint (BF16 pour les expériences Qwen3) ;
  logits, log-probabilités, sommes de log-vraisemblance, différences et normalisation sont
  calculés en **float32**.
- La correction n'est **validée** que si le normaliseur plein vocabulaire est fini et
  strictement positif ; sinon on conserve `r_i`. Les écarts finis ne sont pas écrêtés ; les
  zéros exacts restent nuls (représentés `-∞` en log-probabilité).
  Conséquence directe : un delta fini assez grand pour que `exp(Δs)` déborde, ou
  assez négatif pour que **tous** les poids sous-débitent à zéro, rend le normaliseur
  invalide et déclenche le **fallback causal**. Un delta `-∞` isolé (suffixe de
  probabilité nulle) est légitime : il annule le poids de ce seul candidat.
- Un score candidat ou une référence non fini rend le normaliseur non fini : le
  fallback causal s'applique également (pas d'exception).
- Si le cache de préfixe étendu dépasse **16 GiB**, le lot de branches est réduit.

### 3.7 Entraînement (App. C.2)

- Un passage sur le mélange OPDLM : **61 816 prompts**, **483 itérations** on-policy,
  8 × NVIDIA H100 80 Go.
- Jusqu'à 128 prompts par itération, un rollout par prompt ; température 1,0, top-p 1,0,
  top-k 0 ; 4 pas de débruitage par bloc ; longueur de réponse maximale 4 000 tokens, avec
  une limite globale qui croît de 100 à 4 000 selon un cosinus sur les 100 premières
  itérations.
- Un état de débruitage par bloc, tiré uniformément parmi les tours visités où le bloc a au
  moins une position masquée ; les états sont combinés en un exemple au niveau trajectoire.
- AdamW, lr `1e-5`, β = (0,9 ; 0,999), ε = `1e-8`, weight decay 0, warmup 5 itérations,
  décroissance cosinus jusqu'à 10 %, clipping de norme globale 1,0, BF16, TF32, gradient
  checkpointing, DeepSpeed ZeRO-3 avec paramètres et états d'optimiseur sur CPU.

### 3.8 Évaluation (App. C.4)

Six benchmarks : MMLU, MMLU-Pro, GPQA-Diamond, GSM8K, MATH500, AIME25. Ensembles complets
(14 042 ; 12 032 ; 198 ; 1 319 ; 500 ; 30). Même taille de bloc que l'entraînement, quatre
pas de débruitage par bloc, remasquage statique à faible confiance, température 1,0,
top-p 1,0, top-k 1, jusqu'à 4 096 nouveaux tokens, zéro-shot avec le template de chat Qwen3.
Checkpoints évalués aux itérations 50, 100, …, 450 et 483 ; on retient la meilleure moyenne
six benchmarks, sans sélection par benchmark.

---

## 4. Différences clés vs OPDLM

| Aspect | OPDLM | d-OPD |
|---|---|---|
| États d'entraînement | états visités par l'élève (on-policy) | **identiques** : mêmes rollouts, même échantillonnage d'état |
| Cible du professeur | `r_i(v) = p_T(v | q, z, x̂_<i)` | `p̃_T,i(v) ∝ r_i(v)·exp(Δs_i(v))`, correction sur `V_i` |
| Information future visible | ignorée | utilisée via la vraisemblance du suffixe complété |
| Candidats | tout le vocabulaire (ou top-k selon l'implémentation) | union des top-k professeur/élève, `k = 16` par défaut |
| Porte de correction | — | vérificateur de tâche ; fallback causal si rollout incorrect |
| Objectif | KL avant `D_KL(r_i ‖ p_θ,i)` | KL avant `D_KL(p̃_T,i ‖ p_θ,i)` |
| Architecture / données / optimiseur | référence | **inchangés** |
| Coût supplémentaire | — | scoring du suffixe pour les candidats ; **2,63 %** de surcoût à `k = 16`, moyenne **non pondérée sur les 4 échelles** de l'augmentation du temps moyen par itération (Table 3, App. E) |
| Effet du bloc | — | le gain **persiste** quand `N` augmente : +3,9 / +4,0 / +3,8 pts sur 8B pour `N = 4 / 8 / 16` ; maximum **4,0 pts** à `N = 8` (Table 1) |

**Ce que d-OPD n'est pas.** Ce n'est pas une méthode d'arithmétique de poids, ni un
transfert de vecteur de tâche, ni un merge. Il n'y a ni `τ_A`, ni `τ_D`, ni coefficient
`α`/`λ`. C'est une **modification de la cible de distillation** pendant l'entraînement. Le
module `dopd` ne doit donc pas importer `academiia.a2d` : il n'y a pas de règle A2D à
réutiliser (ADR-003), et les deux méthodes sont indépendantes.

---

## 5. Cibles chiffrées du papier (à citer, non reproduites)

### 5.1 Table 1 — moyenne six benchmarks

Extrait des lignes Qwen3 (Table 1, §5.1). Les colonnes complètes sont MMLU, MMLU-P,
GPQA-D, GSM8K, MATH500, AIME25 ; seule la moyenne est recopiée ici.

| Modèle / réglage | OPDLM | d-OPD | Gain |
|---|---:|---:|---:|
| Qwen3-0.6B, N = 4 | 25,2 | **27,8** | **+2,6** |
| Qwen3-1.7B, N = 4 | 39,3 | **42,3** | **+3,0** |
| Qwen3-4B, N = 4 | 51,5 | **55,1** | **+3,6** |
| Qwen3-8B, N = 4 | 55,9 | **59,8** | **+3,9** |
| Qwen3-8B, N = 8 | 46,3 | **50,3** | **+4,0** |
| Qwen3-8B, N = 16 | 43,1 | **46,9** | **+3,8** |

Lecture : la plage annoncée dans l'abstract et l'introduction est **2,6–4,0 points**. Le
gain n'est pas limité à une taille de modèle, et il persiste quand `N` augmente. Il
n'est **pas monotone** en `N` : 3,9 pts à `N = 4`, 4,0 pts à `N = 8`, 3,8 pts à
`N = 16`. À taille de modèle fixée (`N = 4`), la plage est **2,6–3,9 pts** ; le
maximum de 4,0 pts n'est atteint que par Qwen3-8B à `N = 8` (§5.1).

### 5.2 Table 2 — temps mural jusqu'à la performance d'OPDLM

| Échelle | OPDLM (h) | d-OPD (h) | Accélération |
|---|---:|---:|---:|
| Qwen3-0.6B | 18,39 | 13,15 | **1,40×** |
| Qwen3-1.7B | 19,92 | 14,75 | **1,35×** |
| Qwen3-4B | 30,72 | 22,34 | **1,38×** |
| Qwen3-8B | 28,26 | 17,91 | **1,58×** |

Plage annoncée : **1,35–1,58×**. C'est un *time-to-matched-performance* en temps mural, pas
un débit d'inférence.

### 5.3 Table 3 — ablations

- **Taille des candidats** : couverture professeur/élève 97,87/95,43 % à `k = 4` ;
  99,45/99,44 % à `k = 16` ; surcoût 2,44 % à `k = 4`, **2,63 %** à `k = 16`, 6,02 % à
  `k = 64`. `k = 16` est le compromis retenu.
  *Portée* : ces pourcentages de couverture sont mesurés **sous chaque
  distribution** (professeur et élève séparément) pour un ensemble de `k`
  candidats ; la Table 3 ne rapporte pas la masse couverte par l'**union**
  elle-même, qui est au moins celle des deux (§4.2 : l'ensemble de candidats
  est l'union des top-k). Le surcoût 2,63 % est la moyenne **non pondérée sur
  les 4 échelles** de l'augmentation relative du temps moyen par itération
  (App. E), et non un surcoût uniforme par échelle.
- **Porte de correction** (8B, N = 4) : OPDLM 55,9 ; d-OPD sans porte 58,8 ; d-OPD avec
  porte **59,8**.

### 5.4 Diagnostic de cible (App. D)

49 948 marginaux token, 10 configurations `(N, V)`, 400 blocs par configuration, quatre
niveaux de bruit `σ ∈ {0 ; 0,5 ; 1,0 ; 1,5}`. KL moyenne au posterior exact : **0,4444**
(causal) → **0,1249** (future-aware), **−71,9 %**.

### 5.5 Ce qu'il faut retenir pour Academiia

Ces chiffres supposent **8 × H100 80 Go**, 61 816 prompts et 483 itérations. La machine
locale (RTX 5090 32 Go) ne permet pas une reproduction à l'échelle. Toute expérience locale
devra être **déclarée comme réduite** (sous-ensemble de prompts, budget, seed, IC) et ne
pourra pas être présentée comme une reproduction du papier.

---

## 6. Décisions d'architecture Academiia

### 6.1 Périmètre du module

**Dedans (J12 et suite) :**

- la **cible future-aware** : calcul des deltas relatifs, renormalisation, fallback,
  ensemble de candidats (implémenté en NumPy pur) ;
- la **configuration** des hyperparamètres du papier et leur sérialisation pour les
  manifestes (implémentée) ;
- les **interfaces** de scoring professeur, de rollout, de vérificateur et de politique
  élève (spécifiées ; implémentation J13+) ;
- les tests CPU de la logique de cible et de la règle de dépendance ;
- plus tard : le harnais d'entraînement réduit sur **Qwen3-0.6B**, puis l'évaluation.

**Dehors (explicite) :**

- tout entraînement, tout rollout de modèle, tout téléchargement de checkpoint en J12 ;
- la reproduction des six benchmarks et des chiffres du papier ;
- l'arithmétique de poids A2D (le module n'importe pas `academiia.a2d`) ;
- un module `modules/opdlm/` (nom interdit) ;
- les portes G1/G2/G3 et tout résultat A2D.

### 6.2 Interfaces minimales attendues

| Interface | Rôle | Statut | Code |
|---|---|---|---|
| `DOPDConfig` | hyperparamètres et valeurs par défaut du papier (`N`, `k`, porte, dtype) | **implémenté** (J12) | `dopd/config.py` |
| `compute_future_aware_target` | correction `p̃_T,i` sur un vecteur de probabilités déjà calculé | **implémenté** (J12) | `dopd/target.py` |
| `candidate_union` | union des top-k professeur/élève, sans doublon | **implémenté** (J12) | `dopd/target.py` |
| `TeacherScorer` | `causal_prior(...)` et `suffix_loglikelihood(...)`, batché et caché | **contrat implémenté** (J13) | `dopd/interfaces.py` |
| `RolloutRecorder` | états visités, premier temps de démasquage, états échantillonnés | **contrat implémenté** (J13) | `dopd/interfaces.py` |
| `TaskVerifier` | étiquette de correction `R(x̂)` par tâche | **contrat implémenté** (J13) | `dopd/interfaces.py` |
| `LineageWriter` | `lineage.json` au schéma ADR-002, `producer.module = "dopd"` | **contrat implémenté** (J13) | `dopd/interfaces.py` |
| `StudentPolicy` | log-probabilités de l'élève aux positions masquées | spécifié, non implémenté | — |
| `DOPDTrainer` | boucle on-policy, échantillonnage d'état, KL, optimiseur | hors J13 | — |

**Portée de J13.** Les quatre contrats de la colonne « contrat implémenté »
sont des **classes abstraites** (`abc.ABC`) assorties de types de données
figés et de quelques helpers déterministes :

- `TeacherScorer.score(request)` fait **un** appel de prior et **un** lot de
  branches (les candidats `V_i` plus la branche de référence `x̂_i`, soit au
  plus `2k + 1 = 33` branches à `k = 16`, App. B) ;
- `RolloutRecorder.sample_state(block_index, rng)` tire **un** état de
  débruitage parmi ceux qui ont au moins une position masquée (App. C.2) ;
- `TaskVerifier.gate_open(completion, config=…)` applique la porte : ouverte
  si `correctness_gate` est actif **et** que le vérificateur accepte ;
- `LineageWriter.build_manifest(...)` construit un manifeste ADR-002 avec
  `producer.module = "dopd"`, `producer.version`, le commit git
  (`academiia.core.git_commit`) et la commande exacte.

Aucun scoring réel, aucun rollout réel, aucun entraînement : les
implémentations concrètes vivent **hors** du paquet pur (règle ADR-001 : ni
`torch`, ni `transformers` dans `academiia.dopd`).

**Types de données associés** (`dopd/interfaces.py`) : `ScoringRequest`
(préfixe causal partagé, suffixe `Ĉ_i`, candidats `V_i`, token de référence
`x̂_i`), `ScoringResult` (prior `r_i(v)` plein vocabulaire + scores `s_i(v)`),
`DenoisingState` (état `X_t` et positions masquées), `CompletedBlock` (bloc
complété `x̂`), `ArtifactRef` et `LineageManifest`.

**Note de nommage.** Le brief désignait le vérificateur `Verifier` ; le code
retient `TaskVerifier`, plus explicite et sans collision avec le module
`academiia.core.text.degenerate`. Le rôle est inchangé.

**Contrat de la fonction de cible.** Elle reçoit une distribution causale **déjà calculée**
(vecteur de probabilités), les indices des candidats, leurs log-vraisemblances de suffixe et
la log-vraisemblance de référence. Elle ne fait **aucun** forward, ne connaît ni torch ni un
tokenizer, et ne persiste rien. C'est ce qui permet de la tester sur CPU avec des
distributions jouets.

Deux paramètres de contrat explicites :

- le **dtype d'arithmétique** (`arithmetic_dtype`, défaut `float32` = App. B) est
  celui dans lequel sont calculés les deltas, les facteurs `exp(Δs)`, les poids
  plein vocabulaire et le normaliseur. Il est partagé avec `DOPDConfig.arithmetic_dtype`,
  qui n'est donc pas une valeur descriptive mais le **défaut effectif** de la
  fonction ;
- le **fallback** est un résultat, pas une exception : tout état numérique qui
  invalide le normaliseur plein vocabulaire (débordement, sous-débitement total,
  score ou référence non fini) rend la cible causale renormalisée, avec le motif
  correspondant (`REASON_FALLBACK` ou `REASON_NON_FINITE_SCORES`).

### 6.3 Modèle de développement : Qwen3-0.6B

Le plus petit modèle du papier est **Qwen3-0.6B** : même famille que les autres échelles,
poids ouverts, empreinte compatible avec une exploration locale. Le papier rapporte pour ce
modèle 25,2 (OPDLM) → 27,8 (d-OPD) de moyenne six benchmarks (Table 1) ; c'est un **point de
comparaison**, pas une cible locale.

Décisions :

- le **professeur** est le checkpoint AR Qwen3-0.6B **gelé** ; il sert à la fois de prior
  causal et de scorer de compatibilité futur (comme dans le papier) ;
- l'**élève** est initialisé depuis le même checkpoint AR, puis converti en dLLM par blocs ;
- la conversion et le code de rollout ne sont **pas** réimplémentés en J12 : ils seront
  évalués en J13 (réutilisation du code publié si la licence le permet, sinon
  réimplémentation documentée) ;
- aucun checkpoint Qwen3 n'est téléchargé en J12 ; aucun modèle > 0,6B n'est autorisé dans
  le périmètre de développement.

### 6.4 Relation avec `academiia.core` et ADR-001

```
core  ←  dopd
core  ←  a2d
a2d   ↛  dopd        (interdit)
dopd  ↛  a2d         (interdit)
```

- `academiia.dopd` n'importe **que** `academiia.core` (et des dépendances externes légères,
  aujourd'hui NumPy).
- Aucun `sys.path.insert`, aucun import par chemin de fichier.
- Un test statique (`modules/dopd/tests/test_dopd_package.py`) vérifie qu'aucun fichier du
  paquet n'importe un module du dépôt autre que `academiia.core` — et qu'aucun n'aménage
  `sys.path`.
- Le paquet est exposé par `pyproject.toml` (`package-dir` + `packages`), pas par un
  aménagement de `PYTHONPATH`.

### 6.5 Manifeste de lignée (ADR-002)

d-OPD n'écrit pas de `merge_config.json` (ce n'est pas une fusion). Tout artefact produit
(adaptateur, checkpoint d'élève, résultat d'évaluation) devra écrire un `lineage.json` au
schéma ADR-002, avec :

- `schema: "academiia.lineage/1"` ;
- `artifact.kind` : `checkpoint` / `adapter` / `eval` / `report` ;
- `parent` : le checkpoint AR Qwen3-0.6B (ou le checkpoint d'élève précédent) ;
- `producer.module: "dopd"`, version du paquet, commit git, commande exacte ;
- `config` : `DOPDConfig` sérialisée, seed, taille de bloc, `k`, porte, dtype ;
- `data` : hash du mélange de prompts et du vérificateur ;
- `verification` : type de contrôle (smoke de chargement, évaluation, etc.).

Le schéma est **figé mais non implémenté** (ADR-002) ; J12 ne fait que le citer.

### 6.6 Tests CPU et critères de sortie de J12

Les tests du squelette vérifient :

1. `import academiia.dopd` fonctionne après `pip install -e .` ;
2. aucun import d'un module du dépôt autre que `academiia.core`, aucun aménagement de
   `sys.path`, aucun import de `torch` dans le paquet ;
3. les valeurs par défaut de `DOPDConfig` sont celles du papier (N = 4, k = 16, porte
   active, scorer BF16, arithmétique float32) et `arithmetic_dtype` est bien le
   **défaut effectif** de la fonction de cible ;
4. la cible future-aware :
   - sans futur visible ou porte fermée, la cible est **exactement** le prior causal ;
   - le token de référence a un delta nul ;
   - les tokens hors candidats gardent un facteur 1 (vérifié par calcul manuel) ;
   - la cible est invariante à un décalage constant de tous les scores ;
   - les zéros exacts restent nuls ;
   - un candidat de score `-∞` a un poids nul, sans annuler la correction ;
   - un **débordement** du facteur de correction (delta fini assez grand) et un
     **sous-débitement total** (tous les poids nuls) déclenchent le fallback causal ;
   - un score candidat ou une référence non fini déclenche le fallback causal, sans
     lever d'exception ;
   - un prior accepté mais non exactement normalisé est renormalisé (la cible somme 1) ;
   - le chemin `float64` est vérifié séparément à haute précision ;
5. les entrées invalides (prior négatif, candidat hors vocabulaire, doublon, longueurs
   incohérentes, dtype d'arithmétique inconnu) échouent explicitement.

Ces tests ne prouvent **pas** que d-OPD améliore un modèle ; ils prouvent que la brique de
cible est mathématiquement cohérente et que le squelette respecte les règles du dépôt.

### 6.7 Étapes proposées après J12

| Étape | Contenu | État |
|---|---|---|
| **J13** | contrats d'interface (`TeacherScorer`, `RolloutRecorder`, `TaskVerifier`, `LineageWriter`) et smoke de chargement Qwen3-0.6B, sans entraînement | **fait** |
| **J14** | diagnostic Annexe D (support restreint exact, KL, bootstrap) ; porte scientifique franchie **avec réserves de lecture** | **fait** |
| **J15** | diagnostic multi-prompt (20 prompts) et mesure de l'écart cible exacte vs top-k = 16 | **fait** |
| **J16** | audit de J15, corrections A–E, reproductibilité des champs de porte ; décision **GO conditionnel** | **fait** |
| **J17+** | protocole pré-enregistré du `TeacherScorer` réel, puis `StudentPolicy` et boucle réduite | à missionner |
| **Plus tard** | tailles 1,7B/4B/8B, blocs 8/16, reproduction des six benchmarks | GPU adapté (8 × H100 ou équivalent) |

Aucune de ces étapes n'est autorisée par le seul fait que J12 est terminé : chacune demande
une mission et, pour J17+, un budget GPU mesuré.

### 6.8 Risques et points ouverts

- **Échelle** : la machine locale ne permet pas 8 × H100. Toute mesure locale est réduite et
  doit être présentée comme telle.
- **Conversion AR → dLLM** : le papier s'appuie sur OPDLM et sur le code publié ; J12 ne
  tranche pas entre réutilisation et réimplémentation. Décision à prendre en J13, avec
  vérification de licence.
- **Vérificateur** : dépend des tâches et du mélange de prompts. Sans vérificateur, la porte
  doit être désactivée et le fallback causal documenté.
- **Approximation à une seule trajectoire** : le papier le reconnaît (App. F) ; moyenniser
  plusieurs futurs compatibles est une extension possible, hors J12.
- **Coût du scoring** : le cache de préfixe peut être volumineux ; le papier réduit le lot
  au-delà de 16 GiB. À instrumenter avant tout entraînement.
- **Reproductibilité** : seed, version du code, hash des données, dtype du scorer et
  paramètres de rollout doivent être consignés (règle 8 d'`AGENTS.md`).
- **Manifeste** : ADR-002 n'est pas implémenté ; J12 ne doit pas inventer un format
  concurrent. Le schéma est cité, pas dupliqué.

### 6.9 Porte scientifique J14 — diagnostic Annexe D (critères pré-enregistrés)

> **Statut** : critères figés le 2026-10-23, **avant** toute implémentation du
> diagnostic et tout premier run GPU J14. Une modification de cette section
> après un résultat rend la porte invalide et doit être traitée comme un
> nouveau protocole, pas comme une correction de rapport.

#### 6.9.1 Question et hypothèse

La porte J14 ne mesure pas une performance aval. Elle mesure si, dans un
cadre teacher AR fini où le posterior exactement conditionné est calculable,
la cible **future-aware à un seul futur échantillonné** réduit la divergence
KL de la cible causale par rapport à ce posterior.

La grandeur principale est la réduction appariée

```
ΔKL = KL(π_exact || r_causal) − KL(π_exact || r_future-aware)
```

Une valeur positive signifie que la cible future-aware est plus proche du
posterior exact. La KL est toujours `KL(π_exact || cible)`, conformément à
l'équation 28 de l'Annexe D.

#### 6.9.2 Protocole local réduit, figé avant exécution

Le diagnostic reproduit la construction de l'Annexe D, avec les écarts
suivants, tous déclarés :

* **Professeur** : `Qwen/Qwen3-0.6B`, gelé, en BF16, avec le même contexte
  causal fixe pour toutes les positions : `Question: What is the capital of
  France?\nAnswer:` ;
* **support énumérable** : construction séquentielle de l'Annexe D.1, avec
 restriction à `V` tokens par position et renormalisation du support fini ;
* **configurations** : `(N,V) ∈ {(4,6), (4,12), (6,6)}`. Le sous-ensemble est
  explicitement justifié : il couvre les deux valeurs `N` principales et deux
  restrictions `V` du papier, tout en gardant l'énumération et l'estimation
  exploitables sur un seul GPU. Les sept autres configurations du papier
  (`V=24/48`, `N=6,V=8/12/16`, `N=8`) sont hors budget local ; elles ne
  seront ni interpolées ni présentées comme reproduites ;
* **blocs** : 100 blocs complets par configuration ;
* **états masqués** : rétention indépendante de chaque position avec
  probabilité 0,45 ; si toutes les positions sont retenues, une position est
  masquée uniformément ;
* **perturbations** : les quatre niveaux `σ ∈ {0, 0,5, 1,0, 1,5}` de
  l'équation 30. Les mêmes états de base sont réutilisés pour les quatre
  niveaux ;
* **futur** : un seul bloc compatible est échantillonné par état et par
  niveau de bruit, puis utilisé pour construire les deux cibles ;
* **correction** : appliquée à **tout le support restreint** à la position,
  sans l'approximation top-k de l'entraînement ;
* **graine** : `20261023` pour le diagnostic ; `20261024` pour le bootstrap ;
* **bootstrap** : 10 000 rééchantillonnages, IC percentile 95 %. L'unité de
  rééchantillonnage est le **bloc de base**, afin de préserver la dépendance
  entre ses marginaux et ses quatre niveaux de bruit ; les blocs sont
  rééchantillonnés dans chaque configuration.

Le budget GPU maximal est de 2 heures pour un run complet. Un dépassement
interrompt le protocole plutôt que de réduire silencieusement le nombre de
blocs ou de configurations.

#### 6.9.3 Critères d'acceptation

La porte est **FRANCHIE** si et seulement si toutes les conditions suivantes
sont satisfaites :

1. **Critère primaire** : la réduction agrégée appariée `ΔKL` est strictement
   positive et son IC bootstrap 95 % **exclut zéro** (borne inférieure
   `> 0`). Ce critère est évalué sur tous les marginaux admissibles du
   sous-ensemble figé, avec IC calculé par bootstrap de blocs comme décrit
   ci-dessus.
2. **Complétude** : les résultats sont publiés pour chacune des trois
   configurations, pour chacun des quatre niveaux `σ`, puis agrégés. Aucun
   résultat négatif ou configuration n'est supprimé a posteriori.
3. **Effectif** : au moins 100 blocs de base par configuration et au moins
   2 000 marginaux token admissibles au total. Un effectif inférieur rend la
   porte **NON CONCLUANTE**, même si le critère primaire est favorable.
4. **Intégrité numérique** : aucun NaN/Inf dans les cibles, KL ou
   probabilités ; la somme de chaque distribution vaut 1 à `1e-5` près ; le
   support exact est vérifié par une énumération indépendante sur un petit
   cas de test. Un écart viole l'intégrité numérique et invalide le run.
5. **Traçabilité** : commande exacte, commit, version de PyTorch/Transformers,
   type de modèle, seed, mémoire GPU, temps mural et manifeste ADR-002 sont
   consignés dans le rapport et le résultat versionné.

La porte **n'est pas** un test d'égalité avec les chiffres `0,4444` et
`0,1249` du papier : le papier utilise Qwen3-8B, dix configurations et
49 948 marginaux. Ces valeurs restent des références de lecture, non des
seuils locaux.

#### 6.9.4 Décision en cas d'échec

* IC primaire contenant zéro, effectif insuffisant ou violation numérique :
  porte **NON FRANCHIE**. J14 s'arrête à la documentation ; aucune implémentation
  de `TeacherScorer` de production ni de `StudentPolicy`, et aucun début de
  boucle d'entraînement n'est autorisé avant une nouvelle mission.
* Critère primaire rempli mais une configuration ou un niveau `σ` présente
  une augmentation de KL : porte **FRANCHIE AVEC RÉSERVE** uniquement si
  l'agrégat reste positif avec IC excluant zéro ; la configuration en défaut
  et la réserve doivent être nommées avant J15.
* Porte franchie : J15 peut proposer un protocole réduit, après chiffrage du
  budget RTX 5090 et validation de la gouvernance. Le franchissement J14
  n'autorise ni entraînement ni modification de G1.

#### 6.9.5 Résultats J14 (exécution, après le gel des critères)

> **Les critères de §6.9.1 à §6.9.4 n'ont pas été modifiés après le run.**
> Le commit `66e9d55` fige les critères ; le commit `6a8a461` porte le code
> producteur ; le commit `0a7ebc9` porte le résultat versionné. La commande
> exacte, les empreintes des sources et le manifeste ADR-002 sont dans
> `modules/dopd/eval/annexe_d_qwen3_0.6b/results.json` ; le rapport est
> `modules/dopd/docs/reports/J14.md`.

**Décision : porte FRANCHIE, avec réserves de lecture** (`gate.status = "passed"`). La porte est franchie au sens des critères pré-enregistrés ; les réserves ci-dessous sont des réserves de **lecture** (échelle, variabilité inter-prompt, queue statistique), pas des réserves de porte. Elles sont assumées en J15 par un diagnostic multi-prompt et une mesure de l'écart top-k.

| Critère pré-enregistré | Seuil | Mesuré | Verdict |
|---|---|---|---|
| 6.9.3-1 critère primaire | `ΔKL` agrégé > 0 et IC 95 % excluant 0 | `ΔKL = +0,16234`, IC 95 % `[+0,07424 ; +0,22522]` | **atteint** |
| 6.9.3-2 complétude | 3 configurations × 4 niveaux `σ` + agrégat | publiés, aucune configuration supprimée | **atteint** |
| 6.9.3-3 effectif | ≥ 100 blocs/config. et ≥ 2 000 marginaux | 100 blocs pour chacune des 3 configurations, **3 056** marginaux | **atteint** |
| 6.9.3-4 intégrité numérique | aucun NaN/Inf, sommes à `1e-5`, énumération de contrôle | écart d'énumération `1,11e-16`, écart de somme `6,66e-16`, toutes valeurs finies | **atteint** |
| 6.9.3-5 traçabilité | commande, commit, versions, seed, mémoire, temps, manifeste | tous présents dans le JSON versionné ; commit `6a8a461` contenant le code producteur | **atteint** |

Aucune configuration ni aucun niveau `σ` ne dégrade la KL : **aucune réserve
n'est requise** au sens de §6.9.4.

Réduction de KL par configuration (IC 95 % bootstrap de blocs de base) :

| Configuration | Marginaux | KL causale | KL future-aware | `ΔKL` | IC 95 % |
|---|---|---|---|---|---|
| `N4_V6` | 860 | 1,1173 | 0,9404 | **+0,1769** | `[+0,1155 ; +0,2512]` |
| `N4_V12` | 880 | 1,3814 | 1,1889 | **+0,1924** | `[+0,1294 ; +0,2662]` |
| `N6_V6` | 1 316 | 0,6853 | 0,5526 | **+0,1327** | `[-0,0535 ; +0,2566]` |
| **Agrégat** | **3 056** | **1,0073** | **0,8450** | **+0,1623** | **`[+0,0742 ; +0,2252]`** |

Réduction de KL par niveau de bruit (toutes configurations) :

| `σ` | Marginaux | `ΔKL` | IC 95 % |
|---|---|---|---|
| 0,0 | 764 | +0,1749 | `[+0,1164 ; +0,2250]` |
| 0,5 | 764 | +0,1502 | `[+0,0319 ; +0,2269]` |
| 1,0 | 764 | +0,1729 | `[+0,1143 ; +0,2233]` |
| 1,5 | 764 | +0,1514 | `[+0,0325 ; +0,2288]` |

**Limites déclarées de la lecture.**

1. La configuration `N6_V6` a une IC 95 % qui **contient zéro** (`[-0,0535 ;
   +0,2566]`) : l'effet y est positif en moyenne mais **non concluant
   individuellement**. Le critère primaire porte sur l'agrégat, conformément à
   §6.9.3-1 ; ce résultat est une réserve de lecture, pas une réserve de porte.
2. L'échelle est réduite : `Qwen3-0.6B` au lieu de 8B, 3 configurations au
   lieu de 10, 100 blocs au lieu de 400, un seul prompt. Les chiffres du
   papier (KL `0,4444` → `0,1249`, −71,9 %) ne sont **pas** un seuil local et
   ne sont pas reproduits.
3. La cible future-aware est ici celle de l'**Annexe D** (correction sur tout
   le support restreint, sans top-k de la boucle d'entraînement) ; elle n'est
   pas la cible `k = 16` de l'entraînement d-OPD.
4. Le run est une **inférence seule** : 83,1 s mural, 1 374 MiB alloués CUDA
   au pic, 21 580 MiB RSS. Aucun poids modifié, aucun backward.

#### 6.9.6 Chiffrage du budget local (J14)

Le chiffrage est une **note de décision**, pas un protocole d'entraînement.
Les mesures (forwards sans gradient, Qwen3-0.6B, BF16, RTX 5090 32 Go) sont
versionnées dans `modules/dopd/eval/budget_rtx5090/bench_rtx5090.json`, le
harnais est `modules/dopd/harness/bench_budget_rtx5090.py`, et la note est
`modules/dopd/docs/BUDGET_RTX5090.md`.

| Poste | Résultat |
|---|---|
| Élève réaliste | **0,6B** (poids + gradients + AdamW ≈ 6,9 GiB) ; 4B–8B **impossible** sur 32 GiB (états d'optimiseur) |
| Longueur de rollout | **256 à 512** tokens, plafond de contexte 1 024 (le rollout de 4 000 tokens du papier est hors de portée : cache KV et scoring) |
| Lot | **32 rollouts** simultanés (1 399 tok/s mesurés ; le lot 64 ne rapporte que +69 % de débit pour 2× le cache) |
| Coût du scoring professeur | **mesuré** à géométrie égale (job réel : prior causal + score de suffixe) : **51,2 ms** par job avec cache de préfixe (`k=16`, contexte 1 024), **65,6 ms** à 33 branches, contre **167,9 ms** sans cache (gain **3,3×–5,6×**) |
| Contrôle numérique | Le même job rejoué en **FP32** : écart cache/rejeu de `3,1e-5` (score de suffixe) et `9e-6` (prior) — l'écart BF16 est un artefact de précision |
| Run significatif réduit | 8 000 prompts, rollouts 256 tokens → **16–21 h** sur une RTX 5090 |
| Verdict | **0,6B faisable en local** avec un protocole réduit ; l'échelle du papier (4B–8B) demande **8 × H100** ou une location |

Le chiffrage **n'autorise aucun entraînement** : la porte J14 reste une
porte scientifique, et le protocole d'entraînement relève de J15, après
pré-enregistrement.

#### 6.9.7 Diagnostic multi-prompt (J15) — critères pré-enregistrés

> **Pré-enregistrement** : cette section est figée **avant** tout run GPU
> multi-prompt. Sa modification après un résultat vaut nouveau protocole, pas
> correction de rapport (même principe que §6.9.5). Le commit des critères est
> ancêtre du commit du code producteur (`diagnostic_annexe_d.py` /
> `diagnostic.py`).

**Protocole.** Reproduction du §6.9.2 sur un jeu de prompts variés (≈ 20) au
lieu d'un seul prompt trivial : chaque prompt construit son propre support
restreint (mêmes configurations `(N,V) ∈ {(4,6),(4,12),(6,6)}`, 100 blocs,
niveaux `σ ∈ {0,0.5,1,1.5}`, rétention 0,45, graines `20261023` / `20261024`).
Les marginaux portent le label du prompt. L'agrégat primaire reste la moyenne
arithmétique de `ΔKL` (équation 28), mais quatre statistiques complémentaires
complètent le critère primaire pour caractéiser la variabilité inter-prompt et
la sensibilité à la queue :

1. **Critère primaire agrégé** : `ΔKL` agrégé > 0 et IC 95 % (bootstrap de
   blocs, clé `(prompt, config_index, block_index)`) exclutant zéro.
2. **Médiane de `ΔKL`** (per-marginal) > 0 : robuste à la queue, contre-preuve
   du critère fondé sur la moyenne (règle de lecture R4 de J14).
3. **Proportion de prompts significatifs** : fraction des prompts dont l'IC 95 %
   per-prompt exclut zéro — **≥ 0,5** (au moins la moitié des prompts montre un
   effet positif significatif).
4. **Test des signes** non paramétrique (approximation normale à deux queues)
   sur les `ΔKL` : valeur en p < 0,05 confirme l'effet directionnel sans
   hypothèse sur la forme de la distribution.
5. **Effectif** : ≥ 100 blocs/configuration et ≥ 2 000 marginaux (identique à
   §6.9.3-3).

**Décision pour J16** (enseignant le passage à un `TeacherScorer` réel) :

- **GO** si les critères 1 à 5 sont tenus ;
- **GO conditionnel** si le critère primaire (1) est tenu mais que la proportion
  de prompts significatifs (3) ou la médiane (2) est fragile — un professeur
  réel doit confirmer sur plus de prompts/échelles avant implémentation ;
- **NO-GO** si le critère primaire (1) n'est pas tenu.

Cette porte reste une **porte de lecture** : elle renforce la confiance dans la
cible future-aware, elle n'autorise aucun entraînement ni aucune implémentation
de `TeacherScorer`/`StudentPolicy`.

> **Note de lecture (ajoutée en J16, non normative).** Cette note ne modifie ni
> les seuils ni la décision pré-enregistrée ; elle lève deux ambiguïtés de
> lecture révélées par l'audit J15.
>
> 1. **Critère 3.** Il est évalué dans sa lecture stricte : proportion de
>    prompts **significativement positifs** — `14/20 = 0,70`, au-dessus du seuil
>    `0,5`. Le décompte historique « IC exclut zéro » (`15/20 = 0,75`) est
>    conservé comme **annotation** : il inclut un prompt significativement
>    négatif (`p18`). Les deux lectures dépassent le seuil.
> 2. **Critère 4.** Il est évalué sur les signes **par prompt** : test binomial
>    exact bilatéral `17/3` → `p = 0,00258`, donc `< 0,05`. Le `p ≈ 0` publié en
>    J15 est l'approximation normale sur les 35 040 marginaux à futur visible
>    (test différent, également significatif) ; il ne doit pas être présenté
>    comme « 20/20 prompts positifs ».

#### 6.9.8 Écart cible exacte vs cible top-k = 16 (J15)

**Question.** L'entraînement d-OPD tronque la cible à `k = 16` tokens ; le
diagnostic §6.9.7 corrige en revanche tout le support restreint, sans
approximation top-`k`. Avant d'investir dans un `TeacherScorer` de production,
on quantifie l'écart entre la cible complète et sa troncade à `k = 16`.

**Mesure.** Pour les `N` premiers marginaux vus (borne `2 048`), on compare par
position la distribution causale complète à sa troncade aux `k = 16` plus grandes
probabilités, renormalisée. Deux scalaires finis :

- `discarded_mass` : masse de probabilité perdue par la troncature
  (`1 − Σ des k plus grandes probabilités`) ;
- `tv_distance` : distance de variation totale entre la distribution complète et
  la version tronquée (0 = identique, 1 = disjoint).

**Interprétation.** Une truncation fidèle (`discarded_mass` et `tv_distance`
faibles) fait de `k = 16` une approximation raisonnable de l'objectif à support
complet : un `TeacherScorer` tronqué reste justifié. Un écart notable invalide
cette hypothèse et rend la décision conditionnelle. La KL par rapport au
posterior exact n'est pas calculée (infinie quand le support exact dépasse le
top-`k`) ; la distance de variation totale l'englobe et reste finie.

> **Correctif J16 (implémentation J15).**
>
> 1. **TV.** La version J15 de `truncate_causal_gap` calculait la TV contre la
>    troncature **non renormalisée** : elle valait donc exactement
>    `discarded_mass / 2`. La TV corrigée contre la troncature **renormalisée**
>    est `2,9534 %` en moyenne et `37,3190 %` au maximum ; elle est
>    **algébriquement égale** à `discarded_mass`. Les deux champs ne sont donc
>    **pas indépendants** : ils portent la même information. La valeur publiée
>    en J15 (`1,4767 %`) est exactement la moitié ; `discarded_mass` reste
>    correct et permet de reconstruire la TV corrigée.
> 2. **Borne `2 048`.** Le compteur était réinitialisé par configuration ; la
>    borne n'a jamais lié. `61 120` marginaux ont été mesurés (≈ 1 019 par
>    prompt et par configuration), soit un **sur-ensemble** du plan
>    pré-enregistré.
> 3. **Vocabulaire.** La distribution de référence est la **distribution
>    causale complète** (softmax des logits du dernier état), pas le posterior
>    exact énuméré. Le pré-enregistrement était correct ; le rapport J15
>    employait « posterior exact » à tort.


#### 6.9.9 Résultats J15 (exécution, après le gel des critères) — corrigés en J16

> **Les critères de §6.9.7 et §6.9.8 n'ont pas été modifiés après le run.** Le
> commit `4a74023` fige les critères ; `fe499ff` porte le code producteur
> (`diagnostic_annexe_d.py` / `diagnostic.py`) ; `01146ec` corrige les résidus
> J14 (`importorskip`) ; `d6a2a74` porte le résultat versionné et le rapport
> J15. La plage J15 est donc `1966111..d6a2a74` (**4 commits**, et non 2).
>
> **Artefacts.** `results.json` (sha256 `ed41ed7e…`) est **immuable** : il a été
> produit par `fe499ff` et n'est pas réécrit. Deux champs de porte y ont été
> ajoutés après coup par un script **non versionné** (`multiprompt.gate_j15_status`,
> `criteria`, `rationale`) ; ils sont **remplacés** par
> `derived_summary.json`, produit par
> `modules/dopd/harness/summarize_multiprompt.py` (CPU, sans torch) et
> vérifiable par `--check`. Le rapport corrigé est
> `modules/dopd/docs/reports/J15.md` ; l'audit est
> `modules/dopd/docs/reports/AUDIT_J15.md` ; les corrections sont détaillées
> dans `modules/dopd/docs/reports/J16.md`.

**Décision : GO conditionnel** (confirmée en J16). Le critère primaire
(§6.9.7-1) est atteint — l'effet future-aware est réel et significatif sur une
séquence de prompts variés ; la médiane nulle (§6.9.7-2) est la réserve qui
rend la décision **conditionnelle**. La correction J16 ne change pas la
décision : la lecture stricte du critère 3 (`14/20 = 0,70`) reste au-dessus du
seuil `0,5`, et le critère 4 reste significatif au test exact par prompt
(`p = 0,00258`).

Critères du diagnostic multi-prompt (§6.9.7, lecture corrigée J16) :

| Critère pré-enregistré | Seuil | Mesuré | Verdict |
|---|---|---|---|
| 6.9.7-1 critère primaire (`ΔKL` agrégé > 0, IC ∉ [0]) | IC ∉ [0] | `ΔKL = +0,0802`, IC 95 % `[+0,0638 ; +0,0962]` | **atteint** |
| 6.9.7-2 médiane de `ΔKL` > 0 | > 0 | **médiane = 0** (42,788 % de marginaux à `ΔKL ≈ 0`) | **non atteint** |
| 6.9.7-3 proportion de prompts **significativement positifs** ≥ 0,5 | ≥ 0,5 | **14 / 20 = 0,70** (lecture historique « IC exclut zéro » : 15/20 = 0,75, dont **1 négatif**) | **atteint** |
| 6.9.7-4 test des signes par prompt, p < 0,05 | p < 0,05 | **17 positifs / 3 négatifs** ; binomial exact `p = 0,00258` (le `p ≈ 0` publié est l'approximation normale sur 35 040 marginaux) | **atteint** |
| 6.9.7-5 effectif ≥ 100 blocs/config et ≥ 2 000 marginaux | seuil minimal | 100 blocs/config, **61 120** marginaux | **atteint** |

Réduction de KL par configuration (IC 95 % bootstrap de blocs de base) :

| Configuration | Marginaux | `ΔKL` moyen | IC 95 % | Temps mural |
|---|---:|---:|---|---:|
| `N4_V6` | 17 200 | +0,1045 | `[+0,0823 ; +0,1262]` | 764 s |
| `N4_V12` | 17 600 | +0,1181 | `[+0,0977 ; +0,1379]` | 806 s |
| `N6_V6` | 26 320 | +0,0390 | `[+0,0063 ; +0,0706]` | 1 660 s |

Les trois configurations sont positives et leur IC exclut zéro (contrairement à
J14, où `N6_V6` contenait zéro).

**Médiane nulle / marginaux sans futur visible.** La médiane n'est pas du
bruit : **42,788 %** des marginaux ont `ΔKL ≈ 0` — **26 080** exactement nuls
(aucun futur visible dans le bloc : la correction future-aware est
structuralement inactive) et **72** artefacts numériques (`< 1e-9`). C'est la
R4/R5 de J14 rendue quantitative sur une séquence de prompts variés : l'effet
agrégé (moyenne) est réel mais **porté par les 57,2 %** de marginaux dotés d'un
futur visible ; la médiane tombe à zéro car plus qu'un marginal sur deux est
structuralement nul. Le critère primaire fondé sur la moyenne arithmétique
(eq. 28) est donc **sensible à la queue**, confirmé ici sur 20 prompts.

**Test des signes et queue nuisible.** La phrase J15 « 20/20 prompts en
direction positive (aucun ne dégrade la KL) » était **fausse**. Le recalcul sur
les 61 120 marginaux donne :

| Mesure | Valeur |
|---|---:|
| Prompts à moyenne `ΔKL` positive | **17 / 20** |
| Prompts à moyenne `ΔKL` négative | **3 / 20** — `p06` (−0,0114), `p10` (−0,1098), `p18` (−0,2760) |
| Prompts significativement positifs (IC > 0) | **14 / 20** |
| Prompts significativement négatifs (IC < 0) | **1 / 20** — `p18`, IC `[−0,4151 ; −0,1381]` |
| Prompts non significatifs | 5 / 20 |
| Marginaux à futur visible **dégradés** | **4 661 / 35 040 = 13,30 %** |
| `ΔKL` moyen des marginaux dégradés | **−1,1863** (pire cas : **−28,7094**) |

La cohérence de direction est donc **majoritaire mais pas unanime** : la queue
favorable qui porte la moyenne s'accompagne d'une **queue nuisible** de 13,3 %.
C'est une réserve de lecture supplémentaire, qui ne renverse pas le critère
primaire mais interdit de présenter l'effet comme uniforme.

**Écart cible exacte vs top-k = 16 (§6.9.8)**, sur les 61 120 marginaux :

| Grandeur | Valeur | Interprétation |
|---|---:|---|
| Masse discardée moyenne (au-delà de k=16) | **2,9534 %** | la queue tronquée pèse ~3 % en moyenne |
| Masse discardée maximale | **37,3190 %** | cas extrêmes : plus d'un tiers de la masse hors du top-16 |
| TV corrigée moyenne (troncature renormalisée) | **2,9534 %** | identique à la masse discardée (relation algébrique) |
| TV publiée en J15 | **1,4767 %** | moitié de la TV réelle (troncature non renormalisée) |

La troncature à k = 16 est **fidèle en moyenne** (2,95 %) mais **hétérogène**
(max 37,3 %). Information de faisabilité avant tout investissement dans un
`TeacherScorer` : l'approximation top-k n'est pas neutre. La Table 3 du papier
rapporte une **couverture de masse** de 99,45 % (professeur) / 99,44 % (élève)
à `k = 16`, cohérente avec les ≈ 97 % locaux ; le papier ne publie **pas** de
TV, et la comparaison doit rester une comparaison de couverture, pas de « TV
contre le posterior exact ».

**Limites déclarées.**

1. **Réduction relative : 8,58 %** (`ΔKL / KL_causale = 0,0801906 / 0,9350779`).
   Le chiffre **16,12 %** est celui de **J14** (prompt unique) ; il ne doit pas
   être réutilisé pour J15. La comparaison au `−71,9 %` du papier n'a pas de
   base directe : le papier utilise Qwen3-8B, 10 configurations, 400 blocs,
   49 948 marginaux et un agrégat poolé sur 4 niveaux `σ` ; J15 utilise
   Qwen3-0.6B, 3 configurations, 100 blocs et 61 120 marginaux. La KL causale
   locale (`0,9351`) est **2,10×** celle du papier (`0,4444`).
2. L'effet est **moyenne-porté-par-queue** : médiane nulle (42,788 % de
   marginaux structurellement nuls) et queue nuisible de 13,3 %. Un
   `TeacherScorer` réel doit confirmer sur plus de prompts et plus d'échelles
   avant qu'un `StudentPolicy` soit justifié.
3. **Cache et règle 9.** Le run a lu le modèle dans
   `/home/sjteb/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/…`,
   **pas** dans `_artefact/hf-cache` (absent). L'écriture est inventoriée dans
   `modules/a2d/docs/reports/OUT_OF_WORKSPACE.md` ; la commande J15 qui affichait
   `HF_HOME=…` a été corrigée. Le harnais consigne désormais `hf_home`,
   `hf_home_source` et `model_source_in_repo`, et émet un avertissement si le
   modèle est lu hors du dépôt.
4. Le run est une **inférence seule** : **3 280 s** de temps mural total
   (3 230 s dans les trois configurations), soit ≈ 55 min ; 21 615 MiB RSS,
   1 607 MiB alloués CUDA au pic. Aucun poids modifié, aucun backward
   (`training.performed = false`).

---
---

### 6.10 `TeacherScorer` réel Qwen3-0.6B (J17) — critères pré-enregistrés

> **Pré-enregistrement** : cette section est figée **avant** l'implémentation du
> `TeacherScorer` et **avant** tout run GPU J17. Le commit des critères est
> ancêtre du commit du code producteur (`modules/dopd/harness/teacher_scorer_qwen3.py`
> / `modules/dopd/harness/run_teacher_scorer_smoke.py`). Toute modification de
> cette section après un résultat vaut nouveau protocole, pas correction de
> rapport (même principe que §6.9.7).

#### 6.10.1 Objet et périmètre

Livrer une implémentation **réelle** du contrat `TeacherScorer` (J13) à
l'échelle `Qwen/Qwen3-0.6B`, utilisable en inférence seule, conforme à la
géométrie de scoring mesurée en J14 (§6.9.6) :

- **préfixe causal partagé** `(q, z, Â_i)` mis en cache (KV cache), une seule fois ;
- **candidat variable** `v` inséré à la position `i`, une branche par candidat ;
- **suffixe partagé** `Ĉ_i = (x̂_{i+1}, …, x̂_N)` conditionnant le score.

Ce que J17 **ne fait pas** : aucun `StudentPolicy`, aucune boucle
d'entraînement, aucun backward, aucun optimizer, aucune écriture de checkpoint,
aucune modification des résultats, seuils ou verdicts G1, aucune release
`v0.1.0`, aucune échelle supérieure à 0,6B.

#### 6.10.2 Placement et contraintes architecturales (ADR-001)

| Élément | Emplacement | Contrainte |
|---|---|---|
| `TeacherScorer` de production | `modules/dopd/harness/teacher_scorer_qwen3.py` | `torch` / `transformers` autorisés **uniquement** sous `harness/` |
| Paquet `academiia.dopd` | `modules/dopd/dopd/` | **aucun** import `torch` / `transformers` ; le paquet reste pur (stdlib + NumPy + `academiia.core`) |
| Cible future-aware | `modules/dopd/dopd/target.py` | **inchangée** : le scorer rend des tableaux NumPy consommables tels quels |
| Harnais de smoke | `modules/dopd/harness/run_teacher_scorer_smoke.py` | produit le JSON versionné et le manifeste ADR-002 |

Le `TeacherScorer` implémente l'ABC `academiia.dopd.TeacherScorer` : il
hérite donc de `score(request)` et de la validation qu'il porte. Aucun
amendement du contrat J13 n'est nécessaire : l'implémentation réelle s'y
conforme **sans modification du paquet**.

#### 6.10.3 Configuration figée

| Paramètre | Valeur figée | Justification |
|---|---|---|
| Modèle professeur | `Qwen/Qwen3-0.6B` | §6.3, checkpoint de développement |
| dtype des poids | **BF16** (run GPU) ; **FP32** (run CPU) | §3.6 (App. B), dtype natif du checkpoint ; le run CPU n'est pas plafonné par la latence et valide la chaîne avec la tolérance FP32 (C3) |
| dtype d'arithmétique | **float32** | §3.6 : logits, log-probabilités, sommes, deltas, normalisation |
| `k` (candidats par distribution) | **16** | §3.4, §6.9.8 (troncature mesurée fidèle) |
| Branches maximales par job | **33** (union ≤ 32 + référence) | §3.6, App. B |
| Longueur de bloc `N` | **4** | §6.1 (défaut du papier) |
| Pas de débruitage | **4** | §6.1 (hors périmètre J17, consigné) |
| Température / top-p / top-k de génération | **1,0 / 1,0 / 0** | App. C.2, App. C.4 (paramètres de rollout du papier, consignés) |
| Seed | **20261025** | date de la mission ; aucune composante aléatoire dans le scoring lui-même |
| Plafond de cache de préfixe étendu | **16 GiB** | §3.6 : le lot de branches est réduit au-delà |

#### 6.10.4 Géométrie de scoring figée

Pour une position masquée `i` d'un bloc `x̂` de longueur `N = 4` :

- `prefix_ids = (q, z, Â_i)` — contexte prompt + partie du bloc avant `i` ;
- `suffix_ids = (x̂_{i+1}, …, x̂_N)` — partie du bloc après `i` (suffixe **visible**) ;
- `candidate_ids = V_i` — union des top-16 professeur et top-16 élève (§3.4), sans doublon ;
- `reference_id = x̂_i` — token réellement produit à la position `i`.

Le scorer rend, pour chaque job :

- `prior` : `r_i(v)` sur **tout** le vocabulaire (151 936 tokens), somme 1 ;
- `candidate_loglikelihoods` : `s_i(v) = log p_T(Ĉ_i | q, z, Â_i, v)` (eq. 14), aligné sur `candidate_ids` ;
- `reference_loglikelihood` : `s_i(x̂_i)`.

Ces trois grandeurs alimentent **directement**
`academiia.dopd.compute_future_aware_target` (eq. 18) : le smoke vérifie
que la chaîne scorer → cible NumPy est finie, somme 1 et applique la
correction quand le futur est visible.

#### 6.10.5 Critères d'acceptation du smoke J17

Le smoke est **conforme** si et seulement si toutes les conditions suivantes
sont satisfaites. Elles sont fixées **avant** le premier run.

| # | Critère | Seuil / condition | Preuve attendue |
|---|---|---|---|
| C1 | **Conformité au contrat** | `Qwen3TeacherScorer` hérite de `academiia.dopd.TeacherScorer` ; `score(ScoringRequest)` rend un `ScoringResult` valide (prior de somme 1, scores alignés) | test unitaire + sortie du smoke |
| C2 | **Exactitude cache vs rejeu** (BF16) | écart absolu maximal sur les scores de suffixe **≤ 0,35** et sur le prior **≤ 0,15** — ordre de grandeur des écarts BF16 mesurés en J14 (≤ 0,162 suffixe et ≤ 0,085 prior à la cellule 1 024/33 branches ; ≤ 0,094 et ≤ 0,036 à 16 branches) | champ `max_abs_*_diff` du JSON |
| C3 | **Exactitude cache vs rejeu** (FP32, contrôle) | écart absolu maximal sur les scores de suffixe **≤ 1e-3** et sur le prior **≤ 1e-3** ; le contrôle FP32 de J14 mesurait `3,1e-5` (suffixe) et `9e-6` (prior) | champ `fp32_control.*` du JSON |
| C4 | **Cohérence du prior** | `|somme(prior) − 1| ≤ 1e-4` ; prior sans NaN/inf ; scores candidates alignés sur `candidate_ids` | smoke |
| C5 | **Chaîne scorer → cible** | `compute_future_aware_target` renvoie `applied = true` (raison `applied`), cible finie, somme 1 à `1e-4` près, sur **tous** les jobs avec futur visible | champ `target.*` du JSON |
| C6 | **Latence (cellule de référence J14)** | à géométrie **identique** à J14 (préfixe 1 024 tokens, suffixe 3, **16 branches au total**, cache de préfixe), la latence médiane du job complet (prior + scores) est **≤ 150 ms** et le chemin cache est **au moins 2× plus rapide** que le chemin de rejeu intégral ; valeurs de référence J14 : 51,2 ms (cache) vs 167,9 ms (rejeu), gain 3,28× | champ `latency.reference_geometry` |
| C7 | **Latence (géométrie d-OPD)** | **sur GPU**, sur les états réels (prompts J15, blocs `N = 4`, 17 branches = 16 candidats + référence), latence médiane **≤ 200 ms** par job ; le run CPU consigne la même mesure **sans plafond** (critère non applicable au CPU) | champ `latency.real_states` |
| C8 | **Latence (charge à 33 branches)** | à 33 branches (union ≤ 32 + référence) et préfixe 1 024, latence médiane **≤ 250 ms** ; référence J14 : 65,6 ms | champ `latency.reference_geometry` (entrée 33 branches) |
| C9 | **Gelé** | `requires_grad` de **tous** les paramètres du professeur = `False` ; `model.training = False` ; **aucun** appel `backward`, `optimizer`, `step` dans le harnais ; empreinte sha256 des poids identique avant/après le run | champ `training.*` du JSON + inspection statique du harnais |
| C10 | **Poids intacts** | sha256 des tenseurs du professeur **identique** avant et après le run | champ `weights.sha256_before/after` |
| C11 | **Provenance ADR-002** | manifeste `academiia.lineage/1` présent, `producer.module = "dopd"`, `producer.commit` = commit du code producteur, commande exacte, versions, seeds, empreintes des sources, `training.performed = false` | bloc `lineage` du JSON |
| C12 | **Règle 9 (cache)** | le snapshot lu vit **dans le dépôt** (`model_source_in_repo = true`) ; sinon avertissement émis et `model_source_in_repo = false` consigné | bloc `cache` du JSON |
| C13 | **Collecte sans torch** | `pytest` collecte et passe sans torch installé (les tests du harnais sont ignorés via `pytest.importorskip`, pas en erreur) | preuve CI locale (venv sans torch) |
| C14 | **Non-régression** | `target.py`, `diagnostic.py` (correctifs J16) et `modules/a2d/eval/dream_a2d/` **inchangés** ; verdict G1 intact | diff de la plage de commits |

#### 6.10.6 Plan de mesures figé

1. **Cellule de référence J14** (GPU) : préfixe de **1 024** tokens, suffixe de **3**
   tokens, **16** branches au total (15 candidats + 1 référence) et **33** branches
   au total (32 candidats + 1 référence) — comparaison directe aux 51,2 / 65,6 ms
   de J14. Ce n'est pas un résultat scientifique : c'est une mesure de débit.
   Identifiants de token tirés aléatoirement avec la seed figée (§6.10.3), comme
   dans `bench_budget_rtx5090.py`.
2. **États réels** (CPU puis GPU) : prompts du jeu versionné de J15
   (`modules/dopd/eval/annexe_d_qwen3_0.6b_multiprompt/prompts.json`), blocs de
   `N = 4` tokens générés **greedily** par le professeur lui-même (température 1,0,
   top-p 1,0, top-k 0 : aucune composante aléatoire), puis scoring de chaque
   position `i` du bloc. Les positions `i = 0, 1, 2` ont un **futur visible**
   (suffixes de 3, 2 et 1 token) ; la position `i = 3` n'en a pas (suffixe vide)
   et exerce le fallback causal (`no_visible_future`). Effectif minimal :
   **4 prompts × 2 positions avec futur visible = 8 jobs réels** ; le plan complet
   score les 4 positions du bloc, soit 16 jobs par jeu de 4 prompts, avec un
   vocabulaire de 151 936 tokens.
   `V_i` est le **top-16 du professeur** : l'union avec le top-16 de l'élève
   (§3.4) n'est pas calculable en J17, puisque aucun `StudentPolicy` n'existe
   (§6.10.1). Le contrat du scorer est donc exercé avec `|V_i| ≤ 16`, et le
   contrat de branches à 33 est exercé par la cellule de référence (point 1).
3. **Contrôle FP32** : le même job rejoué avec un modèle FP32, sur la cellule de
   référence (1 024 / 3 / 16 branches), pour distinguer l'écart de précision BF16
   d'un défaut de logique. Le contrôle est **exécuté une fois** (GPU) : il sert
   de borne haute d'exactitude, pas de mesure répétée.
4. **Exactitude** : pour chaque job, le chemin cache est comparé au chemin de
   **rejeu intégral** (contexte complet rejoué par branche, sans cache).

#### 6.10.7 Décision pour J18

La décision pour J18 (`StudentPolicy` ou mini-boucle d'entraînement) est
**GO conditionnel** si C1–C14 sont tenues, et **NO-GO** si l'une des
conditions C1, C3, C5, C9, C10 ou C11 échoue. Les réserves de lecture J15/J16
sont rappelées dans le rapport : effet **non uniforme** (médiane nulle, queue
nuisible de 13,3 %), et le passage à un `StudentPolicy` exigerait un protocole
d'entraînement pré-enregistré **distinct** de J17.

---

### 6.11 Livraison J17 — `TeacherScorer` réel (état après exécution)

> Cette section est **postérieure** aux runs : elle **publie** l'état livré.
> Elle ne modifie pas les critères de §6.10, qui restent figés (§6.10.1).
> Les chiffres viennent de `modules/dopd/eval/teacher_scorer_qwen3_0.6b/results.json`
> (GPU, BF16) et `modules/dopd/eval/teacher_scorer_qwen3_0.6b_cpu/results.json`
> (CPU, FP32).

#### 6.11.1 Ce qui est livré

| Élément | Emplacement | Garantie |
|---|---|---|
| `Qwen3TeacherScorer` | `modules/dopd/harness/teacher_scorer_qwen3.py` | implémente l'ABC J13 `TeacherScorer` ; professeur gelé (`eval()`, `requires_grad=False`) ; cache de préfixe (App. B) ; arithmétique float32 ; plafond de cache étendu 16 GiB (§3.6) |
| Harnais de smoke | `modules/dopd/harness/run_teacher_scorer_smoke.py` | produit le JSON versionné + manifeste ADR-002 ; chemins de contrôle cache et rejeu |
| Tests | `modules/dopd/tests/test_dopd_harness_teacher_scorer.py` | `pytest.importorskip("torch")` : ignorés sans torch, **pas** en erreur |

Le paquet `academiia.dopd` est **inchangé** : `target.py`, `diagnostic.py` et
`interfaces.py` ne sont pas modifiés par J17. Le scorer rend des tableaux
NumPy (`prior` float64 de somme 1, `candidate_loglikelihoods` alignés sur
`candidate_ids`, `reference_loglikelihood`) **directement** consommables par
`compute_future_aware_target`.

#### 6.11.2 Résultat des critères pré-enregistrés (§6.10.5)

| # | Critère | Verdict | Preuve mesurée |
|---|---|---|---|
| C1 | Conformité au contrat | **TENU** | `isinstance(scorer, TeacherScorer)` ; `score()` rend un `ScoringResult` valide (test unitaire + smoke) |
| C2 | Exactitude cache vs rejeu (BF16) | **NON TENU sur les états réels** | suffixe max **0,395069** > 0,35 (2 états sur 16 : `p05`/`p18` position 0) ; prior max **0,04824** ≤ 0,15 ✔. Tenue sur la cellule de référence : suffixe 0,116714 (16 branches) et 0,142105 (33 branches) ≤ 0,35 ✔ |
| C3 | Exactitude cache vs rejeu (FP32) | **TENU** | cellule de référence **1,1444e-05** ≤ 1e-3 ✔ ; **état réel BF16 rejoué en FP32 : 6,1035e-05** ≤ 1e-3 ✔ ; run CPU FP32 complet : suffixe max **2,7e-05**, prior max **2e-06** ✔ |
| C4 | Cohérence du prior | **TENU** | `|somme − 1|` max **4,8e-07** ≤ 1e-4 ; aucun NaN/inf |
| C5 | Chaîne scorer → cible | **TENU** | **12/12** jobs à futur visible : `applied = true`, cible finie, `|somme − 1|` max **1,5e-07** ; les 4 jobs à futur vide activent le fallback causal (`no_visible_future`) |
| C6 | Latence (cellule de référence J14) | **TENU** | 16 branches : cache **53,2 ms**, rejeu 168,4 ms, gain **3,162×** ≥ 2× (J14 : 51,2 ms / 3,28×) |
| C7 | Latence (géométrie d-OPD) | **TENU (réserve de géométrie)** | cache **médiane 51,63 ms** (max 53,29 ms) ≤ 200 ms ; CPU FP32 (hors plafond) : médiane 326,41 ms, max 373,61 ms. Les états réels ont **16 branches** (référence déjà dans le top-16), §6.10.5 fige 17 : seuil tenu avec une géométrie légèrement **inférieure** au protocole |
| C8 | Latence (33 branches) | **TENU** | cache **67,1 ms** ≤ 250 ms (J14 : 65,6 ms) |
| C9 | Gelé | **TENU** | `requires_grad` de tous les paramètres = `False` ; `model.training = False` ; aucun `backward`/`optimizer`/`step` dans le harnais |
| C10 | Poids intacts | **TENU** | sha256 des tenseurs **identique** avant/après (runs GPU et CPU) |
| C11 | Provenance ADR-002 | **TENU** | manifeste `academiia.lineage/1` : `producer.module = "dopd"`, commit, commande exacte, versions, seeds, empreintes des sources, `training.performed = false` |
| C12 | Règle 9 (cache) | **TENU** | `model_source_in_repo = true` (snapshot sous `_artefact/hf-cache/`) ; `HF_HOME` redirigé dans le dépôt |
| C13 | Collecte sans torch | **TENU** | venv sans torch : **114 passed, 3 skipped**, 0 erreur |
| C14 | Non-régression | **TENU** | `target.py`, `diagnostic.py`, `interfaces.py` et `modules/a2d/eval/dream_a2d/` **inchangés** sur la plage J17 |

#### 6.11.3 Écart C2 — diagnostic (exécuté **après** le run, hors protocole)

Le seuil C2 (≤ 0,35) est **échoué** sur les états réels BF16 (max 0,395069) et
**tenu** sur la cellule de référence. Le seuil n'est pas retouché. Deux
diagnostics supplémentaires (champs `max_abs_delta_diff`,
`replay_batch_sensitivity`, `suffix_score_abs_max`) quantifient la cause :

1. **Le chemin cache est exact au sens du seuil C3.** Le même job, rejoué en
   FP32, tombe à **6,1035e-05** (état réel, ×16 sous le seuil) et
   **1,1444e-05** (cellule de référence, ×87 sous le seuil). Le run CPU FP32
   complet confirme : suffixe max **2,7e-05** (×37 sous le seuil) sur les 16
   états réels, et `replay_batch_sensitivity = 0,0` (rejeu par lot de 1 et par
   lot de 16 **identiques** en FP32). Les écarts ne sont pas nuls : le cache
   est exact **dans la tolérance pré-enregistrée**, pas au sens bit-à-bit.
2. **L'écart BF16 vient du chemin de rejeu lui-même, pas du cache.** En BF16,
   `replay_batch_sensitivity` atteint **0,369659** : rejouer le même contexte
   par lot de 1 ou par lot de 16 donne déjà des scores différents de ~0,37.
   Ce maximum est atteint **au même job** que le maximum d'écart cache/rejeu
   (`p05`, position 0) : sur ce job, l'écart cache/rejeu n'excède la
   sensibilité du rejeu à lui-même que de **0,025410**. Cette comparaison
   n'est pas une décomposition exacte (elle oppose deux maxima), mais elle situe
   l'ordre de grandeur : l'écart est **de la taille de la sensibilité du rejeu**.
   En FP32, la sensibilité tombe à **0** (run CPU, 16 états) ou **1,5259e-05**
   (contrôle GPU).
3. **Amplification par la magnitude.** L'écart absolu est proportionnel à
   `|s_i(v)|` : les deux états en échec sont ceux dont les log-probabilités de
   suffixe sont les plus négatives (**|s|max = 20,67** et **27,41**).
   **Interprétation** (non dérivée d'un artefact, mais cohérente avec les
   mesures) : BF16 a 8 bits de significande ; une erreur relative de ~0,4 %
   sur un logit de magnitude 20 donne ~0,08 d'écart absolu par token, cumulé
   sur 3 tokens de suffixe.

**Conclusion.** L'écart C2 est un **artefact de précision BF16 du protocole de
comparaison**, pas un défaut de la mécanique du cache de préfixe. La grandeur
réellement consommée par la cible est `Δs_i(v) = s_i(v) − s_i(x̂_i)` ; son
écart cache/rejeu en BF16 est du même ordre (max **0,395436**), et tombe à
**3,2e-05** en FP32 (run CPU). Le passage en FP32 coûte **+0,68 ms** de latence
médiane (**+1,3 %**) et **+1 161 MiB** (**+1,13 GiB**) de pic mémoire sur cette
machine : le surcoût en latence est négligeable, celui en mémoire est réel.
*Mesure complémentaire non versionnée* (sonde jetable, hors artefact ADR-002 ;
voir `modules/dopd/docs/reports/J17.md` §4.5).

#### 6.11.4 Limites de la livraison

- Le cache de préfixe ne paie qu'au-delà d'un préfixe long : sur les états
  réels (préfixes de **12 à 21** tokens), le cache est **plus lent** que le
  rejeu (médiane **51,63 ms** contre **26,45 ms** ; max **53,29 ms** contre
  **27,01 ms**), car le surcoût fixe par forward (~24,85 ms en moyenne)
  domine. Le gain mesuré (3,16× à 16 branches, 5,48× à 33 branches, préfixe
  1 024) est le gain de la **cellule de référence J14**, pas celui de la
  géométrie courte. Le point de basculement du cache reste celui mesuré par
  J14 (plus lent à 256 tokens, 2,00× plus rapide à 512) : J17 ne mesure que
  12 à 21 tokens et **corrobore le sens** de cette règle sans la redoiter.
- `V_i` est le **top-16 du professeur** : l'union avec le top-16 de l'élève
  (§3.4) n'est pas calculable tant que `StudentPolicy` n'existe pas. Le
  contrat à 33 branches est donc exercé par la cellule de référence.
- Aucun rollout réel, aucun `StudentPolicy`, aucun entraînement, aucun
  backward, aucun poids modifié, aucun checkpoint écrit, aucun résultat,
  seuil ou verdict G1 touché.
- Les réserves de lecture J15/J16 s'appliquent inchangées : effet future-aware
  **non uniforme** (médiane nulle, queue nuisible de 13,3 %), 3 prompts sur 20
  dégradés.

#### 6.11.5 Décision pour J18

**GO conditionnel**, avec une réserve de règle explicite : §6.10.7 conditionne
le « GO conditionnel » à « C1–C14 **sont tenues** ». C2 échouant, cette clause
n'est pas littéralement remplie. La décision repose sur le fait qu'**un seul**
critère échoue, **hors** de la liste NO-GO (C1, C3, C5, C9, C10, C11 — aucun
déclenché), avec une cause identifiée (précision BF16 du protocole de
comparaison) et un contournement mesuré (FP32 : écart 6,1e-05 sur état réel,
2,7e-05 sur le run CPU complet, +0,68 ms de latence).

Conditions à lever avant d'ouvrir J18 : (1) professeur en **FP32** ; (2)
**protocole d'entraînement pré-enregistré distinct** ; (3) référence de
comparaison fixée sur le **rejeu intégral FP32** ; (4) prise en compte du fait
que le cache est **plus lent que le rejeu** aux préfixes de 12 à 21 tokens.
Détail et points ouverts : `modules/dopd/docs/reports/J17.md`.

---
---

## 7. Hors scope explicite de J12

- Entraînement d-OPD, rollout, distillation, optimization.
- Téléchargement ou utilisation de checkpoints Qwen3 (même 0,6B) ; aucun modèle lourd
  touché.
- Reproduction des benchmarks, des tableaux ou des chiffres du papier.
- Création de `modules/opdlm/` (nom interdit) ou d'un second module concurrent.
- Modification des résultats, seuils, protocoles ou verdicts A2D (G1/G2/G1b/G3).
- Ajout de dépendances lourdes (torch, transformers, deepspeed) au paquet `academiia.dopd`.
  Le squelette n'utilise que NumPy.
- Entry-points CLI d-OPD : aucun à ce stade (pas nécessaires au smoke d'import).
- Import paresseux de torch / modification de M3 ou d'ADR-004 : non traités ici.

---

## 8. Références

1. R. Liu, Q. Hu, S. Han. *d-OPD: Future-Aware On-Policy Distillation for Block Diffusion
   Language Models.* arXiv:2609.35362v1, 2026. <https://arxiv.org/abs/2609.35362>
2. X. Su et al. *Data-efficient autoregressive-to-diffusion language models via on-policy
   distillation.* arXiv:2606.06712, 2026 (OPDLM, baseline principal).
3. ADR-001 — disposition et règles de dépendance : `docs/adr/ADR-001-disposition.md`.
4. ADR-002 — manifeste de provenance : `docs/adr/ADR-002-provenance.md`.
5. ADR-003 — règle de deux : `docs/adr/ADR-003-regle-de-deux.md`.
6. ADR-004 — import de torch paresseux (M3, reporté) :
   `docs/adr/ADR-004-import-torch-paresseux.md`.
