# Clôture formelle de J17

**Date :** 2026-10-26  
**Commit d'entrée :** `c1d6eaf` (HEAD, main)  
**Statut :** validée techniquement par la revue N2 d'architecte.

---

## 1. Validation technique de J17

Le `TeacherScorer` réel Qwen3-0.6B est livré :
- Implémentation du contrat ABC `academiia.dopd.TeacherScorer` (J13) ;
- Cache de préfixe conforme à App. B du papier (§3.6) ;
- Arithmétique float32, plafond de cache étendu 16 GiB ;
- Professeur gelé (`requires_grad=False`, `model.training=False`) ;
- Manifeste ADR-002 complet (`producer.module = "dopd"`, commit, commande exacte, empreintes des sources) ;
- Aucun entraînement, aucun rollout réel, aucun `StudentPolicy`, aucun backward, aucun poids modifié.

**13/14 critères pré-enregistrés tenus.** Seul **C2** échoue sur les états réels BF16 (max 0,395069 > seuil 0,35), sans atteindre les conditions de NO-GO (C1, C3, C5, C9, C10, C11 — aucun déclenché).

**Revue N2 :** verdict « Incorrect » (3 majeurs, 4 mineurs) ; **les 7 constats ont été vérifiés puis corrigés** :
- Contrainte `transformers>=4.56.2` avec garde-fou explicite ;
- Empreintes **de fichier** du checkpoint dans `parent.hashes` (ADR-002) ;
- Tests rendus sensibles au préfixe/aux positions/masque avec contrôle sur un vrai Qwen3 ;
- Échec dur hors dépôt ;
- Révision du snapshot et positions effectives consignées.

**Sensibilité prouvée par injection de fautes :** 6, 4 et 3 tests échouent selon la faute (masque, positions, préfixe). Seconde revue : 6/7 constats corrigés ; écart résiduel F3 (fake enregistrait le masque sans l'utiliser) corrigé — les tokens masqués sont exclus de l'agrégat du fake, propriété **verrouillée par un test** (écart des scores de suffixe, masque plein vs nul : 0,0 → 0,377).

---

## 2. Décision R1 (C2 / §6.10.7)

> Avec le professeur en **FP32** comme précision de production, et le rejeu intégral FP32 comme référence d'exactitude, **on autorise la poursuite**.
> La règle §6.10.7 est amendée pour l'avenir : tout critère hors liste NO-GO qui échoue impose une **décision explicite écrite** (ADR ou STATUS) ; le silence n'est pas un GO.
> Cette décision est **propre à J17** et ne préjuge pas des critères de J18.

C2 a échoué (0,395069 > 0,35). La cause est un artefact de précision BF16 du protocole de comparaison (prouvé par les contrôles FP32 : écart ≤ 6,2e-05 en état réel GPU, ≤ 2,7e-05 en run CPU complet). La décision est une **interprétation**, pas une application littérale de §6.10.7.

**Consignée dans :**
- `docs/STATUS.md` (bandeau dopd, ligne 12) ;
- `modules/a2d/STATUS.md` (section J17, paragraphe R1) ;
- `modules/dopd/docs/DOPD_SPEC.md` (§6.10.8).

---

## 3. Pré-enregistrement J18

Le pré-enregistrement complet est versionné dans :
- **`modules/dopd/docs/reports/J18_PRE.md`** (document unique, figé avant tout code producteur).

Contenu : objectif (expérience de décision à budget borné), précision FP32, référence rejeu intégral FP32, géométrie préfixée 12–21 tokens, critères C1–C10 avec seuils numériques, budget strict (≤ 4 h GPU, ≤ 16 GiB), hors périmètre explicite, condition préalable recommandée (Annexe D sur Qwen3-1.7B/4B).

---

## 4. Nettoyage process (R2)

Chiffres opérationnels redondants supprimés de `docs/STATUS.md` ligne 12 : latences médianes (52,75 ms, 53,905 ms, 68,187 ms), compte CI (242 tests). Source de vérité = artefacts versionnés + rapport de mission. Les STATUS ne contiennent plus de chiffres qui forcent des commits de synchronisation.

---

## 5. Vérifications bloquantes

| Vérification | Résultat |
|---|---|
| `git log --oneline 1966111..HEAD \| head` | Ancestry confirmé : `c1d6eaf` ancêtre de `1966111` (J15) |
| `pytest modules/dopd/tests -q` | **160 passed**, 0 erreur |
| `ruff check` sur fichiers touchés | Aucun fichier Python modifié ; markdown non concerné par ruff |
| Fichiers sous `modules/dopd/dopd/` ou `harness/` modifiés | **Aucun** (git diff : uniquement STATUS.md, DOPD_SPEC.md) |
| Artefacts d'évaluation régénérés | **Aucun** (aucun JSON versionné touché) |

---

## 6. Absence de code producteur et de push

- **Aucun code J18 n'a été écrit.** Le seul nouveau fichier est `J18_PRE.md` (documentaire).
- **Aucun push vers `origin/main`.** Les commits restent locaux (règle 11 d'AGENTS.md).
- La réserve de gouvernance : push non autorisé sur `origin/main` (de `08d62bd` à `50464eb`) ; attribution indéterminée, contenu sans secret.

---

## 7. Prochaines étapes

1. **Revue N2 du pré-enregistrement J18** (`modules/dopd/docs/reports/J18_PRE.md`).
2. Aucun code J18 ne sera autorisé tant que ce pré-enregistrement n'aura pas été revu et accepté.
3. Après approbation : exécution de l'expérience de décision à budget borné (≤ 4 h GPU, ≤ 50 prompts).

---

**Ce rapport ne liste pas ses propres commits de consignation** (règle 12 d'AGENTS.md). Il cite la plage `c1d6eaf..HEAD` et les preuves : 160 tests passés, git diff (3 fichiers modifiés + 1 nouveau), absence de code producteur.
