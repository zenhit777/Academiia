# Resume des evaluations Dream-7B (J6, porte G1)

Harnais : fork lm-eval de Dream (`diffllm`), T=0.1, top_p=0.9, alg=entropy,
diffusion_steps = max_new_tokens, bfloat16, chat template, 0-shot (GSM8K/MATH).

**MATH-175** = sous-ensemble `minerva_math` (175 problemes sur 5000). Le jeu du
papier est **MATH-500** (500 problemes) : la cible Table 16 ci-dessous est donc
comparee a un PERIMETRE DIFFERENT — ce n'est pas une reproduction. IFEval (n=8)
est un controle de fonctionnement, pas une mesure.

| modele | benchmark | filtre | score | n | IC 95 % (Wilson) | cible Table 16 | ecart |
|---|---|---|---|---|---|---|---|
| dream_7b_base | GSM8K | exact_match,flexible-extract | 68.7 % | 300 | [63.2, 73.7] | 66.5 | +2.2 |
| dream_7b_base | GSM8K | exact_match,strict-match | 42.7 % | 300 | [37.2, 48.3] | 66.5 | -23.8 |
| dream_7b_base | IFEval | inst_level_loose_acc,none | 64.3 % | 8 | [30.6, 86.3] | 49.7 | +14.6 |
| dream_7b_base | IFEval | inst_level_strict_acc,none | 64.3 % | 8 | [30.6, 86.3] | 49.7 | +14.6 |
| dream_7b_base | IFEval | prompt_level_loose_acc,none | 50.0 % | 8 | [21.5, 78.5] | 49.7 | +0.3 |
| dream_7b_base | IFEval | prompt_level_strict_acc,none | 50.0 % | 8 | [21.5, 78.5] | 49.7 | +0.3 |
| dream_7b_base | MATH-175 | exact_match,none | 0.0 % | 175 | [0.0, 2.1] | 20.8 | -20.8 |
| dream_7b_base | MATH-175 | math_verify,none | 24.0 % | 175 | [18.3, 30.8] | 20.8 | +3.2 |
| dream_7b_instruct | GSM8K | exact_match,flexible-extract | 84.0 % | 300 | [79.4, 87.7] | 83.0 | +1.0 |
| dream_7b_instruct | GSM8K | exact_match,strict-match | 66.0 % | 300 | [60.5, 71.1] | 83.0 | -17.0 |
| dream_7b_instruct | IFEval | inst_level_loose_acc,none | 71.4 % | 8 | [40.9, 92.9] | 63.0 | +8.4 |
| dream_7b_instruct | IFEval | inst_level_strict_acc,none | 71.4 % | 8 | [40.9, 92.9] | 63.0 | +8.4 |
| dream_7b_instruct | IFEval | prompt_level_loose_acc,none | 50.0 % | 8 | [21.5, 78.5] | 63.0 | -13.0 |
| dream_7b_instruct | IFEval | prompt_level_strict_acc,none | 50.0 % | 8 | [21.5, 78.5] | 63.0 | -13.0 |
| dream_7b_instruct | MATH-175 | exact_match,none | 0.0 % | 175 | [0.0, 2.1] | 38.4 | -38.4 |
| dream_7b_instruct | MATH-175 | math_verify,none | 36.6 % | 175 | [29.8, 43.9] | 38.4 | -1.8 |
| dream_7b_base | IFEval | moyenne des 4 metriques | 57.1 % | 8 | [30.6, 86.3] | 49.7 | +7.4 |
| dream_7b_instruct | IFEval | moyenne des 4 metriques | 60.7 % | 8 | [30.6, 86.3] | 63.0 | -2.3 |

Effectifs reduits : le budget GPU local ne permet pas les jeux complets (mesure : 162 s/prompt pour IFEval a 1280 pas de debruitage). L'intervalle de Wilson est donne pour chaque score. MATH-175 != MATH-500 : la cible Table 16 porte sur 500 problemes, la mesure porte sur 175 — comparaison a PERIMETRE DIFFERENT, ce n'est pas une reproduction. IFEval (n=8) est un controle de fonctionnement, pas une mesure.
