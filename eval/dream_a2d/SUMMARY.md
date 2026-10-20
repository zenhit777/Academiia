# Resume des evaluations Dream-7B (J6, porte G1)

Harnais : fork lm-eval de Dream (`diffllm`), T=0.1, top_p=0.9, alg=entropy,
diffusion_steps = max_new_tokens, bfloat16, chat template, 0-shot (GSM8K/MATH).

| modele | benchmark | filtre | score | n | IC 95 % (Wilson) | cible Table 16 | ecart |
|---|---|---|---|---|---|---|---|
| dream_7b_base | GSM8K | exact_match,flexible-extract | 68.7 % | 300 | [63.2, 73.7] | 66.5 | +2.2 |
| dream_7b_base | GSM8K | exact_match,strict-match | 42.7 % | 300 | [37.2, 48.3] | 66.5 | -23.8 |
| dream_7b_base | MATH-500 | exact_match,none | 0.0 % | 175 | [0.0, 2.1] | 20.8 | -20.8 |
| dream_7b_base | MATH-500 | math_verify,none | 24.0 % | 175 | [18.3, 30.8] | 20.8 | +3.2 |
| dream_7b_instruct | GSM8K | exact_match,flexible-extract | 84.0 % | 300 | [79.4, 87.7] | 83.0 | +1.0 |
| dream_7b_instruct | GSM8K | exact_match,strict-match | 66.0 % | 300 | [60.5, 71.1] | 83.0 | -17.0 |
| dream_7b_instruct | MATH-500 | exact_match,none | 0.0 % | 175 | [0.0, 2.1] | 38.4 | -38.4 |
| dream_7b_instruct | MATH-500 | math_verify,none | 36.6 % | 175 | [29.8, 43.9] | 38.4 | -1.8 |

Effectifs reduits : le budget GPU local ne permet pas les jeux complets (mesure : 162 s/prompt pour IFEval a 1280 pas de debruitage). L'intervalle de Wilson est donne pour chaque score.
