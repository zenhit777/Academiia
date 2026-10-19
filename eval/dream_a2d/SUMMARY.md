# Resume des evaluations Dream-7B (J6, porte G1)

Harnais : fork lm-eval de Dream (`diffllm`), T=0.1, top_p=0.9, alg=entropy,
diffusion_steps = max_new_tokens, bfloat16, chat template, 0-shot (GSM8K/MATH).

| modele | benchmark | filtre | score | n | IC 95 % (Wilson) | cible Table 16 | ecart |
|---|---|---|---|---|---|---|---|
| dream_7b_base | GSM8K | flexible-extract | 68.7 % | 300 | [63.2, 73.7] | 66.5 | +2.2 |
| dream_7b_base | GSM8K | strict-match | 42.7 % | 300 | [37.2, 48.3] | 66.5 | -23.8 |

Effectifs reduits : le budget GPU local ne permet pas les jeux complets (mesure : 162 s/prompt pour IFEval a 1280 pas de debruitage). L'intervalle de Wilson est donne pour chaque score.
