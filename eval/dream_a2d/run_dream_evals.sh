#!/usr/bin/env bash
# Baseline Dream-7B (A2D, arXiv:2610.08108 App. A.1 / Table 16) — J6 mission C0.
#
# Harnais : fork lm-evaluation-harness de Dream (DreamLM/Dream, eval_instruct/lm_eval),
#           modele "diffllm" (decodage par diffusion, entropy unmasking).
# Parametres : ceux publies par le fork (eval_instruct/eval.sh) :
#           temperature 0.1, top_p 0.9, alg=entropy, diffusion_steps = max_new_tokens,
#           un pas de debruitage par token genere, bfloat16, chat template applique.
# Taches et budgets (A2D App. A.1) :
#           gsm8k_cot_a2d : 0-shot, 256 tokens/steps, system prompt A.1 (#### <nombre>)
#           minerva_math   : 0-shot, 512 tokens/steps (MATH-500 = sous-ensemble du test MATH)
#           ifeval         : 0-shot, 1280 tokens/steps
# Sous-ensembles : --limit N (N premiers exemples dans l'ordre du dataset, cf. A2D App. A.3
#           "Math sweeps use first 200 examples in dataset order per benchmark").
#           Le budget GPU local ne permet pas les jeux complets (mesure : 162 s/prompt
#           pour IFEval a 1280 pas de debruitage) : les effectifs et l'intervalle de
#           confiance sont consignes dans docs/reports/J6.md.
set -u

VENV=/home/sjteb/dream-harness/venv
FORK=/home/sjteb/dream-harness/Dream/eval_instruct
TASKS_A2D=/home/sjteb/dream-harness/tasks_a2d
ROOT=/mnt/d/Intelligence_Artificielle/Projects/Academiia
CKPT=$ROOT/checkpoints
OUT=$ROOT/eval/dream_a2d
LOG=/home/sjteb/dream-harness/logs
mkdir -p "$OUT" "$LOG"

GSM8K_SYSTEM="Solve the math problem with brief, direct reasoning. Do NOT restate the problem, do NOT use bullet points, and use as few steps as possible. On the final line, write the answer as '#### <number>' and nothing else."

BATCH=${BATCH:-4}
LIMIT_GSM8K=${LIMIT_GSM8K:-300}
LIMIT_MATH=${LIMIT_MATH:-25}
LIMIT_IFEVAL=${LIMIT_IFEVAL:-30}
MATH_TASKS="minerva_math_algebra,minerva_math_counting_and_prob,minerva_math_geometry,minerva_math_intermediate_algebra,minerva_math_num_theory,minerva_math_prealgebra,minerva_math_precalc"

run () {   # run <model_dir> <task> <max_new_tokens> <limit> <extra args...>
  local model=$1 task=$2 maxtok=$3 limit=$4; shift 4
  local label; label=$(echo "$task" | cut -d, -f1)
  local outdir="$OUT/$(basename "$model")_$label"
  echo "=== $(date +%H:%M:%S) $model | $task | max_new_tokens=$maxtok limit=$limit batch=$BATCH"
  cd "$FORK"
  PYTHONPATH=. "$VENV/bin/python" -m lm_eval \
      --model diffllm \
      --model_args "pretrained=$CKPT/$model,trust_remote_code=True,max_new_tokens=$maxtok,diffusion_steps=$maxtok,dtype=bfloat16,temperature=0.1,top_p=0.9,alg=entropy" \
      --tasks "$task" \
      --include_path "$TASKS_A2D" \
      --num_fewshot 0 \
      --batch_size "$BATCH" \
      --limit "$limit" \
      --output_path "$outdir" \
      --log_samples --confirm_run_unsafe_code \
      --apply_chat_template "$@" 2>&1 | tee "$LOG/$(basename "$model")_$label.log" | tail -25
  echo "=== $(date +%H:%M:%S) fin $model $task (rc=${PIPESTATUS[0]})"
}

for model in dream_7b_base dream_7b_instruct; do
  run "$model" gsm8k_cot_a2d 256 "$LIMIT_GSM8K" --system_instruction "$GSM8K_SYSTEM"
  run "$model" "$MATH_TASKS" 512 "$LIMIT_MATH"
  run "$model" ifeval_a2d    1280 "$LIMIT_IFEVAL"
done
echo "ALL DONE $(date +%H:%M:%S)"
