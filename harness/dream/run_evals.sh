#!/usr/bin/env bash
# run_evals.sh — evaluation Dream-7B, harnais dans le dépôt (mission J6bis C1/C3).
#
# Remplace les scripts `run_dream_evals.sh` (V1) et `run_dream_evals2.sh` (V2)
# qui vivaient HORS du depot (/home/sjteb/dream-harness) : meme commande
# lm_eval, meme entrelacement, mais tout est parametrable et tous les chemins
# sont RELATIFS au depot.
#
#   bash harness/dream/run_evals.sh                      # base + instruct, 3 taches
#   MODEL_DIR=checkpoints/dream_7b_base TASK=gsm8k LIMIT=8 \
#     bash harness/dream/run_evals.sh                    # un seul modele, une tache
#
# Variables d'environnement :
#   MODEL_DIR  liste de checkpoints (relatifs a la racine du depot ou absolus),
#              separes par des virgules. Defaut : checkpoints/dream_7b_base,
#              checkpoints/dream_7b_instruct. L'alias de sortie est le nom du
#              dossier (ex. dream_7b_base).
#   TASK       liste de taches : gsm8k | math | ifeval (ou une tache lm_eval
#              complete). Defaut : gsm8k,math,ifeval.
#   LIMIT      limite par tache : "gsm8k=300,math=25,ifeval=8" ou un nombre
#              unique applique a toutes. Defaut : 300 / 25 / 8.
#   SEED      graines lm_eval "random,numpy,torch,fewshot". Defaut : la
#              valeur du fork (0,1234,1234,1234) — celle utilisee par J6.
#   OUTDIR    dossier de resultats (relatif a la racine du depot). Defaut :
#              eval/dream_a2d. Structure de sortie : <OUTDIR>/<alias>/<tache>/
#   BATCH     taille de batch lm_eval. Defaut : 4 (mesure J6).
#   SKIP_EXISTING=1 (defaut) : ne rejoue pas une tache dont le
#              `results_*.json` existe deja dans <OUTDIR>/<alias>/<tache>/.
#   FORCE=1   : rejoue meme si le resultat existe.
#
# Protocole (A2D App. A.1) : temperature 0.1, top_p 0.9, alg=entropy,
# diffusion_steps = max_new_tokens (un pas de debruitage par token), bfloat16,
# chat template applique, 0-shot, budgets 256 (GSM8K) / 512 (MATH) / 1280
# (IFEval).
set -u

HARNESS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HARNESS_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

VENV="${HARNESS_DIR}/.venv"
FORK="${HARNESS_DIR}/.fork"
TASKS_A2D="${HARNESS_DIR}/tasks"
CACHE="${HARNESS_DIR}/.cache"
STAGE="${CACHE}/stage"
LOG_DIR="${CACHE}/logs"

export HF_HOME="${CACHE}/huggingface"
export HF_DATASETS_CACHE="${CACHE}/huggingface/datasets"
export HUGGINGFACE_HUB_CACHE="${CACHE}/huggingface/hub"
export XDG_CACHE_HOME="${CACHE}/xdg"
export TMPDIR="${CACHE}/tmp"
mkdir -p "${STAGE}" "${LOG_DIR}" "${HF_HOME}" "${TMPDIR}"

MODEL_DIR="${MODEL_DIR:-checkpoints/dream_7b_base,checkpoints/dream_7b_instruct}"
TASK="${TASK:-gsm8k,math,ifeval}"
SEED="${SEED:-0,1234,1234,1234}"
OUTDIR="${OUTDIR:-eval/dream_a2d}"
BATCH="${BATCH:-4}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
FORCE="${FORCE:-0}"
LIMIT="${LIMIT:-}"

if [ ! -x "${VENV}/bin/python" ]; then
  echo "ERREUR: environnement absent (${VENV}). Lancer : bash harness/dream/setup.sh" >&2
  exit 2
fi
if [ ! -d "${FORK}/eval_instruct" ]; then
  echo "ERREUR: fork du harnais absent (${FORK}). Lancer : bash harness/dream/setup.sh" >&2
  exit 2
fi

# --- limites par tache -----------------------------------------------------
limit_for () {  # limit_for <tache>
  local task=$1
  if [ -n "${LIMIT}" ]; then
    local entry
    for entry in ${LIMIT//,/ }; do
      case "${entry}" in
        "${task}"=*) echo "${entry#*=}"; return 0 ;;
      esac
    done
    case "${LIMIT}" in
      *[!0-9=]*) : ;;                       # contient des paires cle=valeur
      *) echo "${LIMIT}"; return 0 ;;       # nombre unique
    esac
  fi
  case "$1" in
    gsm8k)  echo 300 ;;
    math)   echo 25 ;;
    ifeval) echo 8 ;;
    *)      echo 25 ;;
  esac
}

GSM8K_SYSTEM="Solve the math problem with brief, direct reasoning. Do NOT restate the problem, do NOT use bullet points, and use as few steps as possible. On the final line, write the answer as '#### <number>' and nothing else."
MATH_TASKS="minerva_math_algebra,minerva_math_counting_and_prob,minerva_math_geometry,minerva_math_intermediate_algebra,minerva_math_num_theory,minerva_math_prealgebra,minerva_math_precalc"

run () {  # run <alias> <model_path_abs> <label> <lm_eval_tasks> <max_new_tokens> <limit> [extra...]
  local alias=$1 model=$2 label=$3 taskspec=$4 maxtok=$5 limit=$6; shift 6
  local outdir="${OUTDIR}/${alias}/${label}"
  if [ "${SKIP_EXISTING}" = "1" ] && [ "${FORCE}" != "1" ] && \
     compgen -G "${outdir}/results_*.json" > /dev/null; then
    echo "=== $(date +%H:%M:%S) SKIP ${alias}/${label} (deja termine)"
    return 0
  fi
  echo "=== $(date +%H:%M:%S) ${alias} | ${label} | max_new_tokens=${maxtok} limit=${limit} batch=${BATCH} seed=${SEED}"
  local stage="${STAGE}/${alias}__${label}"
  rm -rf "${stage}"; mkdir -p "${stage}"
  ( cd "${FORK}/eval_instruct" && PYTHONPATH=. "${VENV}/bin/python" -m lm_eval \
      --model diffllm \
      --model_args "pretrained=${model},trust_remote_code=True,max_new_tokens=${maxtok},diffusion_steps=${maxtok},dtype=bfloat16,temperature=0.1,top_p=0.9,alg=entropy" \
      --tasks "${taskspec}" \
      --include_path "${TASKS_A2D}" \
      --num_fewshot 0 \
      --batch_size "${BATCH}" \
      --limit "${limit}" \
      --seed "${SEED}" \
      --output_path "${stage}" \
      --log_samples --confirm_run_unsafe_code \
      --apply_chat_template "$@" ) 2>&1 \
    | tee "${LOG_DIR}/${alias}_${label}.log" | tail -12
  local rc=${PIPESTATUS[0]}
  # lm_eval ecrit dans <output_path>/<nom_model_sanitise>/ : on aplatit dans
  # <OUTDIR>/<alias>/<label>/ (noms de dossiers portables, pas de chemin absolu).
  mkdir -p "${outdir}"
  find "${stage}" -maxdepth 2 -type f \( -name 'results_*.json' -o -name 'samples_*.jsonl' \) \
    -exec mv -t "${outdir}" {} +
  echo "=== $(date +%H:%M:%S) fin ${alias}/${label} (rc=${rc}) -> ${outdir}"
  return "${rc}"
}

# Alias de sortie = nom du dossier du checkpoint ; le chemin est resolu en
# absolu (le processus lm_eval tourne dans le fork, pas a la racine).
resolve_model () {  # resolve_model <spec> -> "<alias>|<chemin_absolu>"
  local spec=$1 path
  case "${spec}" in
    /*) path="${spec}" ;;
    *)  path="${REPO_ROOT}/${spec}" ;;
  esac
  printf '%s|%s\n' "$(basename "${path}")" "${path}"
}

status=0
# Entrelacement par tache (base puis instruct pour chaque tache) : si le
# budget GPU manque, les deux modeles restent couverts par les memes taches.
for task in ${TASK//,/ }; do
  for spec in ${MODEL_DIR//,/ }; do
    resolved="$(resolve_model "${spec}")"
    alias="${resolved%%|*}"; model="${resolved##*|}"
    case "${task}" in
      gsm8k)  run "${alias}" "${model}" gsm8k_cot_a2d gsm8k_cot_a2d 256 "$(limit_for gsm8k)" \
                --system_instruction "${GSM8K_SYSTEM}" || status=1 ;;
      math)   run "${alias}" "${model}" minerva_math_algebra "${MATH_TASKS}" 512 "$(limit_for math)" || status=1 ;;
      ifeval) run "${alias}" "${model}" ifeval_a2d ifeval_a2d 1280 "$(limit_for ifeval)" || status=1 ;;
      *)      run "${alias}" "${model}" "${task}" "${task}" 512 "$(limit_for "${task}")" || status=1 ;;
    esac
  done
done
echo "ALL DONE $(date +%H:%M:%S) (status=${status})"
exit "${status}"
