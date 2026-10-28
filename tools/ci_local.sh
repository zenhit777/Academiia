#!/usr/bin/env bash
# Miroir local du job `checks` de .github/workflows/ci.yml (Python 3.12, sans GPU).
#
# Usage : bash tools/ci_local.sh      # sortie != 0 a la premiere etape echouee
#
# Fidelite : les 12 etapes sont dans le meme ordre et avec les memes commandes
# que le workflow ; l'etape d'installation pip est sautee (l'environnement local
# est deja installe, `pip install -e ".[dev,eval,peft]"`).
#
# Regle 9 (AGENTS.md) : TMPDIR et PIP_CACHE_DIR sont dans le depot
# (`_artefact/`). La capture de pytest est forcee en `--capture=sys` : la
# capture par defaut cree une tmpfile sous TMPDIR, et DrvFs (/mnt/d) ne
# supporte pas `O_TMPFILE` -> FileNotFoundError au truncate de fin de session
# (note d'inventaire 2026-10-21 dans modules/a2d/docs/reports/OUT_OF_WORKSPACE.md
# §1.3). Avec `--capture=sys`, les temporaires restent dans le depot et la
# suite est vert.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
TMPDIR="$REPO/_artefact/ci_local_tmp"
PIP_CACHE_DIR="$REPO/_artefact/ci_pip_cache"
export TMPDIR PIP_CACHE_DIR
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"

STEP=0
fail() { echo "### ETAPE $STEP echouee : $1 (sortie $2)"; exit 1; }

run() {
  local n="$1" name="$2"
  shift 2
  STEP="$n"
  echo "### [$n] $name"
  local rc=0
  "$@" || rc=$?
  [ "$rc" -eq 0 ] || fail "$name" "$rc"
}

run 1 "Imports des paquets installes" \
  python3 -c "import academiia.core, academiia.a2d, academiia.dopd; print('OK')"

run 2 "Entry-points CLI (--help)" bash -c '
  for cli in merge-a2d verify-merge measure-geometry verify-checkpoint download-models; do
    echo "--- ${cli} --help"
    "${cli}" --help > /dev/null || exit 1
  done
  echo "5/5 entry-points OK"'

run 3 "Lint (ruff, regles E/F, perimetre complet)" \
  python3 -m ruff check academiia/ modules/ tests/ tools/ infra/

run 4 "Syntaxe recursive des modules et de l'infra" python3 - <<'EOF'
import ast, glob
files = (sorted(glob.glob('academiia/**/*.py', recursive=True))
         + sorted(glob.glob('modules/a2d/**/*.py', recursive=True))
         + sorted(glob.glob('modules/dopd/**/*.py', recursive=True))
         + sorted(glob.glob('infra/vllm/**/*.py', recursive=True)))
assert files, "aucun script trouve"
for f in files:
    ast.parse(open(f, encoding='utf-8').read(), filename=f)
print(f"syntax OK : {len(files)} fichiers")
EOF

run 5 "Compilation du correctif B1 (verify_checkpoint.py)" \
  python3 -m py_compile modules/a2d/cli/verify_checkpoint.py

run 6 "Merge legacy (merge_jainius.py) sur mini-checkpoint" bash -c '
python3 - <<EOF
import json, os
import torch
from safetensors.torch import save_file
d = os.path.join(os.environ["TMPDIR"], "dummy_dg")
t = os.path.join(os.environ["TMPDIR"], "dummy_tau")
os.makedirs(d, exist_ok=True)
os.makedirs(t, exist_ok=True)
save_file({"model.decoder.layers.0.self_attn.q_proj.weight": torch.ones(8, 8, dtype=torch.bfloat16)},
          os.path.join(d, "model-00001-of-00001.safetensors"))
json.dump({"metadata": {}, "weight_map": {"model.decoder.layers.0.self_attn.q_proj.weight": "model-00001-of-00001.safetensors"}},
          open(os.path.join(d, "model.safetensors.index.json"), "w"))
json.dump({"architectures": ["DiffusionGemmaForBlockDiffusion"]}, open(os.path.join(d, "config.json"), "w"))
for f in ["tokenizer.json", "tokenizer_config.json", "generation_config.json", "chat_template.jinja", "processor_config.json"]:
    open(os.path.join(d, f), "w").write("{}")
import numpy as np
json.dump({"metadata": {}, "weight_map": {"model.language_model.layers.0.self_attn.q_proj.weight": "model_language_model_layers_0_self_attn_q_proj_weight.npy"}},
          open(os.path.join(t, "tau_manifest.json"), "w"))
np.save(os.path.join(t, "model_language_model_layers_0_self_attn_q_proj_weight.npy"),
        np.full((8, 8), 0.05, dtype=np.float32))
EOF
python3 modules/a2d/legacy/scripts/merge_jainius.py --diffusiongemma "${TMPDIR}/dummy_dg" --tau-dir "${TMPDIR}/dummy_tau" --output "${TMPDIR}/dummy_out" --alpha 0.3 --overwrite
python3 modules/a2d/cli/verify_checkpoint.py --checkpoint "${TMPDIR}/dummy_out"
python3 modules/a2d/legacy/scripts/measure_effect.py --diffusiongemma "${TMPDIR}/dummy_dg" --tau-dir "${TMPDIR}/dummy_tau" --alphas 0.3 --output "${TMPDIR}/measure.json"'

run 7 "Merge A2D conforme (merge_a2d.py) + verification (verify_merge.py)" bash -c '
python3 - <<EOF
import json, os
import torch
from safetensors.torch import save_file
root = os.path.join(os.environ["TMPDIR"], "a2d")
spec = {
    "model.embed_tokens.weight": (8, 4),
    "model.layers.0.self_attn.q_proj.weight": (4, 4),
    "model.layers.0.mlp.gate_proj.weight": (6, 4),
    "model.norm.weight": (4,),
    "lm_head.weight": (8, 4),
}
for name, seed in (("ar_base", 1), ("ar_post", 2), ("d_base", 3), ("d_post", 4)):
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    g = torch.Generator().manual_seed(seed)
    tensors = {k: torch.randn(*s, generator=g) for k, s in spec.items()}
    save_file(tensors, os.path.join(d, "model-00001-of-00001.safetensors"))
    json.dump({"metadata": {"total_size": 0},
               "weight_map": {k: "model-00001-of-00001.safetensors" for k in spec}},
              open(os.path.join(d, "model.safetensors.index.json"), "w"))
    json.dump({"architectures": ["Toy"]}, open(os.path.join(d, "config.json"), "w"))
EOF
# A2D-Transfer : D_b + alpha*tau_A
python3 modules/a2d/cli/merge_a2d.py --mode transfer \
  --diffusion-base "${TMPDIR}/a2d/d_base" \
  --tau-a-post "${TMPDIR}/a2d/ar_post" --tau-a-base "${TMPDIR}/a2d/ar_base" \
  --alpha 0.9 --mapping identity --output "${TMPDIR}/a2d/transfer"
python3 modules/a2d/cli/verify_merge.py --merged "${TMPDIR}/a2d/transfer" --reference "${TMPDIR}/a2d/d_base" \
  --mode transfer --tau-a-post "${TMPDIR}/a2d/ar_post" --tau-a-base "${TMPDIR}/a2d/ar_base" \
  --alpha 0.9 --mapping identity
# A2D-Merge : D_p + lambda*(tau_A - tau_D)
python3 modules/a2d/cli/merge_a2d.py --mode merge \
  --diffusion "${TMPDIR}/a2d/d_post" \
  --tau-a-post "${TMPDIR}/a2d/ar_post" --tau-a-base "${TMPDIR}/a2d/ar_base" \
  --tau-d-post "${TMPDIR}/a2d/d_post" --tau-d-base "${TMPDIR}/a2d/d_base" \
  --lambda 0.6 --mapping identity --output "${TMPDIR}/a2d/merge"
python3 modules/a2d/cli/verify_merge.py --merged "${TMPDIR}/a2d/merge" --reference "${TMPDIR}/a2d/d_post" \
  --mode merge --tau-a-post "${TMPDIR}/a2d/ar_post" --tau-a-base "${TMPDIR}/a2d/ar_base" \
  --tau-d-post "${TMPDIR}/a2d/d_post" --tau-d-base "${TMPDIR}/a2d/d_base" \
  --lambda 0.6 --mapping identity
# un coefficient errone doit etre detecte
if python3 modules/a2d/cli/verify_merge.py --merged "${TMPDIR}/a2d/transfer" --reference "${TMPDIR}/a2d/d_base" \
    --mode transfer --tau-a-post "${TMPDIR}/a2d/ar_post" --tau-a-base "${TMPDIR}/a2d/ar_base" \
    --alpha 0.3 --mapping identity; then
  echo "ERREUR: verify_merge aurait du detecter alpha=0.3 sur un merge a 0.9"
  exit 1
fi'

run 8 "Suite de tests (hors mutations)" \
  python3 -m pytest modules/a2d/tests modules/dopd/tests tests infra/vllm/scripts/tests -q \
  --ignore=modules/a2d/tests/test_a2d_mutation.py --capture=sys

run 9 "Tests adversariaux (mutations) de la bibliotheque A2D" \
  python3 -m pytest modules/a2d/tests/test_a2d_mutation.py -v --capture=sys

run 10 "Modes A2D non couverts par le test unitaire (CLI)" bash -c '
# 1. ablation-dp-minus-tau-d : doit FUSIONNER tau_D (BLK-01)
python3 modules/a2d/cli/merge_a2d.py --mode ablation-dp-minus-tau-d \
  --diffusion "${TMPDIR}/a2d/d_post" \
  --tau-d-post "${TMPDIR}/a2d/d_post" --tau-d-base "${TMPDIR}/a2d/d_base" \
  --lambda 0.5 --mapping identity --output "${TMPDIR}/a2d/minus"
python3 modules/a2d/cli/verify_merge.py --merged "${TMPDIR}/a2d/minus" --reference "${TMPDIR}/a2d/d_post" \
  --mode ablation-dp-minus-tau-d \
  --tau-d-post "${TMPDIR}/a2d/d_post" --tau-d-base "${TMPDIR}/a2d/d_base" \
  --lambda 0.5 --mapping identity
# 2. verify_merge --mode merge SANS tau_D doit etre refuse (MAJ-01)
if python3 modules/a2d/cli/verify_merge.py --merged "${TMPDIR}/a2d/transfer" --reference "${TMPDIR}/a2d/d_base" \
    --mode merge --tau-a-post "${TMPDIR}/a2d/ar_post" --tau-a-base "${TMPDIR}/a2d/ar_base" \
    --lambda 0.6 --mapping identity; then
  echo "ERREUR: verify_merge aurait du refuser le mode merge sans tau_D"
  exit 1
fi
# 3. un tau_D partiel doit etre refuse en politique strict (BLK-02)
python3 - <<EOF
import json, os
import torch
from safetensors.torch import save_file
root = os.path.join(os.environ["TMPDIR"], "a2d")
spec = {"model.layers.0.self_attn.q_proj.weight": (4, 4)}
for name, seed in (("partial_base", 201), ("partial_post", 202)):
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    g = torch.Generator().manual_seed(seed)
    save_file({k: torch.randn(*s, generator=g) for k, s in spec.items()},
              os.path.join(d, "model-00001-of-00001.safetensors"))
    json.dump({"metadata": {"total_size": 0}, "weight_map": {k: "model-00001-of-00001.safetensors" for k in spec}},
              open(os.path.join(d, "model.safetensors.index.json"), "w"))
EOF
if python3 modules/a2d/cli/merge_a2d.py --mode merge \
    --diffusion "${TMPDIR}/a2d/d_post" \
    --tau-a-post "${TMPDIR}/a2d/ar_post" --tau-a-base "${TMPDIR}/a2d/ar_base" \
    --tau-d-post "${TMPDIR}/a2d/partial_post" --tau-d-base "${TMPDIR}/a2d/partial_base" \
    --lambda 0.6 --mapping identity --output "${TMPDIR}/a2d/partial_merge"; then
  echo "ERREUR: un tau_D partiel aurait du etre refuse en politique strict"
  exit 1
fi
# 4. en politique zero-fill, le meme cas passe et est declare
python3 modules/a2d/cli/merge_a2d.py --mode merge \
  --diffusion "${TMPDIR}/a2d/d_post" \
  --tau-a-post "${TMPDIR}/a2d/ar_post" --tau-a-base "${TMPDIR}/a2d/ar_base" \
  --tau-d-post "${TMPDIR}/a2d/partial_post" --tau-d-base "${TMPDIR}/a2d/partial_base" \
  --lambda 0.6 --mapping identity --coverage zero-fill --output "${TMPDIR}/a2d/partial_zf"
python3 - <<EOF
import json, os
m = json.load(open(os.path.join(os.environ["TMPDIR"], "a2d/partial_zf/merge_config.json")))
assert m["coverage"]["policy"] == "zero-fill", m["coverage"]
assert m["coverage"]["n_zero_filled"] > 0, m["coverage"]
assert m["post_write_verification"]["ok"], m["post_write_verification"]
print("zero-fill declare :", m["coverage"]["n_zero_filled"], "modules")
EOF'

run 11 "Liens relatifs des fichiers Markdown" \
  python3 tools/check_links.py

run 12 "Syntaxe shell de l'infra + compilation smoke_chat.py" bash -c '
  bash -n infra/vllm/diagnose_image.sh infra/vllm/smoke.sh
  python3 -m py_compile infra/vllm/smoke_chat.py'

echo "### CI locale verte sur HEAD $(git rev-parse --short HEAD)"
