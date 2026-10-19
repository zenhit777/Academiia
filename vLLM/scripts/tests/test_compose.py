"""Test de non-régression pour vLLM/docker-compose.yml (pytest).

Reproduit le pipeline RÉEL : Compose interpole d'abord `$VAR`/`${VAR}`
(y compris `${V:-défaut}`) dans TOUT le fichier — seul `$$` produit un `$`
littéral — puis le shell du conteneur exécute le script résultant via
`bash -c` (simulé ici avec une fonction vllm() imprimant ses arguments).

Vérifie :
  - entrypoint == ["bash","-c"] ; command = liste d'UN element ;
  - aucun backslash dans le script (bug 686c5a8) ;
  - interpolation Compose conforme ($$ -> $, ${V}, $V, ${V:-d}, ${V-d}) ;
  - (1) défauts -> --enforce-eager présent ; (2) VLLM_ENFORCE_EAGER=0 ->
    absent ; (3) VLLM_EXTRA_ARGS multi-flags -> arguments distincts ;
  - (4) aucun argument vide/réduit à un espace ;
  - (5) aucun ${...}/$VAR résiduel inattendu dans le script interpolé ;
  - VLLM_USE_V2_MODEL_RUNNER absent de environment: ;
  - override runner exporté seulement s'il est défini ;
  - PYTORCH_CUDA_ALLOC_CONF configurable : defaut si la variable n'est pas
    définie, valeur VIDE conservée si elle est définie vide (forme `${V-d}`,
    PAS `${V:-d}` qui réappliquerait le défaut) ;
  - si `docker compose` est disponible : la commande RÉSOLUE contient
    bien $${EAGER} (échappé pour le shell), sinon test skipped.
"""

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

COMPOSE_PATH = Path(__file__).resolve().parents[2] / "docker-compose.yml"

HARNESS = (
    "vllm() { echo \"RUNNER=${VLLM_USE_V2_MODEL_RUNNER:-<unset>}\"; "
    'for a in "$@"; do echo "ARG:$a"; done; } ; '
)


def load_compose() -> dict:
    with open(COMPOSE_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


# --- Interpolation conforme à Compose (spécification : $$ -> $, ${V}, $V,
# --- ${V:-d} (défaut si non défini OU vide), ${V-d} (défaut si non défini)).

_VAR_BRACED = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::?-([^}]*))?\}")
_VAR_BARE = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")


def compose_interpolate(script: str, env: dict[str, str]) -> str:
    """Interpole comme Compose : $$ d'abord (protect), puis ${V...}/$V.

    Compose remplace $$ par un marqueur unique, résout les variables,
    puis restaure les $ littéraux.

    Sémantique des défauts (spécification Compose) :
      - `${V-d}`  : défaut si V n'est PAS DÉFINIE (V="" reste "") ;
      - `${V:-d}` : défaut si V n'est pas définie OU est vide.
    """
    DOLLAR = "\x00DOLLAR\x00"
    out = script.replace("$$", DOLLAR)

    def sub_braced(m: re.Match) -> str:
        name, default = m.group(1), m.group(2)
        if default is None:
            return env.get(name, "")
        if name not in env:
            return default
        if ":-" in m.group(0) and env[name] == "":
            return default
        return env[name]

    out = _VAR_BRACED.sub(sub_braced, out)
    out = _VAR_BARE.sub(lambda m: env.get(m.group(1), ""), out)
    return out.replace(DOLLAR, "$")


def run_script(env_compose: dict[str, str], env_shell: dict[str, str] | None = None) -> tuple[str, list[str]]:
    """Interpole (Compose) puis exécute via bash -c (harnais vllm)."""
    script = load_compose()["services"]["vllm"]["command"][0]
    interpolated = compose_interpolate(script, env_compose)
    assert "exec vllm" in interpolated
    env = {k: v for k, v in os.environ.items() if not k.startswith("VLLM_")}
    env.update(env_shell or {})
    proc = subprocess.run(
        ["bash", "-c", HARNESS + interpolated.replace("exec vllm", "vllm")],
        env=env, capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, f"bash a échoué : {proc.stderr}"
    lines = proc.stdout.splitlines()
    runner = lines[0].removeprefix("RUNNER=")
    args = [l[4:] for l in lines[1:] if l.startswith("ARG:")]
    return runner, args


def env_shell_for(env_compose: dict[str, str]) -> dict[str, str]:
    """Variables que Compose passerait aussi au conteneur via environment:."""
    shell = {}
    if "VLLM_USE_V2_MODEL_RUNNER_OVERRIDE" in env_compose:
        shell["VLLM_USE_V2_MODEL_RUNNER_OVERRIDE"] = env_compose["VLLM_USE_V2_MODEL_RUNNER_OVERRIDE"]
    return shell


# --- Structure YAML ---

def test_entrypoint_bash_c():
    assert load_compose()["services"]["vllm"]["entrypoint"] == ["bash", "-c"]


def test_command_single_element():
    command = load_compose()["services"]["vllm"]["command"]
    assert isinstance(command, list) and len(command) == 1


def test_no_backslash_in_script():
    script = load_compose()["services"]["vllm"]["command"][0]
    assert "\\" not in script, "backslash dans le bloc >- (bug 686c5a8)"


def test_no_v2_runner_in_environment():
    for entry in load_compose()["services"]["vllm"]["environment"]:
        key = entry.split("=", 1)[0]
        assert key != "VLLM_USE_V2_MODEL_RUNNER"


# --- (5) Pas de $ résiduel inattendu après interpolation Compose ---

def test_no_uninterpolated_dollar_residue():
    script = load_compose()["services"]["vllm"]["command"][0]
    out = compose_interpolate(script, {"VLLM_EXTRA_ARGS": ""})
    residue = re.findall(r"\$\{[^}]*\}|\$[A-Za-z_][A-Za-z0-9_]*", out)
    # Seuls restent les $ littéraux voulus (shell du conteneur).
    allowed = {"${VLLM_USE_V2_MODEL_RUNNER_OVERRIDE:-}", "$VLLM_USE_V2_MODEL_RUNNER_OVERRIDE",
               "${EAGER}"}
    unexpected = [r for r in residue if r not in allowed]
    assert not unexpected, f"variables non interpolées par Compose ni échappées en $$ : {unexpected}"


# --- (1)(2)(3)(4) Simulation bash après interpolation Compose ---

def test_enforce_eager_present_by_default():
    _, args = run_script({})
    assert "--enforce-eager" in args, f"--enforce-eager absent par défaut : {args}"
    bad = [a for a in args if not a.strip()]
    assert not bad, f"arguments vides : {bad}"


def test_enforce_eager_absent_when_disabled():
    _, args = run_script({"VLLM_ENFORCE_EAGER": "0"})
    assert "--enforce-eager" not in args, f"doit être absent si VLLM_ENFORCE_EAGER=0 : {args}"


def test_extra_args_split_into_distinct_args():
    _, args = run_script({"VLLM_EXTRA_ARGS": "--seed 1 --max-num-batched-tokens 2048"})
    for expected in ("--seed", "1", "--max-num-batched-tokens", "2048"):
        assert expected in args, f"{expected} absent/distinct : {args}"
    bad = [a for a in args if not a.strip()]
    assert not bad, f"arguments vides avec VLLM_EXTRA_ARGS : {bad}"


def test_runner_override_exported_only_when_set():
    runner, _ = run_script({"VLLM_USE_V2_MODEL_RUNNER_OVERRIDE": "0"},
                           env_shell={"VLLM_USE_V2_MODEL_RUNNER_OVERRIDE": "0"})
    assert runner == "0"
    runner, _ = run_script({})
    assert runner == "<unset>", f"override fuiterait dans le défaut : {runner}"


# --- Sémantique d'interpolation Compose (forme avec / sans « : ») ---

def test_interpolation_default_forms():
    """`${V-d}` (défaut si NON définie) vs `${V:-d}` (défaut si vide).

    Sémantique vérifiée contre `docker compose config` (voir
    test_compose_config_resolves_alloc_conf).
    """
    assert compose_interpolate("${V-def}", {}) == "def"
    assert compose_interpolate("${V-def}", {"V": ""}) == ""
    assert compose_interpolate("${V-def}", {"V": "x"}) == "x"
    assert compose_interpolate("${V:-def}", {}) == "def"
    assert compose_interpolate("${V:-def}", {"V": ""}) == "def"
    assert compose_interpolate("${V:-def}", {"V": "x"}) == "x"
    assert compose_interpolate("$V", {"V": "x"}) == "x"
    assert compose_interpolate("$$V", {}) == "$V"


def test_default_extra_args_enable_fp8():
    """Par défaut, le compose passe `--quantization fp8` (2 arguments distincts)."""
    _, args = run_script({})
    assert "--quantization" in args, f"--quantization absent par défaut : {args}"
    assert "fp8" in args, f"fp8 absent par défaut : {args}"
    bad = [a for a in args if not a.strip()]
    assert not bad, f"arguments vides : {bad}"


# --- PYTORCH_CUDA_ALLOC_CONF configurable (variable d'env du conteneur) ---

ALLOC_VAR = "PYTORCH_CUDA_ALLOC_CONF"
ALLOC_DEFAULT = "expandable_segments:True"


def alloc_entry() -> str:
    """Entrée environment: de PYTORCH_CUDA_ALLOC_CONF (forme 'CLE=valeur')."""
    for entry in load_compose()["services"]["vllm"]["environment"]:
        if entry.split("=", 1)[0] == ALLOC_VAR:
            return entry
    raise AssertionError(f"{ALLOC_VAR} absent de environment:")


def resolve_alloc(env: dict[str, str]) -> str:
    """Valeur résolue de l'entrée environment: (interpolation Compose)."""
    value = alloc_entry().split("=", 1)[1]
    return compose_interpolate(value, env)


def test_alloc_conf_uses_dash_form_not_colon_form():
    """`${V-d}` (défaut si NON définie) et NON `${V:-d}` (défaut si vide).

    Avec `${V:-d}`, une ligne `PYTORCH_CUDA_ALLOC_CONF=` (vide) dans .env
    serait réécrite par le défaut -> impossible de désactiver
    `expandable_segments` (cf. expérience E1 du 2026-10-19).
    """
    value = alloc_entry().split("=", 1)[1]
    expected_form = "${" + ALLOC_VAR + "-"
    forbidden_form = "${" + ALLOC_VAR + ":-"
    assert expected_form in value, f"forme attendue {expected_form}defaut : {value}"
    assert forbidden_form not in value, (
        f"forme {forbidden_form}defaut réapplique le défaut à une valeur vide : {value}"
    )


def test_alloc_conf_default_when_undefined():
    assert resolve_alloc({}) == ALLOC_DEFAULT


def test_alloc_conf_empty_value_is_preserved():
    """Valeur vide dans .env => variable vide (expandable_segments désactivé)."""
    assert resolve_alloc({ALLOC_VAR: ""}) == ""


def test_alloc_conf_override_value():
    assert resolve_alloc({ALLOC_VAR: "garbage_fragment:1234"}) == "garbage_fragment:1234"


def test_compose_config_resolves_alloc_conf(tmp_path):
    """Vérifie la résolution RÉELLE par `docker compose config` (3 cas)."""
    if shutil.which("docker") is None:
        pytest.skip("docker compose non disponible sur cette machine")
    env = {k: v for k, v in os.environ.items() if k != ALLOC_VAR}
    cases = {
        "defaut": None,  # variable non définie -> défaut du compose
        "vide": "",
        "surcharge": "garbage_fragment:1234",
    }
    for label, value in cases.items():
        env_file = tmp_path / f"env_{label}"
        content = "" if value is None else f"{ALLOC_VAR}={value}\n"
        env_file.write_text(content, encoding="utf-8")
        proc = subprocess.run(
            ["docker", "compose", "--env-file", str(env_file), "config"],
            cwd=COMPOSE_PATH.parent, capture_output=True, text=True, timeout=120, env=env,
        )
        assert proc.returncode == 0, f"docker compose config a échoué ({label}) : {proc.stderr}"
        resolved = [
            line.split(":", 1)[1].strip().strip('"')
            for line in proc.stdout.splitlines()
            if line.strip().startswith(f"{ALLOC_VAR}:")
        ]
        assert len(resolved) == 1, f"entrée {ALLOC_VAR} introuvable/dupliquée : {resolved}"
        expected = ALLOC_DEFAULT if value is None else value
        assert resolved[0] == expected, (
            f"{label} : {ALLOC_VAR} résolu à {resolved[0]!r}, attendu {expected!r}"
        )


# --- docker compose config (si disponible) ---

def test_compose_config_resolved_command():
    if shutil.which("docker") is None:
        pytest.skip("docker compose non disponible sur cette machine")
    proc = subprocess.run(
        ["docker", "compose", "config"],
        cwd=COMPOSE_PATH.parent, capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, f"docker compose config a échoué : {proc.stderr}"
    # $$EAGER doit SURVIVRE à l'interpolation Compose (littéral pour le shell) ;
    # --enforce-eager est positionné par le shell, pas visible en statique.
    assert "$${EAGER}" in proc.stdout or "${EAGER}" in proc.stdout, (
        "l'interpolation Compose a avalé ${EAGER} -> --enforce-eager ne serait jamais passé"
    )
