"""Moteur de fusion A2D (transfert, merge, ablations).

Formules (A2D §2.4, PDF p. 6) :

    tau_A = A_p - A_b                          (§2.1)
    tau_D = D_p - D_b                          (§2.1)
    D_Transfer(alpha) = D_b + alpha * tau_A                    (eq. 1)
    D_Merge(lambda)  = D_b + (1-lambda)*tau_D + lambda*tau_A
                     = D_p + lambda * (tau_A - tau_D)           (eq. 2)

Modes de la bibliotheque :

    transfer                D_b + alpha * tau_A
    merge                   D_p + lambda * (tau_A - tau_D)
    ablation-dp-plus-tau-a  D_p + alpha * tau_A        (Table 9 du papier)
    ablation-dp-minus-tau-d D_p - lambda * tau_D        (Table 10 du papier)

Proprietes garanties :

* tau n'est JAMAIS persiste sur disque : il est recalcule a la volee,
  tenseur par tenseur, depuis les sources (3-4 handles `safe_open`), en FP32.
* arithmetique FP32 sequentielle, UN SEUL cast BF16 (A2D App. A.3, PDF p. 20).
* ecriture atomique (fichier temporaire + `os.replace`).
* verification POST-ECRITURE obligatoire : relecture des tenseurs ecrits,
  regression des moindres carres SIGNEE sur [tau_A, tau_D], comparaison des
  coefficients signes, identite binaire des tenseurs non fusionnes, ecart
  maximal <= 1 ULP par rapport a `round_bf16(expected)` recalcule.

Choix structurants (J6bis, B1-B5) :

* le plan de fusion est construit sur l'UNION des modules de tau_A et de tau_D
  (chaque module est classe `both` / `A-only` / `D-only`) ;
* la politique de couverture `--coverage {strict,zero-fill}` est explicite :
  `strict` refuse une couverture inegale, `zero-fill` traite un vecteur
  absent comme nul et le declare au manifeste ;
* il existe UNE SEULE definition de la valeur attendue (`expected_tensor`),
  utilisee par le writer ET par `verify_merge` ;
* la verification compare des coefficients SIGNES (projection), pas des normes.
"""

from __future__ import annotations

import datetime
import json
import math
import os
import shutil
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .naming import (FAMILY_EMBED, IDENTITY, NameMapping, family_of, module_of)
from .sources import VectorSource, load_weight_map

MODES = ("transfer", "merge", "ablation-dp-plus-tau-a", "ablation-dp-minus-tau-d")

#: Seuil d'avertissement pour ALPHA (au-dela, le merge degrade generalement le
#: modele, cf. A2D Fig. 8). L'eq. (1) ne plafonne pas alpha (Table 6 : 1.4).
COEFFICIENT_WARNING_THRESHOLD = 1.5
#: L'eq. (2) ne borne lambda que dans [0,1] : hors de cet intervalle, le role
#: de D_b et de D_p s'inverse.
LAMBDA_MAX = 1.0

#: Politiques de couverture des sources.
COVERAGE_POLICIES = ("strict", "zero-fill")

#: Au-dela de ce cosinus, la regression a 2 regresseurs est mal conditionnee :
#: on replie sur des projections simples (et on avertit).
ILL_CONDITIONED_COS = 0.99


class MergeError(RuntimeError):
    """Erreur metier du merge (exit 1)."""


class MergeRefused(MergeError):
    """Refus d'ecraser / etat incoherent (exit 2)."""


class MissingVectorError(MergeError):
    """Vecteur de tache manquant pour le mode demande."""


class ShapeMismatchError(MergeError):
    """Forme incompatible entre le vecteur et le tenseur receveur."""


class CoverageError(MergeError):
    """Couverture inegale entre tau_A et tau_D (politique `strict`)."""


@dataclass(frozen=True)
class MergePolicy:
    """Politiques appliquees pendant la fusion et la verification.

    * `coverage` : `strict` (couverture inegale => erreur) ou `zero-fill`
      (un vecteur absent est traite comme nul, et declare au manifeste) ;
    * `embedding_policy` : `rows` (intersection des lignes d'embedding si le
      vocabulaire differe) ou `strict` (erreur sur forme differente) ;
    * `on_shape_mismatch` : `error` ou `skip` (tenseur laisse inchange).
    """

    coverage: str = "strict"
    embedding_policy: str = "rows"
    on_shape_mismatch: str = "error"


@dataclass
class PlanEntry:
    """Un module du plan de fusion.

    * `source_a` : module de l'espace AR (tau_A), a mapper vers le receveur ;
    * `source_d` : module de l'espace du receveur (tau_D est deja dans
      l'espace de diffusion) ;
    * `coverage` : `both` / `A-only` / `D-only`.
    """

    family: str
    source_a: Optional[str] = None
    source_d: Optional[str] = None
    coverage: str = "both"


@dataclass
class Expected:
    """Resultat de `expected_tensor`."""

    value: torch.Tensor
    delta_a: Optional[torch.Tensor] = None
    delta_d: Optional[torch.Tensor] = None
    skipped: bool = False
    note: Optional[str] = None


@dataclass
class MergeConfig:
    mode: str = "transfer"
    recipient: Path = Path(".")
    output: Path = Path(".")
    alpha: float = 0.0
    lam: float = 0.0
    mapping: NameMapping = IDENTITY
    tau_a: Optional[VectorSource] = None
    tau_d: Optional[VectorSource] = None
    dtype: torch.dtype = torch.bfloat16
    on_shape_mismatch: str = "error"
    coverage: Optional[str] = None
    allow_empty: bool = False
    allow_empty_coefficient: bool = False
    overwrite: bool = False
    tol_abs: float = 0.02
    max_ulp: float = 1.0
    verify: bool = True
    copy_sidecars: bool = True
    extra_manifest: dict = field(default_factory=dict)


@dataclass
class MergeResult:
    manifest: dict
    verification: dict
    n_fused: int = 0
    n_unchanged: int = 0


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #

def mode_coefficients(mode: str, alpha: float, lam: float) -> Tuple[float, float]:
    """Coefficients (c_a, c_d) appliques au receveur : out = W + c_a*tau_A + c_d*tau_D."""
    if mode == "transfer":
        return alpha, 0.0
    if mode == "merge":
        return lam, -lam
    if mode == "ablation-dp-plus-tau-a":
        return alpha, 0.0
    if mode == "ablation-dp-minus-tau-d":
        return 0.0, -lam
    raise ValueError(f"mode inconnu : {mode!r} (attendu : {MODES})")


def effective_coefficient(mode: str, alpha: float, lam: float) -> Tuple[str, float]:
    """(nom du coefficient, valeur) qui doit etre retrouve par la verification."""
    if mode in ("transfer", "ablation-dp-plus-tau-a"):
        return "alpha", alpha
    return "lambda", lam


def required_sources(mode: str) -> frozenset:
    """Sources de vecteur utilisees par le mode : {'A'}, {'D'} ou {'A','D'}."""
    if mode in ("transfer", "ablation-dp-plus-tau-a"):
        return frozenset({"A"})
    if mode == "ablation-dp-minus-tau-d":
        return frozenset({"D"})
    return frozenset({"A", "D"})


def checked_coefficients(mode: str) -> Tuple[str, ...]:
    """Coefficients dont la projection signee est comparee a la valeur attendue."""
    if mode in ("transfer", "ablation-dp-plus-tau-a"):
        return ("a",)
    if mode == "ablation-dp-minus-tau-d":
        return ("d",)
    return ("a", "d")


# --------------------------------------------------------------------------- #
# Garde-fou « coefficient nul »
# --------------------------------------------------------------------------- #

def first_nonzero_delta(cfg: MergeConfig, plan: Dict[str, PlanEntry]) -> Optional[str]:
    """Premier tenseur du plan dont un vecteur source est non nul (None sinon).

    Le parcours s'arrete au premier delta non nul : dans le cas courant (un
    seul couple de tenseurs lu) le cout est borne ; si tous les deltas sont
    nuls, le plan entier est parcouru (cas pathologique : A_p == A_b).
    """
    for tensor_name, entry in plan.items():
        for source, module in ((cfg.tau_a, entry.source_a), (cfg.tau_d, entry.source_d)):
            if source is None or module is None:
                continue
            delta = source.delta(module)
            if delta is None:
                continue
            if int(torch.count_nonzero(delta).item()) > 0:
                return tensor_name
    return None


def evaluate_coefficient_guard(cfg: MergeConfig,
                               plan: Dict[str, PlanEntry]) -> Dict[str, object]:
    """Garde-fou : un transfert/merge a coefficient nul est refuse par defaut.

    Un transfert a `alpha=0` (eq. 1) ecrit une simple **copie de D_b** ; un
    merge a `lambda=0` (eq. 2) ecrit une simple **copie de D_p**. La
    verification post-ecriture ne peut pas le voir : elle compare les
    coefficients effectifs aux coefficients **demandes** (0,0). Le resultat
    est donc indiscernable d'un modele non fusionne.

    Le garde-fou refuse l'operation si le plan contient des deltas non nuls.
    `allow_empty_coefficient` (CLI : `--allow-empty-coefficient`) l'assume
    explicitement ; l'avertissement est alors inscrit au manifeste.

    Renvoie l'etat du garde-fou (inscrit au manifeste) ; leve `MergeRefused`
    (exit 2) si l'operation doit etre refusee.
    """
    coeff_name, coeff_value = effective_coefficient(cfg.mode, cfg.alpha, cfg.lam)
    state: Dict[str, object] = {
        "coefficient": coeff_name,
        "value": coeff_value,
        "zero_coefficient": coeff_value == 0.0,
        "allow_empty_coefficient": bool(cfg.allow_empty_coefficient),
        "plan_has_nonzero_deltas": None,
        "status": "not_applicable",
    }
    if coeff_value != 0.0:
        return state
    nonzero = first_nonzero_delta(cfg, plan)
    state["plan_has_nonzero_deltas"] = nonzero is not None
    if nonzero is None:
        # Les vecteurs sont eux-memes nuls : le no-op vient des donnees, pas
        # du coefficient. Aucun refus (l'avertissement reste inscrit).
        state["status"] = "noop_data"
        return state
    if cfg.allow_empty_coefficient:
        state["status"] = "bypassed"
        return state
    state["status"] = "enforced"
    reference = "D_b" if cfg.mode == "transfer" else "D_p"
    raise MergeRefused(
        f"coefficient nul refuse : le mode '{cfg.mode}' avec {coeff_name}=0.0 "
        f"ecrit une simple COPIE de {reference}, alors que le plan de fusion "
        f"contient des vecteurs non nuls (ex. {nonzero}). Passer un coefficient "
        f"explicite (--alpha / --lambda), ou assumer explicitement le no-op "
        f"avec --allow-empty-coefficient (avertissement inscrit au manifeste).")


# --------------------------------------------------------------------------- #
# Normes et produits scalaires (float64)
# --------------------------------------------------------------------------- #

def _sq_norm(tensor: torch.Tensor) -> float:
    """Norme au carre, accumulatee en float64.

    `torch.norm` en float32 se trompe de ~0,75 % sur un tenseur de 68 M elements
    (mesure) : les normes de famille du manifeste et les bornes de tolerance de
    la verification post-ecriture meritent l'exactitude float64.
    """
    return float(torch.linalg.vector_norm(tensor, dtype=torch.float64).item()) ** 2


def _dot(a: torch.Tensor, b: torch.Tensor) -> float:
    """Produit scalaire, accumule en float64."""
    return float((a * b).sum(dtype=torch.float64).item())


# --------------------------------------------------------------------------- #
# Politiques
# --------------------------------------------------------------------------- #

def resolve_policy(cfg: MergeConfig) -> MergePolicy:
    """Politique de fusion effective (couverture + alignement)."""
    coverage = cfg.coverage if cfg.coverage is not None else "strict"
    if coverage not in COVERAGE_POLICIES:
        raise ValueError(f"politique de couverture inconnue : {coverage!r} "
                         f"(attendu : {list(COVERAGE_POLICIES)})")
    return MergePolicy(coverage=coverage,
                       embedding_policy=cfg.mapping.embedding_policy,
                       on_shape_mismatch=cfg.on_shape_mismatch)


def _requires_explicit_coverage(cfg: MergeConfig) -> bool:
    """Vaut-il choisir explicitement la politique de couverture ?

    Oui des que le mode combine deux sources dont l'une n'est pas une paire
    de checkpoints complets (adaptateur LoRA, vecteur legacy) : l'union des
    modules n'est alors plus garantie et la politique doit etre dite.
    """
    used = [s for s in (cfg.tau_a, cfg.tau_d) if s is not None]
    if len(used) <= 1:
        return False
    return not all(getattr(s, "kind", "") == "checkpoint-pair" for s in used)


def validate_config(cfg: MergeConfig) -> List[str]:
    """Valide la configuration ; retourne la liste des avertissements."""
    if cfg.mode not in MODES:
        raise ValueError(f"mode inconnu : {cfg.mode!r} (attendu : {MODES})")
    warnings: List[str] = []
    if cfg.alpha < 0 or cfg.lam < 0:
        raise ValueError("alpha et lambda doivent etre >= 0 (le papier n'impose "
                         "pas de plafond pour alpha ; cf. Table 6 : alpha jusqu'a 1.4)")
    if cfg.alpha > COEFFICIENT_WARNING_THRESHOLD:
        warnings.append(
            f"alpha={cfg.alpha} > {COEFFICIENT_WARNING_THRESHOLD} : au-dela de cette "
            f"valeur le merge degrade generalement le modele (cf. A2D Fig. 8)")
    if cfg.lam > LAMBDA_MAX:
        warnings.append(
            f"lambda={cfg.lam} > {LAMBDA_MAX} : l'eq. (2) ne borne lambda que dans "
            f"[0,1] ; hors de cet intervalle le role de D_b et de D_p s'inverse")
    if cfg.allow_empty_coefficient:
        coeff_name, coeff_value = effective_coefficient(cfg.mode, cfg.alpha, cfg.lam)
        if coeff_value == 0.0:
            warnings.append(
                f"--allow-empty-coefficient : {coeff_name}=0.0 explicitement autorise ; "
                f"la fusion n'applique aucun vecteur et ecrit une copie de la reference")
        else:
            warnings.append(
                f"--allow-empty-coefficient : option sans effet ({coeff_name}="
                f"{coeff_value} != 0)")
    if cfg.mode in ("transfer", "ablation-dp-plus-tau-a", "merge") and cfg.tau_a is None:
        raise MissingVectorError("tau_A absent : ce mode exige un vecteur AR")
    if cfg.mode == "merge" and cfg.tau_d is None:
        raise MissingVectorError(
            "tau_D absent : l'A2D-Merge (D_p + lambda*(tau_A - tau_D)) est "
            "impossible sans le vecteur de post-training diffusion ; utiliser le "
            "mode transfer (D_b + alpha*tau_A)")
    if cfg.mode == "ablation-dp-minus-tau-d" and cfg.tau_d is None:
        raise MissingVectorError("tau_D absent : ce mode exige un vecteur diffusion")
    if cfg.coverage is not None and cfg.coverage not in COVERAGE_POLICIES:
        raise ValueError(f"politique de couverture inconnue : {cfg.coverage!r} "
                         f"(attendu : {list(COVERAGE_POLICIES)})")
    if cfg.coverage is None and _requires_explicit_coverage(cfg):
        kinds = ", ".join(getattr(s, "kind", "?") for s in (cfg.tau_a, cfg.tau_d)
                           if s is not None)
        raise MergeError(
            "politique de couverture non declaree : le mode combine des sources "
            f"({kinds}) dont l'une n'est pas une paire de checkpoints complets ; "
            "l'union de leurs modules n'est donc pas garantie. Choisir explicitement "
            "--coverage strict (refus d'une couverture inegale) ou --coverage "
            "zero-fill (vecteur absent traite comme 0, declare au manifeste)")
    if cfg.coverage == "zero-fill":
        warnings.append(
            "couverture zero-fill : les modules absents d'une source recoivent un "
            "vecteur nul ; la liste exacte est declaree dans merge_config.json")
    return warnings


# --------------------------------------------------------------------------- #
# Plan de fusion
# --------------------------------------------------------------------------- #

def _new_stats() -> dict:
    return {"n_a": 0.0, "n_d": 0.0, "dot": 0.0, "w": 0.0, "resid": 0.0, "exact": 0.0,
            "num": 0.0, "n_tensors": 0, "n_a_tensors": 0, "n_d_tensors": 0}


def _new_acc() -> dict:
    """Accumulateur de la verification v2 (Gram float64)."""
    return {"aa": 0.0, "ad": 0.0, "dd": 0.0, "ba": 0.0, "bd": 0.0,
            "resid": 0.0, "n": 0, "w": 0.0}


def _module_candidates(cfg: MergeConfig, recipient_modules: Dict[str, str],
                       counters: dict) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Modules cibles couverts par tau_A et par tau_D.

    * tau_A vit dans l'espace AR : on applique le mapping (`mapping.target`).
    * tau_D vit dans l'espace du receveur (D_p - D_b) : le nom de module EST
      le nom du tenseur cible.

    Retourne (cand_a, cand_d) : {nom_tenseur_cible: nom_module_source}.
    """
    cand_a: Dict[str, str] = {}
    cand_d: Dict[str, str] = {}

    modules_a = cfg.tau_a.modules() if cfg.tau_a is not None else None
    if modules_a:
        for source_module in sorted(modules_a):
            target_module = cfg.mapping.target(source_module)
            if target_module is None:
                counters["no_counterpart"] += 1
                continue
            if target_module not in recipient_modules:
                counters["phantom_target"] += 1
                raise MergeError(
                    f"nom cible fantome : le mapping '{cfg.mapping.id}' transforme "
                    f"'{source_module}' en '{target_module}', absent du checkpoint "
                    f"receveur. Modeles de familles incompatibles ?")
            family = family_of(target_module)
            if not cfg.mapping.is_fusable(family):
                counters["not_fusable"] += 1
                continue
            cand_a[recipient_modules[target_module]] = source_module

    modules_d = cfg.tau_d.modules() if cfg.tau_d is not None else None
    if modules_d:
        for module in sorted(modules_d):
            if module not in recipient_modules:
                # un adaptateur peut viser un module que le receveur n'a pas
                counters["tau_d_absent_from_recipient"] += 1
                continue
            family = family_of(module)
            if not cfg.mapping.is_fusable(family):
                counters["not_fusable"] += 1
                continue
            cand_d[recipient_modules[module]] = module

    return cand_a, cand_d


def build_plan(weight_map: Dict[str, str], cfg: MergeConfig) -> Tuple[Dict[str, PlanEntry], Dict[str, int]]:
    """Plan de fusion sur l'UNION des modules de tau_A et de tau_D.

    Retourne (plan, compteurs) :
      * plan : {nom_tenseur_cible: PlanEntry(family, source_a, source_d, coverage)}
      * compteurs : selection et classes de couverture.

    Classes de couverture (par module de l'union) :
      * `both`    : couvert par tau_A et par tau_D ;
      * `A-only` : couvert par tau_A seulement ;
      * `D-only` : couvert par tau_D seulement.

    La politique `strict` refuse un plan dont la couverture est inegale pour
    les sources exigees par le mode ; `zero-fill` traite le vecteur absent
    comme nul (et le declare au manifeste).
    """
    plan: Dict[str, PlanEntry] = {}
    counters = {"no_counterpart": 0, "not_fusable": 0, "no_tau": 0, "phantom_target": 0,
                "tau_d_absent_from_recipient": 0, "both": 0, "A-only": 0, "D-only": 0,
                "zero_filled": 0}
    recipient_modules = {module_of(t): t for t in weight_map}
    required = required_sources(cfg.mode)
    policy = resolve_policy(cfg)

    modules_a = cfg.tau_a.modules() if cfg.tau_a is not None else None
    modules_d = cfg.tau_d.modules() if cfg.tau_d is not None else None

    if modules_a is None and modules_d is None:
        # Source non enumerable (`.npy` legacy) : on part des tenseurs du
        # receveur et on cherche l'equivalent source via le mapping inverse.
        for tensor_name in weight_map:
            module = module_of(tensor_name)
            source_module = cfg.mapping.source(module)
            if source_module is None:
                counters["no_counterpart"] += 1
                continue
            family = family_of(module)
            if not cfg.mapping.is_fusable(family):
                counters["not_fusable"] += 1
                continue
            if cfg.tau_a is None or not cfg.tau_a.has(source_module):
                counters["no_tau"] += 1
                continue
            plan[tensor_name] = PlanEntry(family=family, source_a=source_module,
                                          source_d=None, coverage="A-only")
        counters["A-only"] = len(plan)
        return plan, counters

    cand_a, cand_d = _module_candidates(cfg, recipient_modules, counters)

    # classes de couverture sur l'union
    union = set(cand_a) | set(cand_d)
    classes: Dict[str, str] = {}
    for tensor_name in union:
        has_a = tensor_name in cand_a
        has_d = tensor_name in cand_d
        if has_a and has_d:
            classes[tensor_name] = "both"
        elif has_a:
            classes[tensor_name] = "A-only"
        else:
            classes[tensor_name] = "D-only"
        counters[classes[tensor_name]] += 1

    if required == frozenset({"A", "D"}):
        missing_a = sorted(t for t, c in classes.items() if c == "D-only")
        missing_d = sorted(t for t, c in classes.items() if c == "A-only")
        if (missing_a or missing_d) and policy.coverage == "strict":
            raise CoverageError(
                "couverture inegale entre tau_A et tau_D (politique 'strict') : "
                f"{len(missing_d)} module(s) couverts par tau_A mais pas par tau_D "
                f"{_sample(missing_d)} ; {len(missing_a)} module(s) couverts par "
                f"tau_D mais pas par tau_A {_sample(missing_a)}. "
                "Utiliser --coverage zero-fill pour traiter les vecteurs absents "
                "comme nuls (declaration au manifeste).")
        for tensor_name in union:
            coverage = classes[tensor_name]
            if coverage == "A-only":
                counters["zero_filled"] += 1
            elif coverage == "D-only":
                counters["zero_filled"] += 1
            plan[tensor_name] = PlanEntry(
                family=family_of(module_of(tensor_name)),
                source_a=cand_a.get(tensor_name),
                source_d=cand_d.get(tensor_name),
                coverage=coverage)
        return plan, counters

    # mode a une seule source : on ne fusionne que les modules de cette source
    source = cand_a if required == frozenset({"A"}) else cand_d
    for tensor_name, source_module in source.items():
        plan[tensor_name] = PlanEntry(
            family=family_of(module_of(tensor_name)),
            source_a=cand_a.get(tensor_name),
            source_d=cand_d.get(tensor_name),
            coverage=classes.get(tensor_name, "A-only" if required == frozenset({"A"})
                                  else "D-only"))
    return plan, counters


def _sample(items: List[str], limit: int = 20) -> str:
    """Extrait lisible d'une liste de modules (les 20 premiers + le compte)."""
    if not items:
        return ""
    head = ", ".join(items[:limit])
    if len(items) > limit:
        return f"({head}, ... +{len(items) - limit} autres)"
    return f"({head})"


# --------------------------------------------------------------------------- #
# Valeur attendue (FONCTION UNIQUE : writer ET verify_merge)
# --------------------------------------------------------------------------- #

def _align_delta(delta: Optional[torch.Tensor], weight: torch.Tensor, tensor_name: str,
                 family: str, policy: MergePolicy) -> Tuple[Optional[torch.Tensor], bool, Optional[str]]:
    """Aligne un vecteur sur la forme du tenseur receveur (embeddings : lignes).

    Retourne (delta_aligne, skipped, note).
    * `skipped` = True : forme incompatible et politique `skip` (le tenseur
      reste inchange) ;
    * `note` : description de l'alignement (embeddings), a journaliser.
    """
    if delta is None:
        return None, False, None
    if delta.shape == weight.shape:
        return delta, False, None
    if (family == FAMILY_EMBED and policy.embedding_policy == "rows"
            and delta.dim() == 2 and weight.dim() == 2
            and delta.shape[1] == weight.shape[1]):
        rows = min(delta.shape[0], weight.shape[0])
        aligned = torch.zeros_like(weight)
        aligned[:rows] = delta[:rows]
        note = (f"{tensor_name}: embedding {tuple(delta.shape)} -> "
                f"{tuple(weight.shape)} ; tau applique sur les {rows} premieres "
                f"lignes, tau=0 sur les {weight.shape[0] - rows} lignes restantes")
        return aligned, False, note
    message = (f"{tensor_name}: forme tau={tuple(delta.shape)} incompatible avec "
               f"le receveur {tuple(weight.shape)}")
    if policy.on_shape_mismatch == "error":
        raise ShapeMismatchError(message)
    return None, True, f"SHAPE WARNING: {message} — tenseur laisse inchange"


def expected_tensor(ref: torch.Tensor, tau_a: Optional[torch.Tensor],
                    tau_d: Optional[torch.Tensor], c_a: float, c_d: float,
                    family: str, policy: MergePolicy,
                    tensor_name: str = "") -> Expected:
    """Valeur attendue d'un tenseur fusionne (FP32) et deltas alignes.

    C'est la SEULE definition de la valeur attendue du dépôt : elle est
    utilisee par le writer (`merge_checkpoint`) ET par la verification
    (`verify_checkpoint` / `verify_merge`). L'alignement des embeddings
    (intersection des lignes) est fait ici, donc les deux chemins ne peuvent
    plus diverger.

    * `ref` : tenseur receveur (FP32) ;
    * `tau_a`, `tau_d` : deltas bruts (FP32) ou None (source absente) ;
    * `c_a`, `c_d` : coefficients appliques (signes) ;
    * `family` : famille du tenseur ;
    * `policy` : politiques de couverture et d'alignement.
    """
    delta_a, skip_a, note_a = _align_delta(tau_a, ref, tensor_name, family, policy)
    delta_d, skip_d, note_d = _align_delta(tau_d, ref, tensor_name, family, policy)
    note = "; ".join(n for n in (note_a, note_d) if n)
    if skip_a or skip_d:
        return Expected(value=ref.clone(), delta_a=None, delta_d=None,
                        skipped=True, note=note or None)
    value = ref
    if delta_a is not None:
        value = value + c_a * delta_a
    if delta_d is not None:
        value = value + c_d * delta_d
    return Expected(value=value, delta_a=delta_a, delta_d=delta_d,
                    skipped=False, note=note or None)


# --------------------------------------------------------------------------- #
# Verification v2
# --------------------------------------------------------------------------- #

def _ulp(dtype: torch.dtype, tensor: torch.Tensor) -> torch.Tensor:
    """Pas unitaire du dtype cible, element par element."""
    finfo = torch.finfo(dtype)
    magnitude = tensor.abs().clamp_min(finfo.tiny)
    exponent = torch.floor(torch.log2(magnitude))
    return torch.pow(2.0, exponent + math.log2(finfo.eps))


def _solve(acc: dict, has_a: bool, has_d: bool) -> dict:
    """Coefficients signes + bornes de tolerance d'un accumulateur.

    Regression des moindres carres de y = out - ref sur [x_a, x_d] :
        c_eff = (X^T X)^-1 X^T y
    L'ecart est borne par la matrice de la moindres carres appliquee au
    residu BF16 : |c_eff - c| <= ||M e_j|| * ||eps_bf16||.

    Avec un seul regresseur (mode a une seule source), c'est une projection
    simple. Si les deux regresseurs sont presque colineaires (mal conditionne),
    on replie sur des projections simples et on le signale.
    """
    aa, ad, dd = acc["aa"], acc["ad"], acc["dd"]
    ba, bd = acc["ba"], acc["bd"]
    resid = math.sqrt(max(acc["resid"], 0.0))
    out: dict = {"ill_conditioned": False, "cos": None}

    if not has_a and not has_d:
        return {"c_a": None, "c_d": None, "bound_a": None, "bound_d": None,
                "ill_conditioned": True, "cos": None}

    if has_a and not has_d:
        if aa <= 0:
            return {"c_a": None, "c_d": None, "bound_a": None, "bound_d": None,
                    "ill_conditioned": True, "cos": None}
        out["c_a"] = ba / aa
        out["c_d"] = None
        out["bound_a"] = resid / math.sqrt(aa)
        out["bound_d"] = None
        return out

    if has_d and not has_a:
        if dd <= 0:
            return {"c_a": None, "c_d": None, "bound_a": None, "bound_d": None,
                    "ill_conditioned": True, "cos": None}
        out["c_a"] = None
        out["c_d"] = bd / dd
        out["bound_a"] = None
        out["bound_d"] = resid / math.sqrt(dd)
        return out

    if aa <= 0 or dd <= 0:
        return {"c_a": None, "c_d": None, "bound_a": None, "bound_d": None,
                "ill_conditioned": True, "cos": None}
    cos = ad / math.sqrt(aa * dd)
    out["cos"] = cos
    det = aa * dd - ad * ad
    ill = (det <= 0.0) or (abs(cos) >= ILL_CONDITIONED_COS)
    if ill:
        # repli sur des projections simples (independantes)
        out["ill_conditioned"] = True
        out["c_a"] = ba / aa
        out["c_d"] = bd / dd
        out["bound_a"] = resid / math.sqrt(aa)
        out["bound_d"] = resid / math.sqrt(dd)
        return out

    inv = 1.0 / det
    g00, g01, g11 = dd * inv, -ad * inv, aa * inv
    out["c_a"] = g00 * ba + g01 * bd
    out["c_d"] = g01 * ba + g11 * bd
    # ||ligne j de M||^2 = c^T G c avec c = (G^-1)[j, :]
    row0_sq = g00 * g00 * aa + 2 * g00 * g01 * ad + g01 * g01 * dd
    row1_sq = g01 * g01 * aa + 2 * g01 * g11 * ad + g11 * g11 * dd
    out["bound_a"] = resid * math.sqrt(max(row0_sq, 0.0))
    out["bound_d"] = resid * math.sqrt(max(row1_sq, 0.0))
    return out


def _acc_add(acc: dict, delta_a: Optional[torch.Tensor], delta_d: Optional[torch.Tensor],
             y: torch.Tensor, expected: torch.Tensor, written: torch.Tensor,
             weight: torch.Tensor, dtype: torch.dtype) -> float:
    """Ajoute un tenseur a l'accumulateur ; retourne l'ecart en ULP."""
    if delta_a is not None:
        acc["aa"] += _sq_norm(delta_a)
        acc["ba"] += _dot(delta_a, y)
    if delta_d is not None:
        acc["dd"] += _sq_norm(delta_d)
        acc["bd"] += _dot(delta_d, y)
        if delta_a is not None:
            acc["ad"] += _dot(delta_a, delta_d)
    acc["n"] += 1
    acc["w"] += _sq_norm(weight)
    # residu d'arrondi BF16 : round(expected) - expected (independant de l'ecart
    # reel du tenseur ecrit : la tolerance ne peut donc pas absorber une
    # corruption).
    rounded = expected.to(dtype).to(torch.float32)
    acc["resid"] += _sq_norm(rounded - expected)
    # ecart du tenseur ecrit a round(expected), en ULP du dtype cible
    dist = (written - rounded).abs()
    scale = _ulp(dtype, rounded)
    ratio = dist / scale
    return float(ratio.max().item()) if ratio.numel() else 0.0


def verify_checkpoint(cfg: MergeConfig, merged_dir, reference_dir,
                      plan: Optional[Dict[str, PlanEntry]] = None,
                      log=print) -> dict:
    """Verification v2 d'un checkpoint fusionne (lecture seule).

    Quatre controles, tous appliquees au checkpoint ecrit :
      1. regression des moindres carres SIGNEE de (out - ref) sur [tau_A, tau_D]
         (Gram float64), par famille et globale ; les coefficients signes
         obtenus sont compares a ceux attendus du mode ;
      2. les tenseurs NON fusionnes sont bit-identiques a la reference ;
      3. l'ecart maximal a `round_bf16(expected)` recalcule independamment est
         <= 1 ULP du dtype cible ;
      4. tolerance derivee du residu BF16 mesure (et non de l'ecart reel).

    Utilisee par le writer (verification post-ecriture) ET par
    `scripts/verify_merge.py` (verification post hoc) : une seule implementation.
    """
    merged = Path(merged_dir)
    reference = Path(reference_dir)
    weight_map = load_weight_map(merged)
    if plan is None:
        plan, counters = build_plan(weight_map, cfg)
    else:
        counters = {}
    policy = resolve_policy(cfg)
    c_a, c_d = mode_coefficients(cfg.mode, cfg.alpha, cfg.lam)
    coeff_name, coeff_value = effective_coefficient(cfg.mode, cfg.alpha, cfg.lam)
    checked = checked_coefficients(cfg.mode)
    has_d = any(entry.source_d is not None for entry in plan.values())
    has_a = any(entry.source_a is not None for entry in plan.values())

    by_file: Dict[str, List[str]] = defaultdict(list)
    for tensor_name, fname in weight_map.items():
        by_file[fname].append(tensor_name)

    acc_family: Dict[str, dict] = defaultdict(_new_acc)
    acc_global = _new_acc()
    failures: List[str] = []
    n_fused = 0
    n_unchanged = 0
    n_unchanged_identical = 0
    max_ulp = 0.0
    max_ulp_tensor = None
    families_seen: List[str] = []

    for fname, tensor_names in sorted(by_file.items()):
        with safe_open(str(reference / fname), framework="pt") as ref_handle, \
                safe_open(str(merged / fname), framework="pt") as out_handle:
            for tensor_name in sorted(tensor_names):
                ref_raw = ref_handle.get_tensor(tensor_name)
                out_raw = out_handle.get_tensor(tensor_name)
                entry = plan.get(tensor_name)
                if entry is None:
                    n_unchanged += 1
                    if _bit_identical(ref_raw, out_raw):
                        n_unchanged_identical += 1
                    else:
                        failures.append(
                            f"{tensor_name}: tenseur NON fusionne modifie (identite "
                            "binaire rompue avec la reference)")
                    continue
                n_fused += 1
                ref_t = ref_raw.to(torch.float32)
                out_t = out_raw.to(torch.float32)
                raw_a = cfg.tau_a.delta(entry.source_a) if entry.source_a is not None else None
                raw_d = cfg.tau_d.delta(entry.source_d) if entry.source_d is not None else None
                exp = expected_tensor(ref_t, raw_a, raw_d, c_a, c_d, entry.family,
                                      policy, tensor_name)
                if exp.skipped:
                    failures.append(
                        f"{tensor_name}: forme incompatible au moment de la verification "
                        f"({exp.note})")
                    continue
                y = out_t - ref_t
                ulp = _acc_add(acc_family[entry.family], exp.delta_a, exp.delta_d, y,
                                exp.value, out_t, ref_t, cfg.dtype)
                _acc_add(acc_global, exp.delta_a, exp.delta_d, y, exp.value, out_t,
                         ref_t, cfg.dtype)
                if ulp > max_ulp:
                    max_ulp = ulp
                    max_ulp_tensor = tensor_name
                if entry.family not in families_seen:
                    families_seen.append(entry.family)

    # ---- coefficients par famille et globaux --------------------------------
    families: Dict[str, dict] = {}
    for family, acc in sorted(acc_family.items()):
        # Les regresseurs disponibles sont ceux du plan de CETTE famille :
        # une famille couverte par une seule source (cas `zero-fill`) doit etre
        # verifiee par une projection simple, pas declaree non estimable.
        fam_has_a = acc["aa"] > 0
        fam_has_d = acc["dd"] > 0
        report = _report_family(family, acc, fam_has_a, fam_has_d, checked, c_a, c_d,
                                 coeff_name, coeff_value, cfg)
        families[family] = report
        failures.extend(report["failures"])
    report_all = _report_family("__all__", acc_global, has_a, has_d, checked, c_a, c_d,
                                 coeff_name, coeff_value, cfg)
    families["__all__"] = report_all
    failures.extend(report_all["failures"])

    if max_ulp > cfg.max_ulp:
        failures.append(
            f"ecart maximal a round_bf16(expected) = {max_ulp:.3f} ULP "
            f"(sur {max_ulp_tensor or '?'}) > {cfg.max_ulp} ULP")

    if n_fused == 0:
        failures.append("aucun tenseur fusionne detecte : rien a verifier")

    log(f"  Verification v2 : {coeff_name}Eff signe "
        f"(a={_fmt(report_all['c_a_eff'])}, d={_fmt(report_all['c_d_eff'])}) "
        f"(demande a={c_a}, d={c_d}) — {'OK' if not failures else 'ECHEC'}")
    return {
        "ok": not failures,
        "version": 2,
        "coefficient": coeff_name,
        "requested": coeff_value,
        "requested_coefficients": {"a": c_a, "d": c_d},
        "effective_global": report_all["c_a_eff"] if coeff_name == "alpha" else report_all["c_d_eff"],
        "c_a_eff": report_all["c_a_eff"],
        "c_d_eff": report_all["c_d_eff"],
        "families": families,
        "n_fused": n_fused,
        "n_unchanged": n_unchanged,
        "n_unchanged_bit_identical": n_unchanged_identical,
        "max_ulp_deviation": max_ulp,
        "max_ulp_tensor": max_ulp_tensor,
        "plan_counters": counters,
        "failures": failures,
    }


def _fmt(value) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _report_family(label: str, acc: dict, has_a: bool, has_d: bool, checked: Tuple[str, ...],
                   c_a: float, c_d: float, coeff_name: str, coeff_value: float,
                   cfg: MergeConfig) -> dict:
    """Rapport de verification pour une famille (ou le global)."""
    solved = _solve(acc, has_a, has_d)
    failures: List[str] = []
    tol = cfg.tol_abs
    c_a_eff = solved["c_a"]
    c_d_eff = solved["c_d"]
    bound_a = solved["bound_a"]
    bound_d = solved["bound_d"]

    dev_a = dev_d = None
    if "a" in checked and c_a_eff is not None:
        dev_a = abs(c_a_eff - c_a)
        limit_a = tol + (bound_a or 0.0)
        if dev_a > limit_a:
            failures.append(
                f"{label}: c_a_eff={c_a_eff:.4f} (demande {c_a}) ecart {dev_a:.4f} > "
                f"tol {tol} + borne BF16 {(bound_a or 0.0):.4f}")
    if "d" in checked and c_d_eff is not None:
        dev_d = abs(c_d_eff - c_d)
        limit_d = tol + (bound_d or 0.0)
        if dev_d > limit_d:
            failures.append(
                f"{label}: c_d_eff={c_d_eff:.4f} (demande {c_d}) ecart {dev_d:.4f} > "
                f"tol {tol} + borne BF16 {(bound_d or 0.0):.4f}")

    return {
        "effective_coefficient": c_a_eff if coeff_name == "alpha" else c_d_eff,
        "c_a_eff": c_a_eff,
        "c_d_eff": c_d_eff,
        "requested": coeff_value,
        "requested_coefficients": {"a": c_a, "d": c_d},
        "deviation": dev_a if "a" in checked else dev_d,
        "deviation_a": dev_a,
        "deviation_d": dev_d,
        "bf16_bound": bound_a if "a" in checked else bound_d,
        "tolerance": tol,
        "cos_tau_A_tau_D": solved["cos"],
        "ill_conditioned": solved["ill_conditioned"],
        "n_tensors": acc["n"],
        "norm_W": math.sqrt(acc["w"]) if acc["w"] > 0 else None,
        "ok": not failures,
        "failures": failures,
    }


def _bit_identical(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Identite binaire stricte de deux tenseurs."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.numel() == 0:
        return True
    if a.dtype in (torch.bfloat16, torch.float16, torch.int16):
        ia = a.view(torch.int16)
        ib = b.view(torch.int16)
    elif a.dtype == torch.float32:
        ia = a.view(torch.int32)
        ib = b.view(torch.int32)
    else:
        return bool(torch.equal(a, b))
    return bool(torch.equal(ia, ib))


# --------------------------------------------------------------------------- #
# Ecriture
# --------------------------------------------------------------------------- #

def merge_checkpoint(cfg: MergeConfig, log=print) -> MergeResult:
    """Execute la fusion et renvoie le manifeste + la verification."""
    warnings = validate_config(cfg)
    for warning in warnings:
        log(f"ATTENTION: {warning}")
    policy = resolve_policy(cfg)

    recipient = Path(cfg.recipient)
    output = Path(cfg.output)
    weight_map = load_weight_map(recipient)

    output.mkdir(parents=True, exist_ok=True)
    existing = [f for f in os.listdir(output) if f.endswith(".safetensors")]
    if existing and not cfg.overwrite:
        raise MergeRefused(
            f"{output} contient deja {len(existing)} shards. Relancer avec "
            f"--overwrite (nettoyage des shards obsoletes) ou vers un dossier "
            f"vide. Pas de merge partiel.")
    for stale in ("model.safetensors.index.json", "merge_config.json"):
        stale_path = output / stale
        if stale_path.exists():
            stale_path.unlink()

    plan, counters = build_plan(weight_map, cfg)
    c_a, c_d = mode_coefficients(cfg.mode, cfg.alpha, cfg.lam)
    coeff_name, coeff_value = effective_coefficient(cfg.mode, cfg.alpha, cfg.lam)
    log(f"Plan de fusion : {len(plan)} tenseurs sur {len(weight_map)} "
        f"(mode={cfg.mode}, {coeff_name}={coeff_value}, couverture={policy.coverage})")
    if not plan:
        if not cfg.allow_empty:
            raise MergeError(
                "plan de fusion VIDE : aucun module n'est couvert par les sources "
                "requises. Refus d'ecrire un checkpoint identique a la reference "
                "(utiliser --allow-empty pour l'assumer explicitement).")
        log("ATTENTION: plan de fusion vide (--allow-empty) : le checkpoint sera "
            "une copie de la reference.")

    # Garde-fou « coefficient nul » : refuse un transfert/merge a coefficient
    # nul (ecriture d'une simple copie de la reference) sauf opt-in explicite.
    guard = evaluate_coefficient_guard(cfg, plan)

    notes: List[str] = []
    stats: Dict[str, dict] = defaultdict(_new_stats)
    ratios: Dict[str, List[float]] = defaultdict(list)

    by_file: Dict[str, List[str]] = defaultdict(list)
    for tensor_name, fname in weight_map.items():
        by_file[fname].append(tensor_name)

    total_written = 0
    for fname, tensor_names in sorted(by_file.items()):
        out_tensors = {}
        with safe_open(str(recipient / fname), framework="pt") as handle:
            for tensor_name in sorted(tensor_names):
                weight = handle.get_tensor(tensor_name).to(torch.float32)
                entry = plan.get(tensor_name)
                if entry is None:
                    out_tensors[tensor_name] = weight.to(cfg.dtype)
                    continue
                raw_a = cfg.tau_a.delta(entry.source_a) if entry.source_a is not None else None
                raw_d = cfg.tau_d.delta(entry.source_d) if entry.source_d is not None else None
                exp = expected_tensor(weight, raw_a, raw_d, c_a, c_d, entry.family,
                                      policy, tensor_name)
                if exp.note:
                    notes.append(exp.note)
                if exp.skipped:
                    # forme incompatible et --on-shape-mismatch=skip : tenseur inchange
                    plan.pop(tensor_name)
                    out_tensors[tensor_name] = weight.to(cfg.dtype)
                    continue
                casted = exp.value.to(cfg.dtype)
                # UN SEUL cast : on mesure l'ecart BF16 pour la verification
                residual = _sq_norm(exp.value - casted.to(torch.float32))
                out_tensors[tensor_name] = casted

                st = stats[entry.family]
                st["n_tensors"] += 1
                st["w"] += _sq_norm(weight)
                st["resid"] += residual
                if exp.delta_a is not None:
                    st["n_a"] += _sq_norm(exp.delta_a)
                    st["n_a_tensors"] += 1
                if exp.delta_d is not None:
                    st["n_d"] += _sq_norm(exp.delta_d)
                    st["n_d_tensors"] += 1
                    if exp.delta_a is not None:
                        st["dot"] += _dot(exp.delta_a, exp.delta_d)
                applied = torch.zeros_like(weight)
                if exp.delta_a is not None:
                    applied = applied + c_a * exp.delta_a
                if exp.delta_d is not None:
                    applied = applied + c_d * exp.delta_d
                w_norm = math.sqrt(_sq_norm(weight))
                if w_norm > 0:
                    ratios[entry.family].append(math.sqrt(_sq_norm(applied)) / w_norm)
        _atomic_save(out_tensors, output / fname)
        total_written += len(tensor_names)
        log(f"    [{fname}] {len(tensor_names)} tenseurs — total {total_written}/{len(weight_map)}")

    # Nettoyage des shards obsoletes
    for entry in os.listdir(output):
        if entry.endswith(".safetensors") and entry not in by_file:
            (output / entry).unlink()
            log(f"  Nettoyage shard obsolete : {entry}")

    total_size = sum((output / fname).stat().st_size for fname in by_file)
    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    _atomic_write_json(output / "model.safetensors.index.json", index)

    if cfg.copy_sidecars:
        _copy_sidecars(recipient, output, log)

    manifest = _build_manifest(cfg, plan, counters, stats, ratios, notes, warnings,
                                weight_map, policy, guard)

    verification: Dict[str, object] = {}
    if cfg.verify:
        verification = verify_checkpoint(cfg, output, recipient, plan=plan, log=log)
        manifest["post_write_verification"] = verification
        if not verification.get("ok", False):
            _atomic_write_json(output / "merge_config.json", manifest)
            raise MergeError(
                "verification post-ecriture FAILED : "
                + json.dumps(verification.get("failures", []), ensure_ascii=False))

    _atomic_write_json(output / "merge_config.json", manifest)
    return MergeResult(manifest=manifest, verification=verification,
                       n_fused=len(plan), n_unchanged=len(weight_map) - len(plan))


def _atomic_save(tensors: Dict[str, torch.Tensor], path: Path) -> None:
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".tmp-")
    os.close(fd)
    try:
        save_file(tensors, tmp_path)
        os.replace(tmp_path, str(path))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _atomic_write_json(path: Path, payload: dict) -> None:
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".tmp-")
    os.close(fd)
    try:
        with open(tmp_path, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, str(path))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _copy_sidecars(recipient: Path, output: Path, log) -> None:
    import sys
    scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from _common import list_source_sidecar_files
    for entry in list_source_sidecar_files(str(recipient)):
        src = recipient / entry
        dst = output / entry
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
    readme_src = recipient / "README.md"
    readme_dst = output / "README.md"
    if readme_dst.exists():
        log("  README.md de destination conserve (non ecrase).")
    elif readme_src.exists():
        shutil.copy2(readme_src, readme_dst)
        log("  README.md source copie (destination sans model card).")


def _family_summary(stats: Dict[str, dict], ratios: Dict[str, List[float]]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for family, st in sorted(stats.items()):
        n_a = math.sqrt(st["n_a"])
        n_d = math.sqrt(st["n_d"])
        w = math.sqrt(st["w"])
        cos = None
        if n_a > 0 and n_d > 0:
            cos = st["dot"] / (n_a * n_d)
        med = None
        values = ratios.get(family) or []
        if values:
            med = sorted(values)[len(values) // 2]
        out[family] = {
            "n_tensors": st["n_tensors"],
            "n_tensors_with_tau_A": st["n_a_tensors"],
            "n_tensors_with_tau_D": st["n_d_tensors"],
            "norm_tau_A": n_a,
            "norm_tau_D": n_d,
            "cos_tau_A_tau_D": cos,
            "norm_W": w,
            "ratio_tau_over_W_global": (math.sqrt(max(st["n_a"], st["n_d"])) / w) if w > 0 else None,
            "ratio_tau_over_W_median": med,
        }
    return out


def _coverage_summary(plan: Dict[str, PlanEntry], counters: Dict[str, int],
                      policy: MergePolicy, cfg: MergeConfig) -> dict:
    """Classes de couverture (both / A-only / D-only) par famille."""
    per_family: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"both": 0, "A-only": 0, "D-only": 0})
    zero_filled: List[str] = []
    for tensor_name, entry in plan.items():
        per_family[entry.family][entry.coverage] += 1
        if entry.coverage in ("A-only", "D-only"):
            zero_filled.append(tensor_name)
    return {
        "policy": policy.coverage,
        "explicit": cfg.coverage is not None,
        "counters": dict(counters),
        "per_family": {k: dict(v) for k, v in sorted(per_family.items())},
        "n_both": counters.get("both", 0),
        "n_A_only": counters.get("A-only", 0),
        "n_D_only": counters.get("D-only", 0),
        "n_zero_filled": counters.get("zero_filled", 0),
        "zero_filled_modules": sorted(zero_filled)[:200],
        "zero_filled_modules_truncated": len(zero_filled) > 200,
    }


def _build_manifest(cfg, plan, counters, stats, ratios, notes, warnings, weight_map,
                    policy: MergePolicy, guard: Optional[dict] = None) -> dict:
    import sys
    scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from _common import git_commit
    import safetensors
    coeff_name, coeff_value = effective_coefficient(cfg.mode, cfg.alpha, cfg.lam)
    manifest = {
        "method": "A2D (arXiv:2610.08108) — " + {
            "transfer": "A2D-Transfer : D_b + alpha*tau_A (eq. 1)",
            "merge": "A2D-Merge : D_p + lambda*(tau_A - tau_D) (eq. 2)",
            "ablation-dp-plus-tau-a": "ablation D_p + alpha*tau_A (Table 9)",
            "ablation-dp-minus-tau-d": "ablation D_p - lambda*tau_D (Table 10)",
        }[cfg.mode],
        "mode": cfg.mode,
        "alpha": cfg.alpha,
        "lambda": cfg.lam,
        "effective_coefficient": {"name": coeff_name, "value": coeff_value},
        "recipient": str(Path(cfg.recipient).resolve()),
        "output": str(Path(cfg.output).resolve()),
        "mapping": cfg.mapping.describe(),
        "coverage": _coverage_summary(plan, counters, policy, cfg),
        "tau_a_source": cfg.tau_a.describe() if cfg.tau_a else None,
        "tau_d_source": cfg.tau_d.describe() if cfg.tau_d else None,
        "n_mapped": len(plan),
        "n_unchanged": len(weight_map) - len(plan),
        "n_tensors": len(weight_map),
        "plan_counters": counters,
        "coefficient_guard": guard,
        "families": _family_summary(stats, ratios),
        "notes": notes,
        "warnings": warnings,
        "dtype": str(cfg.dtype).replace("torch.", ""),
        "date": datetime.datetime.now().astimezone().isoformat(),
        "git_commit": git_commit(),
        "torch_version": torch.__version__,
        "safetensors_version": safetensors.__version__,
    }
    manifest.update(cfg.extra_manifest)
    return manifest


def verify_written(cfg: MergeConfig, weight_map, plan, stats=None, log=print) -> dict:
    """Verification POST-ECRITURE (wrapper de `verify_checkpoint`).

    La verification v2 relit le checkpoint ecrit et le receveur, recalcule les
    vecteurs depuis les sources et compare les coefficients SIGNES. `stats` et
    `weight_map` ne sont plus utilises (la verification est autonome) mais
    restent acceptes pour compatibilite d'appel.
    """
    return verify_checkpoint(cfg, cfg.output, cfg.recipient, plan=plan, log=log)


def compute_effective_coefficients(merged_dir, reference_dir, cfg: MergeConfig,
                                    log=print) -> dict:
    """Recalcule les coefficients effectifs sur un checkpoint deja fusionne.

    Utilise par `scripts/verify_merge.py` : relit le checkpoint fusionne, le
    receveur, et recalcule les vecteurs depuis les sources. La configuration
    est d'abord validee (`validate_config`) : un merge verifie sans tau_D est
    refuse, au lieu d'etre accepte au titre d'un transfert.
    """
    validate_config(cfg)
    report = verify_checkpoint(cfg, merged_dir, reference_dir, plan=None, log=log)
    return report
