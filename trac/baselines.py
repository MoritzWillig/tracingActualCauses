#!/usr/bin/env python3
"""
baselines.py  –  Timing wrappers for baseline actual-cause algorithms.

Each public function returns a dict:
    {"elapsed_s": float, "error": str}

elapsed_s is -1.0 when the baseline is unavailable or raises an error.
All exceptions are caught and surfaced as non-empty "error" strings so the
benchmark never crashes because of a baseline.

Supported baselines
-------------------
  actualcauses (Reyd et al.)  – beam_search and ISI variants
      pip install actualcauses
  HP2SAT (Ibrahim et al.)     – via JPype bridge to compiled JAR
      repos/HP2SAT1.0/target/hp2sat-1.0.jar  (hp-optimization-library branch)
      Strategies timed: BRUTE_FORCE, SAT, MAX_SAT
      ILP/ILP_WHY skipped (require Gurobi, unavailable)
"""

from __future__ import annotations

import glob
import os
import signal
import sys
import time
from itertools import product as _iproduct
from typing import Any, Dict, Optional, Set

# ── Path setup ────────────────────────────────────────────────────────────

_HERE         = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT    = os.path.dirname(_HERE)

# Prefer the local checkout of actualcauses over any installed package so
# that fixes applied to repos/ActualCausesIdentification take effect.
_AC_LOCAL_SRC = os.path.join(_REPO_ROOT, "repos", "ActualCausesIdentification", "src")
if _AC_LOCAL_SRC not in sys.path:
    sys.path.insert(0, _AC_LOCAL_SRC)

# ── actualcauses availability ──────────────────────────────────────────────

_AC_AVAILABLE = False
try:
    import numpy as np
    from actualcauses import SCM as _AcSCM, BaseNumpyModel as _BNM  # noqa: F401
    _AC_AVAILABLE = True
except ImportError:
    pass


# ── HP2SAT JAR paths ──────────────────────────────────────────────────────
_HP2SAT_JAR   = os.path.join(_REPO_ROOT, "repos", "HP2SAT1.0", "target", "hp2sat-1.0.jar")
_HP2SAT_DEPDIR = os.path.join(os.path.dirname(_HP2SAT_JAR), "dependency")

_HP2SAT_AVAILABLE = os.path.exists(_HP2SAT_JAR)

# Per-process JVM state (each benchmark worker is a separate subprocess)
_JVM_READY = False   # True once JVM has been started in this process

# ── HP2SAT strategy registry ──────────────────────────────────────────────────
# name → (csv_t_key, csv_e_key, use_subprocess)
# use_subprocess=True  → run in a fresh spawned process (hard-kills on timeout)
# use_subprocess=False → run in-process (SIGALRM timeout, best-effort for JVM)
_STRATEGY_META: Dict[str, tuple] = {
    "BRUTE_FORCE": ("t_bf_s",  "e_bf",  True),
    "SAT":         ("t_sat_s", "e_sat", False),
    "MAX_SAT":     ("t_ms_s",  "e_ms",  False),
    "ILP":         ("t_ilp_s", "e_ilp", True),
}
_DEFAULT_HP2SAT_STRATEGIES = ["SAT", "MAX_SAT"]


def check_required_baselines(hp2sat_strategies: list) -> list:
    """
    Verify that every baseline needed for the current run is importable/present.

    Returns a list of human-readable error strings; an empty list means all OK.
    Call this once at benchmark startup and abort if the list is non-empty.

    beam / ISI are always checked because run_task() runs them unconditionally.
    HP2SAT strategies are only checked when *hp2sat_strategies* is non-empty.
    """
    errors = []
    if not _AC_AVAILABLE:
        errors.append(
            "actualcauses (beam / ISI) is not importable.  "
            "Clone repos/ActualCausesIdentification and apply actualCauses.patch, "
            "or run: pip install actualcauses"
        )
    if hp2sat_strategies and not _HP2SAT_AVAILABLE:
        errors.append(
            f"HP2SAT JAR not found at {_HP2SAT_JAR}.  "
            "Clone repos/HP2SAT1.0 (branch hp-optimization-library), apply "
            "HP2SAT.patch, and run: mvn package -DskipTests"
        )
    return errors


def _ensure_jvm() -> bool:
    """Start the JVM exactly once per process. Returns True if available."""
    global _JVM_READY, _HP2SAT_AVAILABLE
    if _JVM_READY:
        return _HP2SAT_AVAILABLE
    _JVM_READY = True
    if not _HP2SAT_AVAILABLE:
        return False
    try:
        import jpype
        if not jpype.isJVMStarted():
            deps = glob.glob(os.path.join(_HP2SAT_DEPDIR, "*.jar"))
            classpath = os.pathsep.join([_HP2SAT_JAR] + deps)
            jpype.startJVM(classpath=classpath, convertStrings=False)
        return True
    except Exception as exc:
        _HP2SAT_AVAILABLE = False
        return False


# ══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════════════════

class _BaselineTimeout(Exception):
    pass


def _alarm(signum, frame):
    raise _BaselineTimeout()


def _timed_call(fn, timeout: Optional[float]):
    """
    Call fn() and return (elapsed_s, error_str).

    Uses SIGALRM for the per-baseline timeout (Linux only); safe because each
    baseline call is already running inside a worker subprocess.
    """
    try:
        if timeout:
            signal.signal(signal.SIGALRM, _alarm)
            signal.alarm(max(1, int(timeout)))
        t0 = time.perf_counter()
        fn()
        return round(time.perf_counter() - t0, 6), ""
    except _BaselineTimeout:
        return round(float(timeout or 0), 6), "timeout"
    except Exception as exc:
        return -1.0, repr(exc)[:300]
    finally:
        if timeout:
            signal.alarm(0)


# ══════════════════════════════════════════════════════════════════════════════
# actualcauses SCM adapter
# ══════════════════════════════════════════════════════════════════════════════

def _build_ac_scm(our_scm, u_dict: Dict[str, int], target: str,
                  y_star: Set[int]):
    """
    Convert a trac SCM into an actualcauses.SCM.

    The adapter subclasses BaseNumpyModel and implements simulate() by
    iterating over the endogenous variables in topological order, evaluating
    our scalar structural equations row-by-row for the batch.  This is slower
    than a fully vectorised implementation but correct and requires no
    introspection of the closed-over gate parameters.

    actualcauses requires the target variable to be the *last* element of V.
    Endogenous variables are the only members of V; exogenous values are read
    from the u argument passed to simulate().
    """
    import networkx as nx

    topo = list(nx.topological_sort(our_scm._graph))
    exo_set  = set(our_scm.exo_vars)
    endo_set = set(our_scm.endo_vars)

    # Exogenous variables in topological order; defines the u-list index.
    exo_topo = [v for v in topo if v in exo_set]
    exo_idx  = {v: i for i, v in enumerate(exo_topo)}

    # Endogenous variables in topological order; target must be last.
    endo_topo = [v for v in topo if v in endo_set]
    endo_V    = [v for v in endo_topo if v != target] + [target]

    u_list      = [int(u_dict[v]) for v in exo_topo]
    parents_map = {v: our_scm._parents.get(v, []) for v in endo_V}
    eq_map      = {v: our_scm.equations[v] for v in endo_V}

    # ISI requires dag keys to be endogenous-only; exo parents are still used
    # by simulate() but must not appear as dag keys (ISI would KeyError on them).
    endo_only_parents = {v: [p for p in parents_map[v] if p in endo_set]
                         for v in endo_V}

    # phi: 1 when target reaches the desired y_star value.
    y_val = next(iter(y_star))
    if y_val == 1:
        phi_fn = lambda s: s[:, -1].astype(int)
    else:
        phi_fn = lambda s: (s[:, -1] == 0).astype(int)

    # Capture locals for the inner class (avoids referencing outer `self`).
    _endo_V    = endo_V
    _exo_set   = exo_set
    _exo_idx   = exo_idx
    _parents   = parents_map
    _eq        = eq_map

    class _AdaptedModel(_BNM):
        def simulate(self_m, u):  # noqa: N805
            n = self_m.S.shape[0]
            for endo_v in _endo_V:
                pa = _parents[endo_v]
                if not pa:
                    continue
                fn = _eq[endo_v]
                if n == 1:
                    p = {
                        pv: (int(u[_exo_idx[pv]]) if pv in _exo_set
                             else int(self_m.S[0, self_m.dim2id[pv]]))
                        for pv in pa
                    }
                    F_val = np.array([fn(p)], dtype=self_m.dtype)
                else:
                    F_val = np.zeros(n, dtype=self_m.dtype)
                    for row in range(n):
                        p = {
                            pv: (int(u[_exo_idx[pv]]) if pv in _exo_set
                                 else int(self_m.S[row, self_m.dim2id[pv]]))
                            for pv in pa
                        }
                        F_val[row] = fn(p)
                self_m[endo_v] = F_val

    model    = _AdaptedModel(V=endo_V, phi=phi_fn)
    dag_dict = endo_only_parents
    D        = [[0, 1]] * len(endo_V)

    return _AcSCM(V=endo_V, U=exo_topo, D=D, u=u_list, model=model,
                  dag=dag_dict)


# ══════════════════════════════════════════════════════════════════════════════
# Public timing functions
# ══════════════════════════════════════════════════════════════════════════════

def run_actualcauses_beam(
    our_scm, u_dict: Dict[str, int], target: str,
    y_star: Set[int], timeout: Optional[float] = None,
) -> Dict[str, Any]:
    """Time actualcauses beam_search on the given SCM."""
    if not _AC_AVAILABLE:
        return {"elapsed_s": -1.0, "error": "unavailable"}
    if target not in our_scm.endo_vars:
        return {"elapsed_s": -1.0, "error": ""}
    try:
        ac_scm = _build_ac_scm(our_scm, u_dict, target, y_star)
    except Exception as exc:
        return {"elapsed_s": -1.0, "error": "build:" + repr(exc)[:200]}
    elapsed, err = _timed_call(lambda: ac_scm.find_causes(ISI=False), timeout)
    return {"elapsed_s": elapsed, "error": err}


def run_actualcauses_isi(
    our_scm, u_dict: Dict[str, int], target: str,
    y_star: Set[int], timeout: Optional[float] = None,
) -> Dict[str, Any]:
    """Time actualcauses ISI on the given SCM."""
    if not _AC_AVAILABLE:
        return {"elapsed_s": -1.0, "error": "unavailable"}
    if target not in our_scm.endo_vars:
        return {"elapsed_s": -1.0, "error": ""}
    try:
        ac_scm = _build_ac_scm(our_scm, u_dict, target, y_star)
    except Exception as exc:
        return {"elapsed_s": -1.0, "error": "build:" + repr(exc)[:200]}
    elapsed, err = _timed_call(lambda: ac_scm.find_causes(ISI=True), timeout)
    return {"elapsed_s": elapsed, "error": err}


def run_hp2sat(
    our_scm, u_dict: Dict[str, int], target: str,
    y_star: Set[int], timeout: Optional[float] = None,
    strategies: Optional[list] = None,
) -> Dict[str, Any]:
    """
    Time HP2SAT (Ibrahim et al.) on the given SCM.

    *strategies* selects which solving strategies to run; defaults to
    ["SAT", "MAX_SAT"].  Pass ["SAT", "MAX_SAT", "BRUTE_FORCE", "ILP"] to
    include the slow strategies.

    Fast strategies (SAT, MAX_SAT) run in-process.
    Slow strategies (BRUTE_FORCE, ILP) run in a freshly spawned subprocess
    so they cannot block the caller even when SIGALRM fails to interrupt the
    blocking JVM call.  The subprocess is hard-killed on timeout.

    Returns a flat dict with t_*_s / e_* keys for every known strategy.
    Strategies not in *strategies* carry e_* = "skipped".
    """
    if strategies is None:
        strategies = _DEFAULT_HP2SAT_STRATEGIES

    # Initialise all keys: requested → blank error, others → "skipped".
    results: Dict[str, Any] = {}
    for name, (tk, ek, _) in _STRATEGY_META.items():
        results[tk] = -1.0
        results[ek] = "" if name in strategies else "skipped"

    if not strategies:
        return results

    if target not in our_scm.endo_vars:
        return results

    if not _ensure_jvm():
        for name in strategies:
            tk, ek, _ = _STRATEGY_META[name]
            results[tk] = -1.0
            results[ek] = "unavailable"
        return results

    in_proc = [s for s in strategies if not _STRATEGY_META[s][2]]
    in_sub  = [s for s in strategies if     _STRATEGY_META[s][2]]

    # ── In-process strategies (SAT, MAX_SAT) ─────────────────────────────────
    if in_proc:
        model = context = phi = None
        build_err = None
        try:
            model, context, phi, _ = _build_hp2sat_model(our_scm, u_dict, target, y_star)
        except Exception as exc:
            build_err = "build:" + repr(exc)[:200]

        for name in in_proc:
            tk, ek, _ = _STRATEGY_META[name]
            if build_err or model is None:
                results[tk] = -1.0
                results[ek] = build_err or "build_failed"
            else:
                elapsed, err = _timed_call(
                    lambda s=name: _run_hp2sat_strategy(model, context, phi, s),
                    timeout,
                )
                results[tk] = elapsed
                results[ek] = err

    # ── Subprocess strategies (BRUTE_FORCE, ILP) ─────────────────────────────
    # Run after in-process ones so fast results are already recorded.
    if in_sub:
        prims: Optional[Dict] = None
        prims_err = None
        try:
            prims = _hp2sat_primitives(our_scm, u_dict, target, y_star)
        except Exception as exc:
            prims_err = "primitives:" + repr(exc)[:200]

        for name in in_sub:
            tk, ek, _ = _STRATEGY_META[name]
            if prims is None:
                results[tk] = -1.0
                results[ek] = prims_err or "primitives_failed"
            else:
                elapsed, err = _run_hp2sat_subprocess(prims, name, timeout)
                results[tk] = elapsed
                results[ek] = err

    return results


def _build_hp2sat_model(our_scm, u_dict: Dict[str, int], target: str, y_star: Set[int]):
    """Convert a trac SCM to an HP2SAT CausalModel via JPype."""
    import jpype
    JHashSet     = jpype.JClass("java.util.HashSet")
    CausalModel  = jpype.JClass("de.tum.in.i4.hp2sat.causality.CausalModel")
    Equation     = jpype.JClass("de.tum.in.i4.hp2sat.causality.Equation")
    FormulaFactory = jpype.JClass("org.logicng.formulas.FormulaFactory")

    FormulaHelper = jpype.JClass("de.tum.in.i4.hp2sat.util.FormulaHelper")
    f = FormulaFactory()
    equations = JHashSet()
    exo_vars  = JHashSet()

    # Exogenous variables
    for var in our_scm.exo_vars:
        exo_vars.add(f.variable(var))

    # Endogenous variables: use stored gate info when available (compact formula),
    # fall back to DNF truth-table synthesis for equations we can't introspect.
    gate_info = getattr(our_scm, "_gate_info", {})
    for var in our_scm.endo_vars:
        parents = our_scm._parents.get(var, [])
        if var in gate_info:
            formula = _gate_formula(f, FormulaHelper, gate_info[var])
        else:
            formula = _dnf_formula(f, FormulaHelper, parents, our_scm.equations[var])
        equations.add(Equation(f.variable(var), formula))

    model = CausalModel("BenchModel", equations, exo_vars, f)

    # Context: assignment to exogenous variables
    context = JHashSet()
    for var, val in u_dict.items():
        context.add(f.literal(var, bool(val)))

    # Phi: target event
    y_val = next(iter(y_star))
    if y_val:
        phi = f.variable(target)
    else:
        phi = FormulaHelper.fNot(f, f.variable(target))

    return model, context, phi, f


def _gate_formula(f, fh, info: Dict):
    """
    Build a compact LogicNG formula from stored gate metadata.

    *info* has keys: gate ("and"/"or"), parents (list of str), negations (list of bool).
    A negated input becomes a negative literal; the gate determines and/or.
    This produces a formula of size O(|parents|) instead of the 2^|parents| DNF.
    """
    import jpype
    JArrayList = jpype.JClass("java.util.ArrayList")

    gate      = info["gate"]
    parents   = info["parents"]
    negations = info["negations"]

    # neg=True in trac means the input is negated → negative literal (phase=False)
    literals = [f.literal(p, not neg) for p, neg in zip(parents, negations)]

    if len(literals) == 1:
        return literals[0]

    jlist = JArrayList()
    for lit in literals:
        jlist.add(lit)
    return fh.fAnd(f, jlist) if gate == "and" else fh.fOr(f, jlist)


def _dnf_formula(f, fh, parents, eq_fn):
    """
    Build a LogicNG formula equivalent to eq_fn using DNF (sum of minterms).
    Works for any Boolean function regardless of its internal structure.

    *fh* is the FormulaHelper JClass (cached by the caller to avoid repeated
    JClass lookups).  FormulaHelper wraps f.and / f.or under Python-safe names.
    """
    if not parents:
        return f.verum()

    import jpype
    JArrayList = jpype.JClass("java.util.ArrayList")

    n = len(parents)
    true_minterms = []

    for bits in _iproduct(range(2), repeat=n):
        if eq_fn(dict(zip(parents, bits))):
            literals = [f.literal(p, bool(b)) for p, b in zip(parents, bits)]
            if len(literals) == 1:
                minterm = literals[0]
            else:
                jlist = JArrayList()
                for lit in literals:
                    jlist.add(lit)
                minterm = fh.fAnd(f, jlist)
            true_minterms.append(minterm)

    if not true_minterms:
        return f.falsum()
    if len(true_minterms) == 2**n:
        return f.verum()
    if len(true_minterms) == 1:
        return true_minterms[0]

    jlist = JArrayList()
    for mt in true_minterms:
        jlist.add(mt)
    return fh.fOr(f, jlist)


def _dnf_formula_from_table(f, fh, parents, table):
    """Build a LogicNG DNF formula from a precomputed truth table (tuple→bool dict)."""
    if not parents:
        return f.verum()

    import jpype
    JArrayList = jpype.JClass("java.util.ArrayList")

    n = len(parents)
    true_minterms = []
    for bits, val in table.items():
        if val:
            literals = [f.literal(p, bool(b)) for p, b in zip(parents, bits)]
            if len(literals) == 1:
                minterm = literals[0]
            else:
                jlist = JArrayList()
                for lit in literals:
                    jlist.add(lit)
                minterm = fh.fAnd(f, jlist)
            true_minterms.append(minterm)

    if not true_minterms:
        return f.falsum()
    if len(true_minterms) == 2**n:
        return f.verum()
    if len(true_minterms) == 1:
        return true_minterms[0]

    jlist = JArrayList()
    for mt in true_minterms:
        jlist.add(mt)
    return fh.fOr(f, jlist)


_JAVA_DEVNULL = None  # cached /dev/null PrintStream for suppressing Java stdout


def _run_hp2sat_strategy(model, context, phi, strategy_name: str) -> None:
    """Call model.getAllCauses for one SolvingStrategy (discards result, only for timing)."""
    import jpype
    global _JAVA_DEVNULL

    SolvingStrategy = jpype.JClass("de.tum.in.i4.hp2sat.causality.SolvingStrategy")
    strategy = getattr(SolvingStrategy, strategy_name)

    # Suppress verbose Java stdout (MaxSAT solver prints progress lines)
    if _JAVA_DEVNULL is None:
        _JAVA_DEVNULL = jpype.JClass("java.io.PrintStream")(
            jpype.JClass("java.io.FileOutputStream")(
                jpype.JClass("java.io.File")("/dev/null")
            )
        )
    System   = jpype.JClass("java.lang.System")
    orig_out = System.out
    try:
        System.setOut(_JAVA_DEVNULL)
        model.getAllCauses(context, phi, strategy)
    finally:
        System.setOut(orig_out)


# ══════════════════════════════════════════════════════════════════════════════
# Subprocess isolation for slow HP2SAT strategies (BRUTE_FORCE, ILP)
# ══════════════════════════════════════════════════════════════════════════════
# SIGALRM cannot interrupt a blocking JVM call, so BRUTE_FORCE can hold up the
# entire worker process for minutes.  The fix is to run slow strategies in a
# fresh subprocess (spawn, not fork) and terminate() it on timeout.
# The primitives dict carries only picklable data (no lambdas).


def _hp2sat_primitives(our_scm, u_dict: Dict[str, int],
                       target: str, y_star: Set[int]) -> Dict:
    """
    Extract a fully picklable snapshot of the SCM needed to rebuild the
    HP2SAT CausalModel in a subprocess.

    For variables with stored gate info the compact {gate, parents, negations}
    record is used.  For black-box equations the truth table is precomputed
    so that no lambdas need to cross process boundaries.
    """
    gate_info = getattr(our_scm, "_gate_info", {})
    var_formulas: Dict[str, Any] = {}
    for var in our_scm.endo_vars:
        parents = our_scm._parents.get(var, [])
        if var in gate_info:
            info = gate_info[var]
            var_formulas[var] = {
                "type":      "gate",
                "gate":      info["gate"],
                "parents":   list(info["parents"]),
                "negations": list(info["negations"]),
            }
        else:
            eq = our_scm.equations[var]
            table = {
                bits: bool(eq(dict(zip(parents, bits))))
                for bits in _iproduct(range(2), repeat=len(parents))
            }
            var_formulas[var] = {
                "type":    "table",
                "parents": list(parents),
                "table":   table,
            }
    return {
        "exo_vars":     list(our_scm.exo_vars),
        "endo_vars":    list(our_scm.endo_vars),
        "var_formulas": var_formulas,
        "u_dict":       dict(u_dict),
        "target":       target,
        "y_star":       list(y_star),
    }


def _build_hp2sat_model_raw(primitives: Dict):
    """
    Build an HP2SAT CausalModel from a picklable primitives dict.
    Used by the subprocess worker (no SCM object available there).
    """
    import jpype
    JHashSet       = jpype.JClass("java.util.HashSet")
    CausalModel    = jpype.JClass("de.tum.in.i4.hp2sat.causality.CausalModel")
    Equation       = jpype.JClass("de.tum.in.i4.hp2sat.causality.Equation")
    FormulaFactory = jpype.JClass("org.logicng.formulas.FormulaFactory")
    FormulaHelper  = jpype.JClass("de.tum.in.i4.hp2sat.util.FormulaHelper")
    f = FormulaFactory()

    equations  = JHashSet()
    exo_vars_j = JHashSet()

    for var in primitives["exo_vars"]:
        exo_vars_j.add(f.variable(var))

    for var in primitives["endo_vars"]:
        vf = primitives["var_formulas"][var]
        if vf["type"] == "gate":
            formula = _gate_formula(f, FormulaHelper, vf)
        else:
            formula = _dnf_formula_from_table(f, FormulaHelper,
                                              vf["parents"], vf["table"])
        equations.add(Equation(f.variable(var), formula))

    model = CausalModel("BenchModel", equations, exo_vars_j, f)

    context = JHashSet()
    for var, val in primitives["u_dict"].items():
        context.add(f.literal(var, bool(val)))

    target = primitives["target"]
    y_val  = primitives["y_star"][0]
    phi    = (f.variable(target) if y_val
              else FormulaHelper.fNot(f, f.variable(target)))

    return model, context, phi, f


def _hp2sat_subprocess_worker(q, primitives: Dict, strategy_name: str) -> None:
    """
    Top-level subprocess entry point (must be picklable).

    Starts a fresh JVM, builds the model, runs one strategy, and puts
    (elapsed_s, error_str) onto *q*.  No SIGALRM is used here — the parent
    provides the hard deadline by calling process.terminate().
    """
    try:
        if not _ensure_jvm():
            q.put((-1.0, "jvm_unavailable"))
            return
        model, context, phi, _ = _build_hp2sat_model_raw(primitives)
    except Exception as exc:
        q.put((-1.0, "build:" + repr(exc)[:200]))
        return
    elapsed, err = _timed_call(
        lambda: _run_hp2sat_strategy(model, context, phi, strategy_name),
        None,
    )
    q.put((elapsed, err))


def _run_hp2sat_subprocess(
    primitives: Dict, strategy_name: str, timeout: Optional[float]
) -> tuple:
    """
    Run one HP2SAT strategy in a freshly spawned subprocess.

    Uses the 'spawn' start method so the child never inherits a partially
    initialised JVM from the parent.  Hard-kills the child on timeout.
    Returns (elapsed_s, error_str).
    """
    import multiprocessing
    ctx = multiprocessing.get_context("spawn")
    q   = ctx.Queue()
    p   = ctx.Process(
        target=_hp2sat_subprocess_worker,
        args=(q, primitives, strategy_name),
        daemon=True,
    )
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join(5)
        if p.is_alive():
            p.kill()
            p.join()
        return round(float(timeout or 0), 6), "timeout"
    try:
        return q.get_nowait()
    except Exception:
        return -1.0, f"subprocess_exit_{p.exitcode}"
