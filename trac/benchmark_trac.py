#!/usr/bin/env python3
"""
benchmark_trac.py  –  Runtime benchmark for the TrAC algorithm.

Measures wall-clock time of trac() on randomly generated Boolean SCMs
built from acyclic Erdős-Rényi DAGs with random AND/OR gates (inputs
independently negated with probability 0.5).

Configuration constants are at the top of this file.

Quick start
-----------
    # single-threaded, write results to benchmark_results.csv
    python trac/benchmark_trac.py

    # 8 parallel workers with a 2-minute per-task timeout
    python trac/benchmark_trac.py --workers 8 --timeout 120

    # convenience launcher (defaults: 8 workers, 120 s timeout)
    bash run_benchmark.sh [WORKERS [TIMEOUT [OUTPUT]]]

Outputs
-------
    results/data/n*_d*.csv        per-cell incremental measurements
    results/benchmark_config.json benchmark parameters + run summary

Dependencies: networkx  (see README for conda setup)
"""

from __future__ import annotations

import argparse
import csv
import datetime
import faulthandler
import glob
import json
import os
import platform
import random
import signal
import statistics
import sys
import time
from concurrent.futures import BrokenExecutor, ProcessPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeout
from typing import Any, Dict, List, Optional

faulthandler.enable()

# Ensure the directory containing trac.py is importable regardless of CWD.
_HERE         = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
from trac import SCM, trac  # noqa: E402
import baselines as _baselines  # noqa: E402


# ══════════════════════════════════════════════════════════════════════════════
# Configuration  ← edit here to change the benchmark matrix
# ══════════════════════════════════════════════════════════════════════════════

NODE_COUNTS: List[int]   = list(range(5, 21))   # 5 … 20 nodes inclusive
DENSITIES:   List[float] = [0.25, 0.50, 0.75]   # Erdős-Rényi edge probability
REPETITIONS: int         = 20                   # independent repetitions per cell
BASE_SEED:   int         = 42                   # reproducibility anchor
# Max resampling attempts to obtain a weakly-connected DAG.  Empirically
# the worst case (n=5, d=0.25) needs ≤ 52 retries across 10 000 seeds.
GRAPH_CONNECT_RETRIES: int = 1000

DEFAULT_TIMEOUT:      Optional[float] = 120.0   # seconds per task; 0 = unlimited
DEFAULT_WORKERS:      int             = 1
DEFAULT_RESULTS_DIR:  str             = os.path.join(_PROJECT_ROOT, "results")
DEFAULT_HP2SAT_STRATEGIES: str        = "SAT,MAX_SAT"  # comma-separated; add BRUTE_FORCE,ILP to opt in


# ══════════════════════════════════════════════════════════════════════════════
# Random SCM Generator
# ══════════════════════════════════════════════════════════════════════════════

def _is_weakly_connected(parents: Dict[int, List[int]], n: int) -> bool:
    """Return True iff the DAG described by *parents* is weakly connected."""
    if n <= 1:
        return True
    adj: Dict[int, List[int]] = {i: [] for i in range(n)}
    for j, pa in parents.items():
        for i in pa:
            adj[i].append(j)
            adj[j].append(i)
    visited = {0}
    stack = [0]
    while stack:
        node = stack.pop()
        for nb in adj[node]:
            if nb not in visited:
                visited.add(nb)
                stack.append(nb)
    return len(visited) == n


def random_bool_scm(
    n_nodes:   int,
    edge_prob: float,
    seed:      int,
):
    """
    Build a random Boolean SCM on *n_nodes* nodes whose DAG is a single
    weakly connected component, with the deepest node(s) chosen as target.

    DAG topology
    ~~~~~~~~~~~~
    Nodes are ordered V0 … V_{n-1}.  For each pair (i, j) with i < j, an
    edge i → j is drawn independently with probability *edge_prob*.  Because
    edges only go forward in index order the resulting graph is acyclic.
    Edge sets are resampled (with the same RNG, advancing its state) until
    the resulting undirected graph is connected.

    The maximum possible edges for n nodes is n*(n-1)/2, so *edge_prob*
    controls the fraction of forward edges that actually exist.

    Gate assignment
    ~~~~~~~~~~~~~~~
    Each variable with ≥1 parent receives a random AND or OR gate; each input
    is independently negated with probability 0.5.
    Variables with no parents become exogenous (Boolean domain {0, 1}).

    Returns
    -------
    (scm, n_exo, n_endo, n_edges, target)
        scm     – fully initialised SCM
        n_exo   – number of exogenous variables
        n_endo  – number of endogenous variables
        n_edges – total edges in the DAG
        target  – the deepest endogenous variable (longest path from any root);
                  chosen uniformly at random among ties
    """
    rng   = random.Random(seed)
    names = [f"V{i}" for i in range(n_nodes)]

    # Sample parent sets: only forward edges (i < j) → acyclicity guaranteed.
    # Resample until the underlying undirected graph is weakly connected.
    for _attempt in range(GRAPH_CONNECT_RETRIES):
        parents: Dict[int, List[int]] = {i: [] for i in range(n_nodes)}
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if rng.random() < edge_prob:
                    parents[j].append(i)
        if _is_weakly_connected(parents, n_nodes):
            break
    else:
        raise RuntimeError(
            f"Could not generate a connected {n_nodes}-node DAG with "
            f"edge_prob={edge_prob} after {GRAPH_CONNECT_RETRIES} attempts."
        )

    scm = SCM()
    scm._gate_info = {}   # var → {gate, parents, negations}
    n_exo = n_endo = n_edges = 0

    for i in range(n_nodes):
        var  = names[i]
        pa_i = parents[i]

        if not pa_i:
            scm.add_exogenous(var, {0, 1})
            n_exo += 1
        else:
            pa_names  = [names[p] for p in pa_i]
            gate      = rng.choice(["and", "or"])
            negations = [rng.random() < 0.5 for _ in pa_names]
            n_edges  += len(pa_names)

            # Capture loop variables by default-argument binding to avoid the
            # classic Python closure-in-loop gotcha.
            def _make_eq(g=gate, pn=pa_names, ng=negations):
                def eq(p: Dict[str, Any]) -> int:
                    vals = [int(not p[k]) if neg else int(p[k])
                            for k, neg in zip(pn, ng)]
                    return int(all(vals)) if g == "and" else int(any(vals))
                return eq

            scm.add_endogenous(var, pa_names, _make_eq(), domain={0, 1})
            scm._gate_info[var] = {
                "gate": gate, "parents": pa_names, "negations": negations,
            }
            n_endo += 1

    # Target: deepest endogenous node (longest path from any root).
    # Depth is computed in topological order (0 … n-1 is already topological).
    if scm.endo_vars:
        depth = [0] * n_nodes
        for i in range(n_nodes):
            if parents[i]:
                depth[i] = 1 + max(depth[p] for p in parents[i])
        endo_set = set(scm.endo_vars)
        max_depth = max(depth[i] for i in range(n_nodes) if names[i] in endo_set)
        deepest = [names[i] for i in range(n_nodes)
                   if names[i] in endo_set and depth[i] == max_depth]
        target = rng.choice(deepest)
    else:
        target = names[-1]
    return scm, n_exo, n_endo, n_edges, target


# ══════════════════════════════════════════════════════════════════════════════
# Task Runner  (each call executes inside one worker process)
# ══════════════════════════════════════════════════════════════════════════════

class _BenchTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _BenchTimeout()


def run_task(task: Dict) -> Dict:
    """
    Execute one benchmark cell.  Safe to call in a subprocess worker.

    *task* keys
    -----------
    n_nodes, density, density_idx, rep  – cell coordinates
    timeout                             – per-task wall-clock limit (s) or None
    base_seed                           – reproducibility anchor
    """
    n_nodes          = task["n_nodes"]
    density          = task["density"]
    density_idx      = task["density_idx"]
    rep              = task["rep"]
    timeout          = task.get("timeout")
    base_seed        = task.get("base_seed", BASE_SEED)
    hp2sat_strategies = task.get("hp2sat_strategies", ["SAT", "MAX_SAT"])

    # Deterministic seed: unique for every (n_nodes, density, rep) triple
    seed = base_seed + rep + n_nodes * 100 + density_idx * 10_000
    out  = {
        "n_nodes": n_nodes,
        "density": density,
        "rep":     rep,
        "seed":    seed,
    }

    _tag = f"[worker pid={os.getpid()} n={n_nodes} d={density:.2f} rep={rep} seed={seed}]"

    def _step(name: str) -> None:
        print(f"{_tag} {name}", file=sys.stderr, flush=True)

    _step("START")

    try:
        if timeout is not None:
            signal.signal(signal.SIGALRM, _alarm_handler)
            signal.alarm(max(1, int(timeout)))

        scm, n_exo, n_endo, n_edges, target = random_bool_scm(n_nodes, density, seed)

        # Exogenous assignment: separate seed offset to keep it independent of
        # the graph structure seed.
        rng = random.Random(seed ^ 0xDEADBEEF)
        u   = {v: rng.randint(0, 1) for v in scm.exo_vars}

        # Y_star = {actual value} so AC1 always holds; Y_hat is the complement.
        wu     = scm.actual_world(u)
        y_star = {wu[target]}

        _step("trac")
        t0      = time.perf_counter()
        result  = trac(scm, u, target, y_star)
        elapsed = time.perf_counter() - t0
        _step("trac done")

        out.update({
            "n_exo":       n_exo,
            "n_endo":      n_endo,
            "n_edges":     n_edges,
            "target":      target,
            "n_causes":    len(result.causes),
            "elapsed_s":   round(elapsed, 6),
            "timed_out":   False,
            "error":       "",
            "n_cands":     result.n_candidates,
            "n_skip_sup":  result.n_skipped_superset,
            "n_skip_up":   result.n_skipped_uprojection,
            "n_part_skip": result.n_partitions_skipped,
            "n_part_test": result.n_partitions_tested,
        })

        # Disarm main alarm; each baseline manages its own timeout via SIGALRM.
        if timeout is not None:
            signal.alarm(0)

        # ── Baselines – timed independently; never crash the task ────────────
        _bl_timeout = timeout  # same per-task budget
        _step("beam")
        beam = _baselines.run_actualcauses_beam(scm, u, target, y_star, _bl_timeout)
        _step("isi")
        isi  = _baselines.run_actualcauses_isi( scm, u, target, y_star, _bl_timeout)
        _step(f"hp2sat {hp2sat_strategies}")
        hp2  = _baselines.run_hp2sat(scm, u, target, y_star, _bl_timeout,
                                     strategies=hp2sat_strategies)
        _step("hp2sat done")
        out.update({
            "t_beam_s":  beam["elapsed_s"], "e_beam":  beam["error"],
            "t_isi_s":   isi["elapsed_s"],  "e_isi":   isi["error"],
            "t_bf_s":  hp2["t_bf_s"],  "e_bf":  hp2["e_bf"],
            "t_sat_s": hp2["t_sat_s"], "e_sat": hp2["e_sat"],
            "t_ms_s":  hp2["t_ms_s"],  "e_ms":  hp2["e_ms"],
            "t_ilp_s": hp2["t_ilp_s"], "e_ilp": hp2["e_ilp"],
        })
        _step("DONE")

    except _BenchTimeout:
        out.update({
            "n_exo": -1, "n_endo": -1, "n_edges": -1,
            "target": "?", "n_causes": -1,
            "elapsed_s": float(timeout or 0),
            "timed_out": True,
            "error":     "timeout",
            "n_cands": -1, "n_skip_sup": -1, "n_skip_up": -1,
            "n_part_skip": -1, "n_part_test": -1,
            "t_beam_s": -1.0, "e_beam": "n/a",
            "t_isi_s":  -1.0, "e_isi":  "n/a",
            "t_bf_s": -1.0, "e_bf": "n/a",
            "t_sat_s": -1.0, "e_sat": "n/a",
            "t_ms_s": -1.0, "e_ms": "n/a",
            "t_ilp_s": -1.0, "e_ilp": "n/a",
        })

    except Exception as exc:
        out.update({
            "n_exo": -1, "n_endo": -1, "n_edges": -1,
            "target": "?", "n_causes": -1,
            "elapsed_s": -1.0,
            "timed_out": False,
            "error":     repr(exc),
            "n_cands": -1, "n_skip_sup": -1, "n_skip_up": -1,
            "n_part_skip": -1, "n_part_test": -1,
            "t_beam_s": -1.0, "e_beam": "n/a",
            "t_isi_s":  -1.0, "e_isi":  "n/a",
            "t_bf_s": -1.0, "e_bf": "n/a",
            "t_sat_s": -1.0, "e_sat": "n/a",
            "t_ms_s": -1.0, "e_ms": "n/a",
            "t_ilp_s": -1.0, "e_ilp": "n/a",
        })

    finally:
        # Always disarm the alarm so the worker can be reused cleanly.
        if timeout is not None:
            signal.alarm(0)

    return out


# ══════════════════════════════════════════════════════════════════════════════
# Output Helpers
# ══════════════════════════════════════════════════════════════════════════════

_CSV_FIELDS = [
    "n_nodes", "density", "rep", "seed",
    "n_exo", "n_endo", "n_edges",
    "target", "n_causes",
    "elapsed_s", "timed_out", "error",
    # Remark-5 pruning counters
    "n_cands", "n_skip_sup", "n_skip_up", "n_part_skip", "n_part_test",
    # Baseline timings (actualcauses: Reyd et al.; HP2SAT: Ibrahim et al.)
    "t_beam_s", "e_beam",
    "t_isi_s",  "e_isi",
    # HP2SAT strategies
    "t_bf_s",  "e_bf",
    "t_sat_s", "e_sat",
    "t_ms_s",  "e_ms",
    "t_ilp_s", "e_ilp",
]

_HDR = (f"{'n':>4}  {'density':>7}  {'rep':>3}  "
        f"{'exo':>4}  {'endo':>4}  {'edges':>5}  "
        f"{'causes':>6}  {'time (s)':>10}  notes")
_SEP = "-" * 68


def _print_header():
    print(_HDR)
    print(_SEP)


def _print_row(r: Dict):
    notes = "TIMEOUT" if r.get("timed_out") else (r.get("error") or "")
    print(
        f"{r['n_nodes']:>4}  {r['density']:>7.2f}  {r['rep']:>3}  "
        f"{r.get('n_exo', -1):>4}  {r.get('n_endo', -1):>4}  "
        f"{r.get('n_edges', -1):>5}  "
        f"{r.get('n_causes', -1):>6}  "
        f"{r.get('elapsed_s', -1.0):>10.4f}  {notes}",
        flush=True,
    )


def _save_config(args, results: List[Dict], path: str) -> None:
    done  = [r for r in results if not r.get("timed_out") and not r.get("error")]
    n_to  = sum(1 for r in results if r.get("timed_out"))
    n_err = sum(1 for r in results if r.get("error") and not r.get("timed_out"))

    cfg = {
        "node_counts":  list(range(args.min_nodes, args.max_nodes + 1)),
        "densities":    DENSITIES,
        "repetitions":  REPETITIONS,
        "base_seed":    args.seed,
        "timeout_s":    args.timeout if args.timeout > 0 else None,
        "workers":      args.workers,
        "timestamp":    datetime.datetime.now().isoformat(timespec="seconds"),
        "python":       sys.version,
        "platform":     platform.platform(),
        "total_tasks":  len(results),
        "completed":    len(done),
        "timed_out":    n_to,
        "errors":       n_err,
    }
    if done:
        times = [r["elapsed_s"] for r in done]
        cfg["elapsed_min_s"]  = round(min(times), 6)
        cfg["elapsed_max_s"]  = round(max(times), 6)
        cfg["elapsed_mean_s"] = round(statistics.mean(times), 6)

    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"  Config  → {path}")


# ── Resume helpers ────────────────────────────────────────────────────────────

def _task_seed(task: Dict) -> int:
    """Compute the deterministic seed for a task dict (mirrors run_task logic)."""
    return task["base_seed"] + task["rep"] + task["n_nodes"] * 100 + task["density_idx"] * 10_000


def _load_completed_seeds(cells_dir: str) -> set:
    """Return the set of seeds already written to any per-cell CSV in cells_dir."""
    seeds: set = set()
    for path in glob.glob(os.path.join(cells_dir, "n*_d*.csv")):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    s = int(row["seed"])
                    if s != -1:
                        seeds.add(s)
                except (KeyError, ValueError):
                    pass
    return seeds


# HP2SAT strategy (time_col, error_col) pairs — used for cross-run merging.
_HP2SAT_STRATEGY_PAIRS = [
    ("t_bf_s",  "e_bf"),
    ("t_sat_s", "e_sat"),
    ("t_ms_s",  "e_ms"),
    ("t_ilp_s", "e_ilp"),
]


def _merge_rows(old: Dict, new: Dict) -> Dict:
    """
    Merge two result rows for the same seed (update mode).

    For each HP2SAT strategy: if the new run skipped it (e_* == "skipped")
    but the old run succeeded (e_* == ""), keep the old timing.  All other
    columns take the new row's value (fresher measurement wins).
    """
    merged = dict(new)
    for tk, ek in _HP2SAT_STRATEGY_PAIRS:
        if new.get(ek) == "skipped" and old.get(ek, "skipped") == "":
            merged[tk] = old[tk]
            merged[ek] = old[ek]
    return merged


# ── Per-cell incremental writer ───────────────────────────────────────────────

def _write_cell_row(r: Dict, results_dir: str, update: bool = False) -> None:
    """
    Write one row to its per-cell CSV file (n*_d*.csv).

    Normal mode  – append to the file (resume behaviour).
    Update mode  – if a row with the same seed already exists, merge it
                   (preserving valid timings from the previous run for
                   strategies not requested this time) and rewrite the file;
                   otherwise append as usual.

    Rows with seed == -1 (executor-level failures) are always skipped.
    Called only from the main process → no concurrency.
    """
    if r.get("seed", -1) == -1:   # FuturesTimeout / executor-level exception
        return
    n, d = r["n_nodes"], r["density"]
    path = os.path.join(results_dir, f"n{n:02d}_d{d:.2f}.csv")

    if update and os.path.exists(path):
        existing: List[Dict] = []
        with open(path, newline="") as f:
            existing = list(csv.DictReader(f))
        seed_str = str(r["seed"])
        replaced = False
        for i, row in enumerate(existing):
            if row.get("seed") == seed_str:
                existing[i] = _merge_rows(row, r)
                replaced = True
                break
        if not replaced:
            existing.append(r)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(existing)
    else:
        new_file = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
            if new_file:
                w.writeheader()
            w.writerow(r)


# ══════════════════════════════════════════════════════════════════════════════
# Task List Builder
# ══════════════════════════════════════════════════════════════════════════════

def build_tasks(
    node_counts:       List[int]        = NODE_COUNTS,
    densities:         List[float]      = DENSITIES,
    repetitions:       int              = REPETITIONS,
    timeout:           Optional[float]  = DEFAULT_TIMEOUT,
    base_seed:         int              = BASE_SEED,
    hp2sat_strategies: Optional[List[str]] = None,
) -> List[Dict]:
    if hp2sat_strategies is None:
        hp2sat_strategies = [s.strip() for s in DEFAULT_HP2SAT_STRATEGIES.split(",") if s.strip()]
    tasks = []
    for n in node_counts:
        for di, d in enumerate(densities):
            for r in range(repetitions):
                tasks.append({
                    "n_nodes":          n,
                    "density":          d,
                    "density_idx":      di,
                    "rep":              r,
                    "timeout":          timeout,
                    "base_seed":        base_seed,
                    "hp2sat_strategies": hp2sat_strategies,
                })
    return tasks


# ══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ══════════════════════════════════════════════════════════════════════════════

def _parse_node_counts(spec: str) -> List[int]:
    """
    Parse a compact node-count spec into a sorted, deduplicated list.

    Supported tokens (comma-separated):
      "5-20"   → range(5, 21)   every integer from 5 to 20 inclusive
      "25"     → [25]           single value
    """
    result: List[int] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            lo, hi = token.split("-", 1)
            result.extend(range(int(lo), int(hi) + 1))
        else:
            result.append(int(token))
    if not result:
        raise ValueError(f"--node-counts produced an empty list from: {spec!r}")
    return sorted(set(result))


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark TrAC on random Boolean SCMs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS,
        help="Number of parallel worker processes",
    )
    parser.add_argument(
        "--results-dir", default=DEFAULT_RESULTS_DIR,
        help="Directory for per-cell CSVs and the config JSON",
    )
    parser.add_argument(
        "--timeout", type=float, default=DEFAULT_TIMEOUT,
        help="Per-task wall-clock timeout in seconds (0 = unlimited)",
    )
    parser.add_argument(
        "--seed", type=int, default=BASE_SEED,
        help="Base random seed for reproducibility",
    )
    parser.add_argument(
        "--repetitions", type=int, default=REPETITIONS,
        help="Number of independent repetitions per (n_nodes, density) cell",
    )
    parser.add_argument(
        "--min-nodes", type=int, default=min(NODE_COUNTS),
        help="Smallest node count (used when --node-counts is not set)",
    )
    parser.add_argument(
        "--max-nodes", type=int, default=max(NODE_COUNTS),
        help="Largest node count inclusive (used when --node-counts is not set)",
    )
    parser.add_argument(
        "--node-counts", default=None,
        help=(
            "Explicit comma-separated node counts, overrides --min/max-nodes. "
            "Ranges like 5-20 expand to every integer; plain integers are used "
            "as-is.  Example: 5-20,25,30,35,40,45,50"
        ),
    )
    parser.add_argument(
        "--hp2sat-strategies", default=DEFAULT_HP2SAT_STRATEGIES,
        help=(
            "Comma-separated HP2SAT strategies to time. "
            "Fast (in-process): SAT, MAX_SAT. "
            "Slow (subprocess, hard timeout): BRUTE_FORCE, ILP. "
            "Example: SAT,MAX_SAT,BRUTE_FORCE"
        ),
    )
    parser.add_argument(
        "--update", action="store_true", default=False,
        help=(
            "Re-run all tasks even if their seeds already exist in the per-cell "
            "CSVs.  Results are merged: timings from a previous run are kept for "
            "strategies not requested in this run (e_* == 'skipped' in the new "
            "result), so you can fill in a single strategy without losing others."
        ),
    )
    args = parser.parse_args()

    if args.node_counts:
        node_counts = _parse_node_counts(args.node_counts)
    else:
        node_counts = list(range(args.min_nodes, args.max_nodes + 1))
    timeout           = args.timeout if args.timeout > 0 else None
    hp2sat_strategies = [s.strip() for s in args.hp2sat_strategies.split(",") if s.strip()]

    preflight_errors = _baselines.check_required_baselines(hp2sat_strategies)
    if preflight_errors:
        print("Benchmark aborted – required baselines are unavailable:", file=sys.stderr)
        for err in preflight_errors:
            print(f"  • {err}", file=sys.stderr)
        sys.exit(1)

    tasks       = build_tasks(node_counts=node_counts, timeout=timeout,
                              base_seed=args.seed,
                              repetitions=args.repetitions,
                              hp2sat_strategies=hp2sat_strategies)
    results_dir = os.path.abspath(args.results_dir)
    cells_dir   = os.path.join(results_dir, "data")

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(cells_dir,   exist_ok=True)

    completed = _load_completed_seeds(cells_dir)
    if completed and not args.update:
        n_before = len(tasks)
        tasks    = [t for t in tasks if _task_seed(t) not in completed]
        n_skip   = n_before - len(tasks)
    else:
        n_skip = 0
    total = len(tasks)

    print(f"TrAC Benchmark")
    if n_skip:
        print(f"  resuming : {n_skip} already-completed tasks skipped")
    print(f"  tasks    : {total}  "
          f"({len(node_counts)} node counts × "
          f"{len(DENSITIES)} densities × "
          f"{args.repetitions} reps)")
    print(f"  nodes    : {min(node_counts)} – {max(node_counts)}  ({len(node_counts)} values)")
    print(f"  workers  : {args.workers}")
    print(f"  timeout  : {timeout} s/task")
    print(f"  base seed: {args.seed}")
    print(f"  hp2sat   : {', '.join(hp2sat_strategies) or '(none)'}")
    print(f"  cell CSVs: {cells_dir}/")
    print()
    _print_header()

    results: List[Dict] = []

    if args.workers <= 1:
        # Sequential path – simpler stack traces, easier to debug.
        for task in tasks:
            r = run_task(task)
            results.append(r)
            _print_row(r)
            _write_cell_row(r, cells_dir, update=args.update)
    else:
        # Parallel path via ProcessPoolExecutor.
        # Tasks complete in arbitrary order; rows stream to stdout as they land.
        # _write_cell_row is called in the main process (single-threaded here),
        # so per-cell file writes are sequential and race-free.
        wait_secs = (timeout + 30) if timeout else None
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_map = {executor.submit(run_task, t): t for t in tasks}
            try:
                for future in as_completed(future_map):
                    t = future_map[future]
                    try:
                        r = future.result(timeout=wait_secs)
                    except FuturesTimeout:
                        r = {
                            "n_nodes": t["n_nodes"], "density": t["density"],
                            "rep": t["rep"], "seed": -1,
                            "n_exo": -1, "n_endo": -1, "n_edges": -1,
                            "target": "?", "n_causes": -1,
                            "elapsed_s": timeout or 0.0,
                            "timed_out": True, "error": "future_timeout",
                        }
                    except Exception as exc:
                        r = {
                            "n_nodes": t["n_nodes"], "density": t["density"],
                            "rep": t["rep"], "seed": -1,
                            "n_exo": -1, "n_endo": -1, "n_edges": -1,
                            "target": "?", "n_causes": -1,
                            "elapsed_s": -1.0,
                            "timed_out": False, "error": repr(exc),
                        }
                    results.append(r)
                    _print_row(r)
                    _write_cell_row(r, cells_dir, update=args.update)
            except BrokenExecutor as exc:
                n_pending = sum(1 for f in future_map if not f.done())
                print(f"\nWORKER POOL CRASHED: {exc}", file=sys.stderr)
                print(f"Pending tasks abandoned: {n_pending}", file=sys.stderr)
                print("Tip: check stderr above for the last [worker] step marker to identify the crashing algorithm.", file=sys.stderr)
                print("OOM check: dmesg -T | grep -i -E 'killed process|out of memory'", file=sys.stderr)
                raise

    print()
    _save_config(args, results, os.path.join(results_dir, "benchmark_config.json"))
    print(f"  Cell CSVs → {cells_dir}/n*_d*.csv")
    print(f"\nTo regenerate the plot:")
    print(f"  python trac/plot_results.py --results-dir {cells_dir}")

    # ── Summary ───────────────────────────────────────────────────────────────
    done  = [r for r in results if not r.get("timed_out") and not r.get("error")]
    n_to  = sum(1 for r in results if r.get("timed_out"))
    n_err = sum(1 for r in results if r.get("error") and not r.get("timed_out"))
    print(f"\nCompleted: {len(done)}/{total}  "
          f"Timed-out: {n_to}  Errors: {n_err}")
    if done:
        times = [r["elapsed_s"] for r in done]
        print(f"Wall-clock  min={min(times):.4f}s  "
              f"max={max(times):.4f}s  "
              f"mean={sum(times)/len(times):.4f}s")


if __name__ == "__main__":
    main()
