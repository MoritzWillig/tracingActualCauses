"""
trac.py  –  Tracing Actual Causes with Counterfactual Witness Maps
==================================================================
Sound-and-complete implementation of Algorithm 1 (TrAC) from:

    "Tracing Actual Causes with Counterfactual Witness Maps"
    Submitted to UAI 2026.

Components implemented
----------------------
  Def 1 – Parent targets         : structural preimage filtering
  Def 2 – Descendant targets     : recursive backward propagation
  Def 3 – Global targets         : intersection over exogenous roots
  Theorem 1 – Witness target characterisation  (validated by construction)
  Corollary 2 – Witness existence via membership test
  Def 4 – Ancestor intervention grammar  G_M(Y)
  Def 5 – Ancestral map          A_M(Y)
  Proposition 3 – Grammar–graph equivalence  (child formula)
  Proposition 4 – Ancestor reachability  (used in completeness)
  Def 6 – Witness map            W_M(Y ∈ Ŷ)
  Lemma 5 – AC-witness identification  (soundness)
  Theorem 6 – Sufficiency of witness map  (completeness)
  Algorithm 1 – TrAC
  Remark 5 – Phase-2 pruning strategies (i) size-first, (ii) U-precheck,
             (iii) superset pruning
  Chockler & Halpern responsibility scores

Worked examples (reproduce paper figures)
------------------------------------------
  • Forest-fire  – overdetermination (OR) and conjunctive causation (AND)
  • Billy-and-Suzy – preemption, non-empty contingency sets

Usage
-----
    python trac.py                    # run all examples
    python trac.py --example bs       # Billy-and-Suzy only
    python trac.py --example ff-or    # Forest-fire OR
    python trac.py --example ff-and   # Forest-fire AND

Dependencies: networkx  (pip install networkx)
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from itertools import chain, combinations, product as iterproduct
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple

import networkx as nx

__all__ = ["SCM", "trac", "TrACResult"]


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _nonempty_subsets(iterable):
    """All non-empty subsets, in non-decreasing size order."""
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))


# ══════════════════════════════════════════════════════════════════════════════
# Structural Causal Model
# ══════════════════════════════════════════════════════════════════════════════

class SCM:
    """
    Markovian Structural Causal Model  M = ⟨U, V, F, P_U⟩.

    Attributes
    ----------
    exo_vars  : U  – exogenous root variables (no structural equations).
    endo_vars : V  – endogenous variables; each X ∈ V has F_X : X_{pa(X)} → X_X.
    domains   : variable name → set of possible values.

    Parameters for F_X are passed as a dict  {parent_name: value}.
    """

    def __init__(self) -> None:
        self.exo_vars:  List[str] = []
        self.endo_vars: List[str] = []
        self.domains:   Dict[str, Set] = {}
        self._parents:  Dict[str, List[str]] = {}
        self.equations: Dict[str, Callable] = {}
        self._graph:    nx.DiGraph = nx.DiGraph()

    # ── Construction ──────────────────────────────────────────────────────────

    def add_exogenous(self, var: str, domain) -> "SCM":
        """Register an exogenous variable."""
        self.exo_vars.append(var)
        self.domains[var] = set(domain)
        self._graph.add_node(var)
        return self

    def add_endogenous(
        self,
        var: str,
        parents: List[str],
        equation: Callable[..., Any],
        domain: Optional[Set] = None,
    ) -> "SCM":
        """
        Register an endogenous variable.

        Parameters
        ----------
        parents  : list of parent variable names (may be exo or endo)
        equation : pa_dict → value  (pa_dict maps parent name → value)
        domain   : value domain; auto-inferred from parents if None
        """
        self.endo_vars.append(var)
        self._parents[var] = list(parents)
        self.equations[var] = equation
        self.domains[var] = set(domain) if domain is not None \
            else self._infer_domain(parents, equation)
        for p in parents:
            self._graph.add_edge(p, var)
        self._graph.add_node(var)
        return self

    def _infer_domain(self, parents: List[str], eq: Callable) -> Set:
        if not parents:
            return set()
        return {
            eq(dict(zip(parents, vals)))
            for vals in iterproduct(*[list(self.domains[p]) for p in parents])
        }

    # ── Graph accessors ───────────────────────────────────────────────────────

    def parents_of(self, var: str) -> List[str]:
        return self._parents.get(var, [])

    def endo_parents_of(self, var: str) -> List[str]:
        """pa^V_M(X) – endogenous parents only."""
        return [p for p in self._parents.get(var, []) if p in self.endo_vars]

    def endo_ancestors_of(self, var: str) -> Set[str]:
        """anc^V_M(Y) – endogenous ancestors of *var* (not including *var*)."""
        anc: Set[str] = set()
        q, seen = deque([var]), {var}
        while q:
            v = q.popleft()
            for p in self._graph.predecessors(v):
                if p not in seen:
                    seen.add(p)
                    if p in self.endo_vars:
                        anc.add(p)
                    q.append(p)
        return anc

    def topo_order(self) -> List[str]:
        return list(nx.topological_sort(self._graph))

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        u: Dict[str, Any],
        interventions: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate M_{do(interventions)} under environment u.

        Returns the complete world assignment w : (U ∪ V) → values.
        Intervened variables bypass their structural equations.
        """
        ivs = interventions or {}
        world: Dict[str, Any] = dict(u)
        for var in self.topo_order():
            if var in self.exo_vars:
                continue
            if var in ivs:
                world[var] = ivs[var]
            else:
                pa = self._parents.get(var, [])
                world[var] = self.equations[var]({p: world[p] for p in pa})
        return world

    def actual_world(self, u: Dict) -> Dict:
        """Actual world w_u induced by environment u (no interventions)."""
        return self.evaluate(u)

    def __repr__(self) -> str:
        return (
            f"SCM(exo={self.exo_vars}, endo={self.endo_vars}, "
            f"|edges|={self._graph.number_of_edges()})"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Global Targets  (Definitions 1 – 3, Theorem 1, Corollary 2)
# ══════════════════════════════════════════════════════════════════════════════

def compute_global_targets(
    scm: SCM,
    S: Set[str],
    Y: str,
    Y_hat: Set,
) -> Set[FrozenSet]:
    """
    Compute  X̂_{M,do(S)}(Ŷ)  (Definition 3).

    Enumerates all environments u ∈ dom(U) and all intervention values
    s ∈ ×_{X∈S} dom(X), collecting every complete configuration
    x^{S=s}_u of M_{do(S=s)} that satisfies  Y ∈ Ŷ.

    This direct enumeration realises the backward-propagation characterisation
    of Theorem 1 exactly (no false positives / negatives).

    The resulting set supports the membership test of Corollary 2:
        (u, w_u(Z)) ∈ X̂_{M,do(S)}(Ŷ) ↾ (U × Z)
        ⟺  ∃ x' ∈ X_X  s.t.  M,u ⊨ [X=x', Z=w_u(Z)](Y ∈ Ŷ)

    Parameters
    ----------
    S     : intervention set (structural equations overridden)
    Y     : target variable
    Y_hat : complement event  Ŷ = dom(Y) \\ Y*
    """
    targets: Set[FrozenSet] = set()

    exo_list = scm.exo_vars
    exo_doms = [list(scm.domains[v]) for v in exo_list]
    S_list   = sorted(S)
    S_doms   = [list(scm.domains[v]) for v in S_list]

    exo_product = iterproduct(*exo_doms) if exo_doms else [()]
    for exo_vals in exo_product:
        u = dict(zip(exo_list, exo_vals))
        s_product = iterproduct(*S_doms) if S_doms else [()]
        for s_vals in s_product:
            s_dict = dict(zip(S_list, s_vals))
            world  = scm.evaluate(u, s_dict)
            if world[Y] in Y_hat:
                targets.add(frozenset(world.items()))

    return targets


def _check_U_projection(
    u: Dict,
    targets: Set[FrozenSet],
    exo_vars: List[str],
) -> bool:
    """
    Node-level pre-check (Remark 5 ii):
        u  ∈  X̂_{M,do(S)}(Ŷ) ↾ U
    i.e. ∃ config ∈ targets agreeing with u on every exogenous variable.
    Failure is a sufficient condition to skip the entire node S.
    """
    for cfg in targets:
        cfg_d = dict(cfg)
        if all(cfg_d.get(v) == u[v] for v in exo_vars):
            return True
    return False


def _check_UZ_membership(
    u: Dict,
    wu_Z: Dict,
    Z: Set[str],
    targets: Set[FrozenSet],
    exo_vars: List[str],
) -> bool:
    """
    AC2 membership test (Corollary 2):
        (u, w_u(Z))  ∈  X̂_{M,do(S)}(Ŷ) ↾ (U × Z)
    i.e. ∃ config ∈ targets agreeing with u on U and with w_u(Z) on Z.
    """
    for cfg in targets:
        cfg_d = dict(cfg)
        if (all(cfg_d.get(v) == u[v] for v in exo_vars) and
                all(cfg_d.get(v) == wu_Z[v] for v in Z)):
            return True
    return False


# ══════════════════════════════════════════════════════════════════════════════
# Ancestral Map  (Definitions 4 – 5, Propositions 3 – 4)
# ══════════════════════════════════════════════════════════════════════════════

def build_ancestral_map(
    scm: SCM,
    Y: str,
) -> Tuple[FrozenSet, Dict[FrozenSet, Set[FrozenSet]]]:
    """
    Build  A_M(Y)  (Definition 5) via the grammar  G_M(Y)  (Definition 4).

    The grammar's alphabet is  Σ_M(Y) = anc^V_M(Y) ∪ {Y}.
    Productions (Definition 4):
        (P_keep)   X  →  {X}
        (P_unfold) X  →  pa^V_M(X)          (contributes ∅ if X is a source)

    Children are computed via Proposition 3 (closed form):
        ch_A(Z) = ∪_{∅≠Z'⊆Z}  P( pa^V_M(Z') ∪ (Z\\Z') )  \\  {∅}
    where Z' is the unfolded part and Z\\Z' the kept part.

    Termination is guaranteed by acyclicity of the Markovian DAG (Remark 3).

    Returns
    -------
    root : frozenset  –  root node {Y}
    adj  : dict[node → set[children]]  –  directed edges of A_M(Y)
    """
    sigma = scm.endo_ancestors_of(Y) | {Y}          # Σ_M(Y)
    root  = frozenset({Y})

    adj:     Dict[FrozenSet, Set[FrozenSet]] = defaultdict(set)
    visited: Set[FrozenSet] = set()
    queue:   deque = deque([root])

    while queue:
        Z_node = queue.popleft()
        if Z_node in visited:
            continue
        visited.add(Z_node)

        Z_list   = list(Z_node)
        children: Set[FrozenSet] = set()

        # Proposition 3: iterate over all non-empty Z' ⊆ Z  (= unfolded part)
        for r in range(1, len(Z_list) + 1):
            for Z_prime in combinations(Z_list, r):
                Z_prime_set = set(Z_prime)
                remain      = Z_node - Z_prime_set          # Z \ Z'  (kept)

                # pa^V_M(Z') = ∪_{X ∈ Z'} endo_parents(X)
                pa_union: Set[str] = set()
                for X in Z_prime_set:
                    pa_union |= set(scm.endo_parents_of(X))

                # candidate set restricted to Σ_M(Y)
                candidate = (pa_union | remain) & sigma

                # all non-empty subsets of candidate become children
                cand_list = sorted(candidate)
                for size in range(1, len(cand_list) + 1):
                    for child_tup in combinations(cand_list, size):
                        child = frozenset(child_tup)
                        if child != root:           # root cannot be a child (DAG)
                            children.add(child)

        adj[Z_node] = children
        for child in children:
            if child not in visited:
                queue.append(child)

    return root, dict(adj)


# ══════════════════════════════════════════════════════════════════════════════
# Result Container
# ══════════════════════════════════════════════════════════════════════════════

class TrACResult:
    """
    Output of a TrAC run.

    Attributes
    ----------
    causes        : minimal actual-cause variable sets (frozensets of names)
    contingencies : X → witnessing contingency set Z  (or None)
    wu            : actual world assignment
    Y, Y_star, Y_hat : target specification

    Pruning statistics (Remark 5)
    ------------------------------
    n_candidates          : non-root nodes in the ancestral map  (= search space)
    n_skipped_superset    : nodes skipped by strategy (iii) at node level
    n_skipped_uprojection : nodes skipped by strategy (ii)  at node level
    n_partitions_skipped  : (X,Z) pairs skipped by strategy (iii) within visited nodes
    n_partitions_tested   : (X,Z) pairs that reached the membership test
    """

    def __init__(
        self,
        causes: List[FrozenSet],
        contingencies: Dict[FrozenSet, Set[str]],
        wu: Dict,
        Y: str,
        Y_star: Set,
        Y_hat: Set,
        *,
        n_candidates:          int = 0,
        n_skipped_superset:    int = 0,
        n_skipped_uprojection: int = 0,
        n_partitions_skipped:  int = 0,
        n_partitions_tested:   int = 0,
    ) -> None:
        self.causes        = causes
        self.contingencies = contingencies
        self.wu            = wu
        self.Y             = Y
        self.Y_star        = Y_star
        self.Y_hat         = Y_hat
        self.n_candidates          = n_candidates
        self.n_skipped_superset    = n_skipped_superset
        self.n_skipped_uprojection = n_skipped_uprojection
        self.n_partitions_skipped  = n_partitions_skipped
        self.n_partitions_tested   = n_partitions_tested

    def responsibility(self, X: FrozenSet) -> Optional[float]:
        """
        Responsibility score  ρ(X) = 1 / (|X| + |Z|)
        following Chockler & Halpern (2004, Def. 3.1).
        """
        Z = self.contingencies.get(X)
        return None if Z is None else 1.0 / (len(X) + len(Z))

    def summary(self) -> str:
        w = 62
        lines = [
            "═" * w,
            f"  TrAC Result",
            f"  Target  : {self.Y} ∈ {self.Y_star}   (Ŷ = {self.Y_hat})",
            f"  Env. u  : { {k: v for k, v in self.wu.items() if k.startswith('U')} }",
            f"  World wᵤ: { {k: v for k, v in self.wu.items() if not k.startswith('U')} }",
            "─" * w,
        ]
        if not self.causes:
            lines.append("  No actual causes found.")
        else:
            for X in self.causes:
                vals  = {v: self.wu[v] for v in sorted(X)}
                Z     = self.contingencies.get(X) or set()
                z_vals = {v: self.wu[v] for v in sorted(Z)}
                rho   = self.responsibility(X)
                rho_s = f"{rho:.3f}" if rho is not None else "n/a"
                ac_type = "but-for cause" if not Z else "actual cause"
                lines.append(
                    f"  {ac_type:14s}  X = {vals}   Z = {z_vals}   ρ = {rho_s}"
                )
        visited = (self.n_candidates
                   - self.n_skipped_superset
                   - self.n_skipped_uprojection)
        n_part_total = self.n_partitions_skipped + self.n_partitions_tested
        lines.append(
            f"  Pruning : {self.n_candidates} candidates | "
            f"skipped {self.n_skipped_superset} (superset) + "
            f"{self.n_skipped_uprojection} (U-proj) nodes, "
            f"{visited} visited | "
            f"{self.n_partitions_skipped}/{n_part_total} partitions skipped"
        )
        lines.append("═" * w)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


# ══════════════════════════════════════════════════════════════════════════════
# TrAC Algorithm  (Algorithm 1)
# ══════════════════════════════════════════════════════════════════════════════

def trac(
    scm: SCM,
    u: Dict[str, Any],
    Y: str,
    Y_star: Set,
    verbose: bool = False,
) -> TrACResult:
    """
    TrAC – Tracing Actual Causes  (Algorithm 1).

    Enumerates *every* actual cause (X = w_u(X)) of (Y ∈ Y_star) in the
    Markovian SCM M under environment u.

    Correctness guarantees (Theorem 7)
    ------------------------------------
    Soundness    : every returned X satisfies AC1–AC3  (Lemma 5).
    Completeness : every actual cause is returned  (Theorem 6).

    Phase-2 pruning (Remark 5, all three strategies)
    -------------------------------------------------
    (i)  Size-first traversal – confirms smallest causes earliest,
         maximising the reach of strategy (iii).
    (ii) Node-level U-projection pre-check – skips node S entirely if
         u ∉ X̂_{M,do(S)}(Ŷ) ↾ U  (necessary condition for any witness).
    (iii) Superset pruning – skips X ⊆ S whenever a confirmed cause X' ⊊ X
         already exists in C, maintaining C as an antichain  (AC3).

    Parameters
    ----------
    scm    : Markovian SCM
    u      : complete assignment of all exogenous variables
    Y      : target (endogenous) variable
    Y_star : target event  (set of values Y must take)
    verbose: print step-by-step trace to stdout

    Returns
    -------
    TrACResult containing all actual causes with one witnessing (Z, X, x')
    per cause and Chockler–Halpern responsibility scores.
    """
    Y_hat = scm.domains[Y] - set(Y_star)
    wu    = scm.actual_world(u)

    def log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    log("\n" + "═" * 62)
    log(f"  TrAC  |  {Y} ∈ {Y_star}   Ŷ = {Y_hat}")
    log(f"  Actual world: {wu}")

    empty = TrACResult([], {}, wu, Y, Y_star, Y_hat)

    # ── AC1: (Y ∈ Y*) holds in actual world ──────────────────────────────────
    if wu[Y] not in Y_star:
        log(f"  AC1 FAILS: {Y} = {wu[Y]} ∉ {Y_star}")
        return empty

    # ── Phase 1: Observational global targets ─────────────────────────────────
    # Check that the actual environment is incompatible with Ŷ observationally
    # (if u already witnesses Y ∈ Ŷ, there are no causes to find)
    obs_targets = compute_global_targets(scm, set(), Y, Y_hat)
    log(f"\n  Phase 1 – |X̂_obs(Ŷ)| = {len(obs_targets)}")

    if _check_U_projection(u, obs_targets, scm.exo_vars):
        log("  u satisfies Y ∈ Ŷ observationally – no causes.")
        return empty

    # ── Build Ancestral Map A_M(Y) ────────────────────────────────────────────
    root, adj = build_ancestral_map(scm, Y)

    all_nodes: Set[FrozenSet] = set(adj.keys())
    for ch_set in adj.values():
        all_nodes |= ch_set
    non_root = sorted(all_nodes - {root}, key=len)       # strategy (i): size-first

    log(f"\n  Ancestral map: {len(all_nodes)} nodes total, "
        f"{len(non_root)} non-root candidates")
    if verbose:
        log(f"  Nodes (sorted): {[set(n) for n in non_root]}")

    # ── Phase 2: Traverse Witness Map ────────────────────────────────────────
    C:              List[FrozenSet]              = []
    contingencies:  Dict[FrozenSet, Set[str]]    = {}

    _n_skip_sup:   int = 0   # nodes skipped by strategy (iii)
    _n_skip_up:    int = 0   # nodes skipped by strategy (ii)
    _n_part_skip:  int = 0   # (X,Z) pairs skipped by strategy (iii) within nodes
    _n_part_test:  int = 0   # (X,Z) pairs reaching the membership test

    for S_frozen in non_root:
        S = set(S_frozen)

        # Strategy (iii): skip S entirely if a confirmed cause X' ⊊ S exists
        if any(confirmed < S_frozen for confirmed in C):
            log(f"  [prune-iii] {set(S_frozen)}")
            _n_skip_sup += 1
            continue

        # Compute X̂_{M,do(S)}(Ŷ)
        targets_S = compute_global_targets(scm, S, Y, Y_hat)

        # Strategy (ii): node-level U-projection pre-check
        if not _check_U_projection(u, targets_S, scm.exo_vars):
            log(f"  [prune-ii ] {set(S_frozen)}  (U-projection empty)")
            _n_skip_up += 1
            continue

        log(f"\n  Node S = {set(S_frozen)}   |X̂_do(S)(Ŷ)| = {len(targets_S)}")

        # Enumerate all non-empty X ⊆ S  (candidate cause),  Z = S \\ X
        for r in range(1, len(S) + 1):
            for X_tup in combinations(sorted(S), r):
                X_frozen = frozenset(X_tup)

                # Strategy (iii) at subset level
                if any(confirmed < X_frozen for confirmed in C):
                    _n_part_skip += 1
                    continue

                Z_frozen = S_frozen - X_frozen
                Z_set    = set(Z_frozen)
                wu_Z     = {v: wu[v] for v in Z_set}

                # Corollary 2 membership test
                _n_part_test += 1
                if _check_UZ_membership(u, wu_Z, Z_set, targets_S, scm.exo_vars):
                    x_vals = {v: wu[v] for v in sorted(X_tup)}
                    log(f"    ✓  X={x_vals}  Z={wu_Z}  → candidate")
                    if X_frozen not in C:
                        C.append(X_frozen)
                        contingencies[X_frozen] = Z_set

    # ── Phase 3: Minimality filter (AC3) ─────────────────────────────────────
    # (largely redundant with strategies (i)+(iii), but kept for formal
    #  correctness per the algorithm statement)
    minimal = [x for x in C if not any(other < x for other in C)]

    log(f"\n  Phase 3 – {len(minimal)} minimal actual cause(s).")
    return TrACResult(
        minimal, contingencies, wu, Y, Y_star, Y_hat,
        n_candidates=len(non_root),
        n_skipped_superset=_n_skip_sup,
        n_skipped_uprojection=_n_skip_up,
        n_partitions_skipped=_n_part_skip,
        n_partitions_tested=_n_part_test,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Built-in Examples
# ══════════════════════════════════════════════════════════════════════════════

def make_forest_fire_scm(mode: str = "or") -> SCM:
    """
    Forest-fire SCM  (Appendix B.1, Figure 3).

    Shared DAG:  L → Y ← A,  with exogenous roots U_L, U_A.
    Equations:   F_L = U_L,   F_A = U_A.

    mode = 'or'  : F_Y = L ∨ A  (symmetric overdetermination)
    mode = 'and' : F_Y = L ∧ A  (conjunctive causation)
    """
    assert mode in ("or", "and"), "mode must be 'or' or 'and'"
    scm = SCM()
    scm.add_exogenous("UL", {0, 1})
    scm.add_exogenous("UA", {0, 1})
    scm.add_endogenous("L", ["UL"], lambda p: p["UL"])
    scm.add_endogenous("A", ["UA"], lambda p: p["UA"])
    if mode == "or":
        scm.add_endogenous("Y", ["L", "A"], lambda p: int(p["L"] or  p["A"]))
    else:
        scm.add_endogenous("Y", ["L", "A"], lambda p: int(p["L"] and p["A"]))
    return scm


def make_billy_suzy_scm() -> SCM:
    """
    Billy-and-Suzy preemption SCM  (Appendix B.2, Figure 1).

    DAG:  UST → ST → SH → BS ← BH ← BT ← UBT
                              ↑
                             SH  (BH = BT ∧ ¬SH)

    Equations:
        F_ST  = U_ST
        F_BT  = U_BT
        F_SH  = ST
        F_BH  = BT ∧ ¬SH
        F_BS  = SH ∨ BH
    """
    scm = SCM()
    scm.add_exogenous("UST", {0, 1})
    scm.add_exogenous("UBT", {0, 1})
    scm.add_endogenous("ST",  ["UST"],       lambda p: p["UST"])
    scm.add_endogenous("BT",  ["UBT"],       lambda p: p["UBT"])
    scm.add_endogenous("SH",  ["ST"],        lambda p: p["ST"])
    scm.add_endogenous("BH",  ["BT", "SH"],  lambda p: int(p["BT"] and not p["SH"]))
    scm.add_endogenous("BS",  ["SH", "BH"],  lambda p: int(p["SH"] or  p["BH"]))
    return scm


def run_forest_fire_or(verbose: bool = False) -> None:
    """
    Case I: Overdetermination  F_Y = L ∨ A.
    Expected: unique joint actual cause  (L=1, A=1),  ρ = 1/2.
    """
    print("\n" + "▓" * 62)
    print("  FOREST FIRE  –  F_Y = L ∨ A  (overdetermination)")
    print("▓" * 62)
    scm = make_forest_fire_scm("or")
    u   = {"UL": 1, "UA": 1}
    res = trac(scm, u, "Y", {1}, verbose=verbose)
    print(res)


def run_forest_fire_and(verbose: bool = False) -> None:
    """
    Case II: Conjunctive causation  F_Y = L ∧ A.
    Expected: two singleton but-for causes  (L=1) and (A=1),  ρ = 1 each.
    """
    print("\n" + "▓" * 62)
    print("  FOREST FIRE  –  F_Y = L ∧ A  (conjunctive causation)")
    print("▓" * 62)
    scm = make_forest_fire_scm("and")
    u   = {"UL": 1, "UA": 1}
    res = trac(scm, u, "Y", {1}, verbose=verbose)
    print(res)


def run_billy_suzy(verbose: bool = False) -> None:
    """
    Billy-and-Suzy preemption.
    Expected actual causes: (ST=1) and (SH=1), each with Z={BH}, ρ = 1/2.
    Expected but-for causes (non-minimal): {ST,BT}, {ST,BH}, {SH,BT}, {SH,BH}.
    """
    print("\n" + "▓" * 62)
    print("  BILLY-AND-SUZY  –  preemption")
    print("▓" * 62)
    scm = make_billy_suzy_scm()
    u   = {"UST": 1, "UBT": 1}
    res = trac(scm, u, "BS", {1}, verbose=verbose)
    print(res)


# ══════════════════════════════════════════════════════════════════════════════
# Utility: Enumerate Ancestral Map  (for inspection)
# ══════════════════════════════════════════════════════════════════════════════

def print_ancestral_map(scm: SCM, Y: str) -> None:
    """Pretty-print the full ancestral map A_M(Y)."""
    root, adj = build_ancestral_map(scm, Y)
    all_nodes: Set[FrozenSet] = set(adj.keys())
    for ch in adj.values():
        all_nodes |= ch

    print(f"\nAncestral map  A_M({Y})")
    print(f"  Σ_M({Y}) = {scm.endo_ancestors_of(Y) | {Y}}")
    print(f"  Nodes ({len(all_nodes)}):  {sorted([set(n) for n in all_nodes], key=len)}")
    print(f"  Edges:")
    for node in sorted(adj, key=len):
        children = sorted([set(c) for c in adj[node]], key=len)
        if children:
            print(f"    {str(set(node)):30s} →  {children}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="TrAC – Tracing Actual Causes (Algorithm 1, UAI 2026)"
    )
    parser.add_argument(
        "--example", choices=["bs", "ff-or", "ff-and", "all"],
        default="all",
        help="Which example to run (default: all)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print step-by-step trace",
    )
    parser.add_argument(
        "--map", action="store_true",
        help="Print ancestral maps",
    )
    args = parser.parse_args()

    if args.map:
        print("\n=== Ancestral Maps ===")
        print_ancestral_map(make_forest_fire_scm("or"),  "Y")
        print_ancestral_map(make_billy_suzy_scm(),        "BS")

    if args.example in ("ff-or",  "all"):
        run_forest_fire_or(verbose=args.verbose)
    if args.example in ("ff-and", "all"):
        run_forest_fire_and(verbose=args.verbose)
    if args.example in ("bs",     "all"):
        run_billy_suzy(verbose=args.verbose)


if __name__ == "__main__":
    main()
