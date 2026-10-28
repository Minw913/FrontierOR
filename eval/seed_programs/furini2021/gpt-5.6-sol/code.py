import argparse
import json
import math
import os
import sys
import time

from solution_logger import SolutionLogger

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception:
    gp = None
    GRB = None


class SearchTimeout(Exception):
    pass


class MaximumCliqueSolver:
    """Exact maximum-clique solver using bitsets and greedy coloring bounds."""

    def __init__(self, adjacency, deadline):
        self.adj = adjacency
        self.n = len(adjacency)
        self.deadline = deadline
        self.best = []
        self.current = []
        self.nodes = 0

    def _check_time(self):
        self.nodes += 1
        if (self.nodes & 1023) == 0 and time.monotonic() >= self.deadline:
            raise SearchTimeout

    def _initial_clique(self):
        if self.n == 0:
            return []

        best = []
        starts = sorted(
            range(self.n),
            key=lambda v: self.adj[v].bit_count(),
            reverse=True
        )[:min(self.n, 32)]

        for start in starts:
            clique = [start]
            candidates = self.adj[start]
            while candidates:
                best_v = -1
                best_score = -1
                q = candidates
                while q:
                    bit = q & -q
                    v = bit.bit_length() - 1
                    score = (self.adj[v] & candidates).bit_count()
                    if score > best_score:
                        best_score = score
                        best_v = v
                    q ^= bit
                clique.append(best_v)
                candidates &= self.adj[best_v]
            if len(clique) > len(best):
                best = clique

        if not best:
            best = [0]
        return best

    def _color_sort(self, vertices):
        order = []
        bounds = []
        remaining = vertices
        color = 0

        while remaining:
            color += 1
            available = remaining
            while available:
                bit = available & -available
                v = bit.bit_length() - 1
                order.append(v)
                bounds.append(color)
                remaining ^= bit
                available ^= bit
                available &= ~self.adj[v]

        return order, bounds

    def _expand(self, candidates):
        self._check_time()

        if not candidates:
            if len(self.current) > len(self.best):
                self.best = self.current.copy()
            return

        order, bounds = self._color_sort(candidates)

        for i in range(len(order) - 1, -1, -1):
            if len(self.current) + bounds[i] <= len(self.best):
                return

            v = order[i]
            bit = 1 << v
            if not (candidates & bit):
                continue

            self.current.append(v)
            new_candidates = candidates & self.adj[v]

            if new_candidates:
                self._expand(new_candidates)
            elif len(self.current) > len(self.best):
                self.best = self.current.copy()

            self.current.pop()
            candidates ^= bit

    def solve(self):
        if self.n == 0:
            return [], True

        self.best = self._initial_clique()
        all_vertices = (1 << self.n) - 1

        try:
            self._expand(all_vertices)
            return self.best, True
        except SearchTimeout:
            return self.best, False


def build_remaining_adjacency(base_adjacency, edges, deleted_indices):
    adjacency = base_adjacency.copy()
    for edge_index in deleted_indices:
        u, v = edges[edge_index]
        adjacency[u] &= ~(1 << v)
        adjacency[v] &= ~(1 << u)
    return adjacency


def exact_maximum_clique(base_adjacency, edges, deleted_indices, deadline):
    adjacency = build_remaining_adjacency(
        base_adjacency, edges, deleted_indices
    )
    solver = MaximumCliqueSolver(adjacency, deadline)
    clique, complete = solver.solve()
    return clique, complete, adjacency


def make_solution(objective, deleted_indices, edges, labels):
    interdicted = []
    for edge_index in sorted(deleted_indices):
        u, v = edges[edge_index]
        interdicted.append([labels[u], labels[v]])
    return {
        "objective_value": int(objective),
        "interdicted_edges": interdicted
    }


def turan_required_deletions(clique_size, target):
    """
    Minimum edges that must be removed from K_clique_size to make its
    clique number at most target, by Turan's theorem.
    """
    if clique_size <= target:
        return 0
    if target <= 0:
        return math.inf

    quotient, remainder = divmod(clique_size, target)
    return (
        remainder * ((quotient + 1) * quotient // 2)
        + (target - remainder) * (quotient * (quotient - 1) // 2)
    )


def select_heuristic_edge(clique, adjacency, edge_to_index, deleted):
    best_edge = None
    best_key = None

    for i in range(len(clique)):
        u = clique[i]
        for j in range(i + 1, len(clique)):
            v = clique[j]
            key_pair = (u, v) if u < v else (v, u)
            edge_index = edge_to_index.get(key_pair)
            if edge_index is None or edge_index in deleted:
                continue

            common = (adjacency[u] & adjacency[v]).bit_count()
            degree_sum = adjacency[u].bit_count() + adjacency[v].bit_count()
            score = (common, degree_sum, -edge_index)

            if best_key is None or score > best_key:
                best_key = score
                best_edge = edge_index

    return best_edge


def generate_clique_subsets(clique, subset_size, limit=16):
    """Generate several violated subsets of size subset_size."""
    clique = list(clique)
    size = len(clique)
    if size < subset_size:
        return []

    subsets = set()
    subsets.add(tuple(sorted(clique[:subset_size])))

    if size > subset_size:
        subsets.add(tuple(sorted(clique[-subset_size:])))

        max_starts = min(size, limit)
        for start in range(max_starts):
            chosen = [
                clique[(start + offset) % size]
                for offset in range(subset_size)
            ]
            subsets.add(tuple(sorted(chosen)))
            if len(subsets) >= limit:
                break

        base = clique[:subset_size]
        for replacement_pos in range(min(subset_size, 6)):
            for outside_pos in range(subset_size, min(size, subset_size + 6)):
                chosen = base.copy()
                chosen[replacement_pos] = clique[outside_pos]
                subsets.add(tuple(sorted(chosen)))
                if len(subsets) >= limit:
                    break
            if len(subsets) >= limit:
                break

    return list(subsets)


def cutting_plane_improvement(
    target,
    budget,
    edges,
    base_adjacency,
    edge_to_index,
    overall_deadline,
):
    """
    Attempts to find at most budget deletions producing clique number <= target.
    Returns:
      ("feasible", deletion_set, actual_clique)
      ("infeasible", None, None)
      ("timeout", None, None)
    """
    if gp is None or target < 1:
        return "timeout", None, None

    remaining_time = overall_deadline - time.monotonic()
    if remaining_time <= 0.05:
        return "timeout", None, None

    model = gp.Model()
    model.Params.OutputFlag = 0
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1

    x = model.addVars(len(edges), vtype=GRB.BINARY, name="delete")
    model.addConstr(gp.quicksum(x[i] for i in range(len(edges))) <= budget)
    model.setObjective(
        gp.quicksum(x[i] for i in range(len(edges))),
        GRB.MINIMIZE
    )

    known_cuts = set()

    while time.monotonic() < overall_deadline:
        remaining_time = overall_deadline - time.monotonic()
        if remaining_time <= 0.05:
            return "timeout", None, None

        model.Params.TimeLimit = max(0.01, remaining_time)
        model.optimize()

        if model.Status == GRB.INFEASIBLE:
            return "infeasible", None, None

        if model.SolCount == 0:
            return "timeout", None, None

        deleted = {
            i for i in range(len(edges))
            if x[i].X > 0.5
        }

        clique, complete, _ = exact_maximum_clique(
            base_adjacency,
            edges,
            deleted,
            overall_deadline
        )
        if not complete:
            return "timeout", None, None

        if len(clique) <= target:
            return "feasible", deleted, clique

        added = 0
        for subset in generate_clique_subsets(clique, target + 1):
            if subset in known_cuts:
                continue

            cut_edges = []
            valid = True
            for i in range(len(subset)):
                for j in range(i + 1, len(subset)):
                    u, v = subset[i], subset[j]
                    pair = (u, v) if u < v else (v, u)
                    edge_index = edge_to_index.get(pair)
                    if edge_index is None:
                        valid = False
                        break
                    cut_edges.append(edge_index)
                if not valid:
                    break

            if valid:
                model.addConstr(
                    gp.quicksum(x[e] for e in cut_edges) >= 1
                )
                known_cuts.add(subset)
                added += 1

        if added == 0:
            subset = tuple(sorted(clique[:target + 1]))
            cut_edges = []
            for i in range(len(subset)):
                for j in range(i + 1, len(subset)):
                    pair = tuple(sorted((subset[i], subset[j])))
                    cut_edges.append(edge_to_index[pair])
            model.addConstr(gp.quicksum(x[e] for e in cut_edges) >= 1)
            known_cuts.add(subset)

        model.update()

    return "timeout", None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    logger = (
        SolutionLogger(args.log_path, sense="minimize")
        if args.log_path else None
    )

    start_time = time.monotonic()
    time_limit = max(0, args.time_limit)
    overall_deadline = start_time + time_limit

    with open(args.instance_path, "r", encoding="utf-8") as f:
        instance = json.load(f)

    labels = list(instance["vertices"])
    n = len(labels)
    label_to_index = {label: i for i, label in enumerate(labels)}

    normalized_edges = set()
    for raw_edge in instance["edges"]:
        if len(raw_edge) != 2:
            continue
        a, b = raw_edge
        if a not in label_to_index or b not in label_to_index:
            continue
        u = label_to_index[a]
        v = label_to_index[b]
        if u == v:
            continue
        if u > v:
            u, v = v, u
        normalized_edges.add((u, v))

    edges = sorted(normalized_edges)
    m = len(edges)
    budget = max(0, min(int(instance["interdiction_budget_k"]), m))

    base_adjacency = [0] * n
    edge_to_index = {}
    for edge_index, (u, v) in enumerate(edges):
        base_adjacency[u] |= 1 << v
        base_adjacency[v] |= 1 << u
        edge_to_index[(u, v)] = edge_index

    if n == 0:
        solution = make_solution(0, set(), edges, labels)
        if logger:
            logger.log_solution(0, solution)
        os.makedirs(os.path.dirname(args.solution_path) or ".", exist_ok=True)
        with open(args.solution_path, "w", encoding="utf-8") as f:
            json.dump(solution, f)
        return

    if m == 0:
        solution = make_solution(1, set(), edges, labels)
        if logger:
            logger.log_solution(1, solution)
        os.makedirs(os.path.dirname(args.solution_path) or ".", exist_ok=True)
        with open(args.solution_path, "w", encoding="utf-8") as f:
            json.dump(solution, f)
        return

    if budget >= m:
        all_deleted = set(range(m))
        solution = make_solution(1, all_deleted, edges, labels)
        if logger:
            logger.log_solution(1, solution)
        os.makedirs(os.path.dirname(args.solution_path) or ".", exist_ok=True)
        with open(args.solution_path, "w", encoding="utf-8") as f:
            json.dump(solution, f)
        return

    # First compute the original clique number exactly. Most of the remaining
    # algorithm relies on exact clique separation.
    initial_clique, initial_complete, initial_adjacency = exact_maximum_clique(
        base_adjacency, edges, set(), overall_deadline
    )

    # If the time limit was exhausted during the initial solve, retain the
    # best clique found. Ordinarily the exact bitset search completes here.
    initial_objective = len(initial_clique)
    best_objective = initial_objective
    best_deleted = set()

    initial_solution = make_solution(
        best_objective, best_deleted, edges, labels
    )
    if logger:
        logger.log_solution(best_objective, initial_solution)

    if not initial_complete or time.monotonic() >= overall_deadline:
        os.makedirs(os.path.dirname(args.solution_path) or ".", exist_ok=True)
        with open(args.solution_path, "w", encoding="utf-8") as f:
            json.dump(initial_solution, f)
        return

    if budget == 0 or best_objective <= 1:
        os.makedirs(os.path.dirname(args.solution_path) or ".", exist_ok=True)
        with open(args.solution_path, "w", encoding="utf-8") as f:
            json.dump(initial_solution, f)
        return

    # Greedy phase: repeatedly destroy a currently maximum clique, preferring
    # edges with many common neighbors.
    heuristic_deadline = min(
        overall_deadline,
        start_time + 0.30 * time_limit
    )
    current_deleted = set()
    current_clique = initial_clique
    current_adjacency = initial_adjacency

    while (
        len(current_deleted) < budget
        and len(current_clique) > 1
        and time.monotonic() < heuristic_deadline
    ):
        chosen_edge = select_heuristic_edge(
            current_clique,
            current_adjacency,
            edge_to_index,
            current_deleted
        )
        if chosen_edge is None:
            break

        trial_deleted = current_deleted | {chosen_edge}
        clique, complete, adjacency = exact_maximum_clique(
            base_adjacency,
            edges,
            trial_deleted,
            heuristic_deadline
        )
        if not complete:
            break

        current_deleted = trial_deleted
        current_clique = clique
        current_adjacency = adjacency

        if len(clique) < best_objective:
            best_objective = len(clique)
            best_deleted = current_deleted.copy()
            solution = make_solution(
                best_objective, best_deleted, edges, labels
            )
            if logger:
                logger.log_solution(best_objective, solution)

        if best_objective <= 1:
            break

    # Exact cutting-plane phase. For a target t, every (t+1)-clique must
    # contain at least one interdicted edge.
    if gp is not None and best_objective > 1:
        while time.monotonic() < overall_deadline:
            target = best_objective - 1
            if target < 1:
                break

            # A maximum clique of the original graph alone yields a useful
            # Turan-theorem lower bound on the required deletion count.
            required = turan_required_deletions(
                initial_objective, target
            )
            if required > budget:
                break

            status, deleted, remaining_clique = cutting_plane_improvement(
                target,
                budget,
                edges,
                base_adjacency,
                edge_to_index,
                overall_deadline,
            )

            if status == "infeasible":
                break
            if status != "feasible":
                break

            actual_objective = len(remaining_clique)
            if actual_objective >= best_objective:
                break

            best_objective = actual_objective
            best_deleted = deleted
            solution = make_solution(
                best_objective, best_deleted, edges, labels
            )
            if logger:
                logger.log_solution(best_objective, solution)

            if best_objective <= 1:
                break

    final_solution = make_solution(
        best_objective, best_deleted, edges, labels
    )
    os.makedirs(os.path.dirname(args.solution_path) or ".", exist_ok=True)
    with open(args.solution_path, "w", encoding="utf-8") as f:
        json.dump(final_solution, f)


if __name__ == "__main__":
    sys.setrecursionlimit(1_000_000)
    main()