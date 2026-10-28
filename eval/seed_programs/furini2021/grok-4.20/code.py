import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
import sys
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    n = instance['num_vertices']
    edges = instance['edges']
    k = instance['interdiction_budget_k']
    vertices = instance['vertices']

    # Map vertices to 0-based indices if needed
    vertex_to_idx = {v: i for i, v in enumerate(vertices)}
    edge_list = []
    for u, v in edges:
        i = vertex_to_idx[u]
        j = vertex_to_idx[v]
        if i > j:
            i, j = j, i
        edge_list.append((i, j))
    edge_list = list(set(edge_list))  # remove duplicates if any
    m = len(edge_list)

    # Build adjacency
    adj = [set() for _ in range(n)]
    edge_idx = {}
    for e_idx, (i, j) in enumerate(edge_list):
        adj[i].add(j)
        adj[j].add(i)
        edge_idx[(i, j)] = e_idx
        edge_idx[(j, i)] = e_idx

    # Upper bound on omega: simple greedy
    def greedy_clique_size():
        deg = [len(adj[i]) for i in range(n)]
        order = sorted(range(n), key=lambda x: -deg[x])
        max_c = 1
        for i in range(n):
            cand = [order[i]]
            for j in range(i+1, n):
                v = order[j]
                if all(v in adj[u] for u in cand):
                    cand.append(v)
            max_c = max(max_c, len(cand))
        return max_c

    ub = greedy_clique_size()
    lb = 1
    best_obj = ub
    best_interdicted = []

    start_time = time.time()
    max_time = args.time_limit - 5  # safety margin

    # Try to find small clique cover / coloring based upper bounds on interdiction
    # But main approach: binary search on target omega = t, solve interdiction MIP for each t

    def can_interdict_to_at_most(t, remaining_time):
        if time.time() - start_time > max_time:
            return False
        model = gp.Model("clique_interdiction")
        model.Params.TimeLimit = remaining_time
        model.Params.Threads = 1
        model.Params.LogToConsole = 0
        model.Params.MIPFocus = 1
        model.Params.Heuristics = 0.2

        # x_v = 1 if v in the potential clique
        x = model.addVars(n, vtype=GRB.BINARY, name="x")
        # y_e = 1 if edge e is interdicted
        y = model.addVars(m, vtype=GRB.BINARY, name="y")

        # Objective: force a feasible clique of size t+1 if not fully interdicted
        model.setObjective(0, GRB.MINIMIZE)

        # Budget
        model.addConstr(gp.quicksum(y[i] for i in range(m)) <= k, "budget")

        # For every pair of vertices that could be in a clique, if both selected then edge must be removed or not exist
        for i in range(n):
            for j in range(i+1, n):
                if j in adj[i]:
                    eidx = edge_idx[(i, j)]
                    model.addConstr(x[i] + x[j] - y[eidx] <= 1, f"clique_{i}_{j}")
                else:
                    # no edge, cannot both be in clique
                    model.addConstr(x[i] + x[j] <= 1, f"noedge_{i}_{j}")

        # Force a clique of size at least t+1
        model.addConstr(gp.quicksum(x[i] for i in range(n)) >= t + 1, "force_clique")

        model.optimize()

        if model.status == GRB.INFEASIBLE or (model.status == GRB.OPTIMAL and model.objVal > 0.5):
            # Even with k removals we cannot have a clique of size t+1
            return True
        else:
            # We could find a clique of size t+1 even after optimal interdiction
            return False

    # Binary search on the objective t
    current_best = best_obj
    for t in range(ub - 1, 0, -1):
        elapsed = time.time() - start_time
        if elapsed > max_time:
            break
        remain = max_time - elapsed + 1
        if can_interdict_to_at_most(t, remain):
            current_best = t
            if logger:
                logger.log(current_best)
            best_obj = t
            # We do not yet have the actual interdicted set. We need to recover one.
            # Re-solve the last successful model with a slightly different formulation to get the interdicted edges.
            # For simplicity we record the best t so far and solve one final model at the end for the actual interdiction set.
        else:
            # Cannot reduce to t, so the minimal achievable is at least t+1
            break

    # Now solve a final model to get the actual interdicted edges that achieve best_obj
    final_t = best_obj
    elapsed = time.time() - start_time
    if elapsed < max_time:
        model = gp.Model("recover_interdiction")
        model.Params.TimeLimit = max_time - elapsed + 2
        model.Params.Threads = 1
        model.Params.LogToConsole = 0

        x = model.addVars(n, vtype=GRB.BINARY, name="x")
        y = model.addVars(m, vtype=GRB.BINARY, name="y")

        model.setObjective(gp.quicksum(y[i] for i in range(m)), GRB.MINIMIZE)

        model.addConstr(gp.quicksum(y[i] for i in range(m)) <= k, "budget")

        # No clique larger than final_t
        for i in range(n):
            for j in range(i + 1, n):
                if j in adj[i]:
                    eidx = edge_idx[(i, j)]
                    model.addConstr(x[i] + x[j] - y[eidx] <= 1)
                else:
                    model.addConstr(x[i] + x[j] <= 1)

        model.addConstr(gp.quicksum(x[i] for i in range(n)) <= final_t, "clique_size")

        # Add a small incentive to maximize the covered clique size (helps find tight solutions)
        model.setObjective(gp.quicksum(y[i] for i in range(m)) - 0.001 * gp.quicksum(x[i] for i in range(n)), GRB.MINIMIZE)

        model.optimize()

        interdicted = []
        if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT) and model.SolCount > 0:
            for i in range(m):
                if y[i].X > 0.5:
                    u, v = edge_list[i]
                    interdicted.append([vertices[u], vertices[v]])
            # update best if we got better, unlikely
            if model.status == GRB.OPTIMAL:
                best_obj = final_t

        best_interdicted = interdicted
    else:
        # If time is up, return a trivial interdiction (remove first k edges) and the greedy upper bound
        best_obj = ub
        best_interdicted = []
        for i in range(min(k, m)):
            u, v = edge_list[i]
            best_interdicted.append([vertices[u], vertices[v]])

    # Final output
    solution = {
        "objective_value": int(best_obj),
        "interdicted_edges": best_interdicted
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

    # One final log of the returned objective
    if logger:
        logger.log(int(best_obj))

if __name__ == "__main__":
    main()