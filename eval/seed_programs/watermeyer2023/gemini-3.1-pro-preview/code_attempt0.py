import sys
import json
import argparse
import math
import collections
import gurobipy as gp
from gurobipy import GRB

def solve():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default="")
    args = parser.parse_args()

    # Solution Logger initialization
    logger = None
    if args.log_path:
        try:
            from solution_logger import SolutionLogger
            logger = SolutionLogger(args.log_path, sense="minimize")
        except ImportError:
            pass

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    V = instance["num_nodes"]
    deadline = instance["deadline"]
    p = instance["processing_times"]
    resources = instance.get("resources", [])

    # ====== STAGE 1: Internal Floyd-Warshall ======
    # Compute maximum internal path length to safely bound start times
    # and find isolated components bounds without exploding variable domain limits.
    D_base = [[-math.inf] * V for _ in range(V)]
    for i in range(V): 
        D_base[i][i] = 0

    for c in instance["temporal_constraints"]:
        u = c["from"]
        v = c["to"]
        lag = c["delta"]
        if lag > D_base[u][v]:
            D_base[u][v] = lag

    for k in range(V):
        D_k = D_base[k]
        for i in range(V):
            dik = D_base[i][k]
            if dik != -math.inf:
                D_i = D_base[i]
                for j in range(V):
                    dkj = D_k[j]
                    if dkj != -math.inf:
                        ans = dik + dkj
                        if ans > D_i[j]:
                            D_i[j] = ans

    max_path = 0
    for i in range(V):
        if D_base[i][i] > 0:
            print("Infeasible: Positive cycle detected in temporal constraints.")
            return
        for j in range(V):
            if D_base[i][j] != -math.inf and D_base[i][j] > max_path:
                max_path = D_base[i][j]

    # Upper bound logic: to completely avoid resource conflict for late tasks,
    # the farthest any component would realistically need to shift past the `deadline` 
    # to completely evade time windows does not exceed `max_path`.
    upper_bound = deadline + max_path

    # ====== STAGE 2: Full Floyd-Warshall with Bounds ======
    D = [[-math.inf] * V for _ in range(V)]
    for i in range(V):
        for j in range(V):
            D[i][j] = D_base[i][j]

    for i in range(V):
        if 0 > D[0][i]: D[0][i] = 0
        if -upper_bound > D[i][0]: D[i][0] = -upper_bound

    if -deadline > D[V-1][0]:
        D[V-1][0] = -deadline

    for k in range(V):
        D_k = D[k]
        for i in range(V):
            dik = D[i][k]
            if dik != -math.inf:
                D_i = D[i]
                for j in range(V):
                    dkj = D_k[j]
                    if dkj != -math.inf:
                        ans = dik + dkj
                        if ans > D_i[j]:
                            D_i[j] = ans

    for i in range(V):
        if D[i][i] > 0:
            print("Infeasible: Constraints conflict with project bounds.")
            return

    ES = [D[0][i] for i in range(V)]
    LS = [-D[i][0] for i in range(V)]

    # Compute resource consumption dictionary U[(i, r_idx, t)] 
    U = {}
    for r_idx, res in enumerate(resources):
        pi_set = set(res["time_periods"])
        d = res["demands"]
        for i in range(V):
            if d[i] > 0 and p[i] > 0:
                for t in range(ES[i], LS[i] + 1):
                    count = 0
                    for k in range(t + 1, t + p[i] + 1):
                        if k in pi_set:
                            count += 1
                    if count > 0:
                        U[(i, r_idx, t)] = d[i] * count

    # ====== MIP MODEL BUILD ======
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    model = gp.Model("PSPRR", env=env)
    model.setParam("TimeLimit", args.time_limit)
    model.setParam("Threads", 1)

    # Step variables Z[i, t] = 1 if S_i <= t
    Z = {}
    for i in range(V):
        for t in range(ES[i], LS[i]):
            Z[i, t] = model.addVar(vtype=GRB.BINARY, name=f"Z_{i}_{t}")

    # Monotonicity constraints: Z[i, t] <= Z[i, t+1]
    for i in range(V):
        for t in range(ES[i], LS[i] - 1):
            model.addConstr(Z[i, t] <= Z[i, t+1])

    # Temporal constraints enforcing S_to - S_from >= delta
    for c in instance["temporal_constraints"]:
        u = c["from"]
        v = c["to"]
        lag = c["delta"]
        for t in range(ES[v], LS[v]):
            t_u = t - lag
            if t_u < ES[u]:
                model.addConstr(Z[v, t] == 0)
            elif t_u >= LS[u]:
                pass
            else:
                model.addConstr(Z[v, t] <= Z[u, t_u])

    # Partially renewable resource constraints
    for r_idx, res in enumerate(resources):
        C_r = res["capacity"]
        z_coeffs = collections.defaultdict(float)
        const_term = 0.0

        for i in range(V):
            has_demand = any((i, r_idx, t) in U for t in range(ES[i], LS[i] + 1))
            if not has_demand:
                continue

            for t in range(ES[i], LS[i] + 1):
                u_val = U.get((i, r_idx, t), 0.0)
                if u_val == 0.0:
                    continue

                if ES[i] == LS[i]:
                    if t == ES[i]: const_term += u_val
                elif t == ES[i]:
                    z_coeffs[(i, t)] += u_val
                elif t == LS[i]:
                    const_term += u_val
                    z_coeffs[(i, t - 1)] -= u_val
                else:
                    z_coeffs[(i, t)] += u_val
                    z_coeffs[(i, t - 1)] -= u_val

        lin_expr = gp.LinExpr(const_term)
        for (i, t), c_val in z_coeffs.items():
            if c_val != 0:
                lin_expr.addTerms(c_val, Z[i, t])
        model.addConstr(lin_expr <= C_r)

    # Objective: Minimize start time equivalent of the end activity Dummy_End
    if ES[V-1] != LS[V-1]:
        obj_expr = LS[V-1] - gp.quicksum(Z[V-1, t] for t in range(ES[V-1], LS[V-1]))
        model.setObjective(obj_expr, GRB.MINIMIZE)

    best_obj = float('inf')

    def cb(mod, where):
        nonlocal best_obj
        if where == GRB.Callback.MIPSOL:
            obj = mod.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj < best_obj - 1e-5:
                best_obj = obj
                if logger:
                    logger.log(int(round(obj)))

    model.optimize(cb)

    if model.SolCount > 0:
        start_times = []
        for i in range(V):
            if ES[i] == LS[i]:
                start_times.append(ES[i])
            else:
                s_i = LS[i]
                for t in range(ES[i], LS[i]):
                    if Z[i, t].X >= 0.5:
                        s_i = t
                        break
                start_times.append(s_i)

        out = {
            "objective_value": int(round(start_times[V-1])),
            "start_times": [int(round(st)) for st in start_times]
        }
        with open(args.solution_path, "w") as f:
            json.dump(out, f, indent=2)

if __name__ == "__main__":
    solve()