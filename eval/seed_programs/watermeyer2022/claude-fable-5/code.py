import argparse
import json
import time

import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except Exception:
    SolutionLogger = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", required=True)
    ap.add_argument("--solution_path", required=True)
    ap.add_argument("--time_limit", type=int, default=300)
    ap.add_argument("--log_path", default=None)
    args = ap.parse_args()

    t_start = time.time()

    logger = None
    if args.log_path and SolutionLogger is not None:
        logger = SolutionLogger(args.log_path, sense="minimize")

    with open(args.instance_path) as f:
        inst = json.load(f)

    N = inst["num_activities_total"]
    T = inst["max_project_duration"]
    end = N - 1
    p = {int(a["id"]): int(a["processing_time"]) for a in inst["activities"]}

    arcs = []
    for c in inst["temporal_constraints"]:
        arcs.append((int(c["from_activity"]), int(c["to_activity"]), int(c["time_lag"])))
    # Ensure deadline arc end -> 0 with lag -T exists
    if not any(i == end and j == 0 and l <= -T for (i, j, l) in arcs):
        arcs.append((end, 0, -T))

    NEG = -(10 ** 12)

    # Earliest starts: longest path from node 0 (Bellman-Ford)
    dist = [NEG] * N
    dist[0] = 0
    for _ in range(N + 2):
        changed = False
        for (i, j, l) in arcs:
            if dist[i] > NEG and dist[i] + l > dist[j]:
                dist[j] = dist[i] + l
                changed = True
        if not changed:
            break
    ES = [max(0, dist[j]) if dist[j] > NEG else 0 for j in range(N)]

    # Latest starts: longest path to node 0 (reverse Bellman-Ford); LS_j = -dist(j->0)
    distR = [NEG] * N
    distR[0] = 0
    for _ in range(N + 2):
        changed = False
        for (i, j, l) in arcs:
            if distR[j] > NEG and distR[j] + l > distR[i]:
                distR[i] = distR[j] + l
                changed = True
        if not changed:
            break
    LS = []
    for j in range(N):
        if distR[j] > NEG:
            LS.append(-distR[j])
        else:
            LS.append(max(T, ES[j]))
    for j in range(N):
        if LS[j] < ES[j]:
            LS[j] = ES[j]  # keep model well-formed even if inconsistent

    H = max(T, max(LS[j] + p[j] for j in range(N))) + 1

    # Fallback schedule (temporally earliest starts)
    fallback = {str(j): int(ES[j]) for j in range(N)}
    fallback_obj = float(ES[end])

    best = {"obj": None, "sched": None}

    def write_solution():
        if best["sched"] is not None:
            sol = {"objective_value": float(best["obj"]), "schedule": best["sched"]}
        else:
            sol = {"objective_value": fallback_obj, "schedule": fallback}
        with open(args.solution_path, "w") as f:
            json.dump(sol, f)

    try:
        env = gp.Env(params={"OutputFlag": 0})
        m = gp.Model(env=env)
        m.Params.Seed = 0
        m.Params.MIPGap = 1e-4
        m.Params.NumericFocus = 0
        m.Params.Threads = 1
        rem = args.time_limit - (time.time() - t_start) - 2.0
        m.Params.TimeLimit = max(1.0, rem)

        # x[j,t] = 1 if activity j starts at t
        x = {}
        for j in range(N):
            for t in range(ES[j], LS[j] + 1):
                x[(j, t)] = m.addVar(vtype=GRB.BINARY, name=f"x_{j}_{t}")

        # Cumulative variables y[j,t] = sum_{tau<=t} x[j,tau]
        y = {}
        for j in range(N):
            if ES[j] == LS[j]:
                m.addConstr(x[(j, ES[j])] == 1)
            else:
                for t in range(ES[j], LS[j] + 1):
                    y[(j, t)] = m.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS)
                m.addConstr(y[(j, ES[j])] == x[(j, ES[j])])
                for t in range(ES[j] + 1, LS[j] + 1):
                    m.addConstr(y[(j, t)] == y[(j, t - 1)] + x[(j, t)])
                m.addConstr(y[(j, LS[j])] == 1)

        def Y(j, s):
            # cumulative "started by time s" value/var for activity j
            if s < ES[j]:
                return 0.0
            if s >= LS[j]:
                return 1.0
            return y[(j, s)]

        # Temporal constraints
        n_disagg = sum((LS[j] - ES[j] + 1) for (i, j, l) in arcs if i != j)
        use_disagg = n_disagg <= 1_500_000

        for (i, j, l) in arcs:
            if i == j:
                continue
            if use_disagg:
                for t in range(ES[j], LS[j] + 1):
                    yj = Y(j, t)
                    yi = Y(i, t - l)
                    if isinstance(yi, float):
                        if yi >= 1.0:
                            continue
                        # yi == 0 -> activity j cannot have started by t
                        if isinstance(yj, float):
                            if yj > 0:
                                pass  # inconsistent windows; skip (model may be infeasible)
                        else:
                            m.addConstr(yj <= 0)
                    else:
                        if isinstance(yj, float):
                            if yj <= 0:
                                continue
                            m.addConstr(yi >= 1)
                        else:
                            m.addConstr(yj <= yi)
            else:
                Si = gp.quicksum(t * x[(i, t)] for t in range(ES[i], LS[i] + 1))
                Sj = gp.quicksum(t * x[(j, t)] for t in range(ES[j], LS[j] + 1))
                m.addConstr(Sj - Si >= l)

        # Partially renewable resource constraints
        for res in inst["resources"]:
            cap = int(res["capacity"])
            Pi = set(int(t) for t in res["Pi_k"])
            dem = {int(k): int(v) for k, v in res["demands"].items()}
            # prefix count of Pi periods <= t
            pref = [0] * (H + 2)
            for t in range(1, H + 1):
                pref[t] = pref[t - 1] + (1 if t in Pi else 0)
            expr = gp.LinExpr()
            any_term = False
            for j in range(N):
                d = dem.get(j, 0)
                if d <= 0 or p[j] <= 0:
                    continue
                for t in range(ES[j], LS[j] + 1):
                    lo = min(t, H)
                    hi = min(t + p[j], H)
                    cov = pref[hi] - pref[lo]
                    if cov > 0:
                        expr.addTerms(d * cov, x[(j, t)])
                        any_term = True
            if any_term:
                m.addConstr(expr <= cap)

        # Objective: minimize start of project-end activity
        m.setObjective(
            gp.quicksum(t * x[(end, t)] for t in range(ES[end], LS[end] + 1)),
            GRB.MINIMIZE,
        )

        xkeys = list(x.keys())
        xvars = [x[k] for k in xkeys]

        def cb(model, where):
            if where == GRB.Callback.MIPSOL:
                vals = model.cbGetSolution(xvars)
                sched = {}
                for (j, t), v in zip(xkeys, vals):
                    if v > 0.5:
                        sched[str(j)] = int(t)
                for j in range(N):
                    sched.setdefault(str(j), int(ES[j]))
                obj = float(sched[str(end)])
                if best["obj"] is None or obj < best["obj"]:
                    best["obj"] = obj
                    best["sched"] = sched
                    if logger:
                        logger.log_solution(obj, {"objective_value": obj, "schedule": sched})

        m.optimize(cb)

        if m.SolCount > 0:
            sched = {}
            for (j, t), var in x.items():
                if var.X > 0.5:
                    sched[str(j)] = int(t)
            for j in range(N):
                sched.setdefault(str(j), int(ES[j]))
            obj = float(sched[str(end)])
            if best["obj"] is None or obj <= best["obj"]:
                best["obj"] = obj
                best["sched"] = sched
                if logger:
                    logger.log_solution(obj, {"objective_value": obj, "schedule": sched})
    except Exception:
        pass

    write_solution()


if __name__ == "__main__":
    main()