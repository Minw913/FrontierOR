import argparse
import json
import time
from collections import deque

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception:
    gp = None
    GRB = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    logger = None
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize")

    time_limit = max(0, int(args.time_limit))
    deadline = time.time() + time_limit

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    m = int(data["num_stations"])
    C = int(data["cycle_time"])
    proc = {int(k): int(v) for k, v in data["task_processing_times"].items()}
    tasks = sorted(proc.keys())

    e_raw = {int(k): int(v) for k, v in data.get("earliest_station", {}).items()}
    l_raw = {int(k): int(v) for k, v in data.get("latest_station", {}).items()}
    E = {i: max(1, min(m, int(e_raw.get(i, 1)))) for i in tasks}
    L = {i: max(1, min(m, int(l_raw.get(i, m)))) for i in tasks}

    preds = {i: [] for i in tasks}
    succs = {i: [] for i in tasks}
    indeg = {i: 0 for i in tasks}
    for a, b in data.get("precedence_arcs", []):
        i, j = int(a), int(b)
        if i in proc and j in proc and i != j:
            succs[i].append(j)
            preds[j].append(i)
            indeg[j] += 1

    def topo_order():
        ind = dict(indeg)
        q = deque([i for i in tasks if ind[i] == 0])
        order = []
        seen = set()
        while q:
            i = q.popleft()
            if i in seen:
                continue
            seen.add(i)
            order.append(i)
            for j in succs[i]:
                ind[j] -= 1
                if ind[j] == 0:
                    q.append(j)
        for i in tasks:
            if i not in seen:
                order.append(i)
        return order

    order = topo_order()
    T = sum(proc.values())
    # Valid lower bound: max idle >= C - floor(total_time / num_stations).
    lb_obj = max(0, C - (T // m))

    allowed = {i: list(range(E[i], L[i] + 1)) if E[i] <= L[i] else [] for i in tasks}
    cand_for_station = [[] for _ in range(m + 1)]
    for i in tasks:
        for s in allowed[i]:
            cand_for_station[s].append(i)

    def compute_loads(assign):
        loads = [0] * (m + 1)
        for i, s in assign.items():
            if 1 <= s <= m:
                loads[s] += proc[i]
        return loads

    def is_feasible(assign):
        if not assign or any(i not in assign for i in tasks):
            return False
        for i in tasks:
            s = assign[i]
            if s < E[i] or s > L[i] or s < 1 or s > m:
                return False
        for i in tasks:
            si = assign[i]
            for j in succs[i]:
                if si > assign[j]:
                    return False
        loads = compute_loads(assign)
        return all(loads[s] <= C for s in range(1, m + 1))

    def make_solution(assign):
        loads = compute_loads(assign)
        obj = float(max(C - loads[s] for s in range(1, m + 1)))
        sol = {
            "objective_value": obj,
            "task_assignments": {str(i): int(assign[i]) for i in tasks},
        }
        return obj, sol

    state = {"obj": float("inf"), "assign": None}

    def record_solution(assign, require_feasible=True):
        if require_feasible and not is_feasible(assign):
            return False
        obj, sol = make_solution(assign)
        if obj < state["obj"] - 1e-9:
            state["obj"] = obj
            state["assign"] = {i: int(assign[i]) for i in tasks}
            if logger:
                logger.log_solution(obj, sol)
            return True
        return False

    def construct_forward():
        assign = {}
        loads = [0] * (m + 1)
        for i in order:
            low = E[i]
            for p in preds[i]:
                if p in assign:
                    low = max(low, assign[p])
            placed = False
            for s in range(low, L[i] + 1):
                if 1 <= s <= m and loads[s] + proc[i] <= C:
                    assign[i] = s
                    loads[s] += proc[i]
                    placed = True
                    break
            if not placed:
                return None
        return assign

    def construct_backward():
        assign = {}
        loads = [0] * (m + 1)
        for i in reversed(order):
            high = L[i]
            for j in succs[i]:
                if j in assign:
                    high = min(high, assign[j])
            placed = False
            for s in range(high, E[i] - 1, -1):
                if 1 <= s <= m and loads[s] + proc[i] <= C:
                    assign[i] = s
                    loads[s] += proc[i]
                    placed = True
                    break
            if not placed:
                return None
        return assign

    def movable_to(i, b, assign):
        for p in preds[i]:
            if assign[p] > b:
                return False
        for j in succs[i]:
            if assign[j] < b:
                return False
        return True

    def improve_balance(assign, improve_deadline):
        loads = compute_loads(assign)
        target = T // m
        upper = (T + m - 1) // m

        for _ in range(3):
            changed = False
            for b in sorted(range(1, m + 1), key=lambda s: loads[s]):
                if time.time() > improve_deadline:
                    return assign
                while loads[b] < target:
                    best = None
                    best_key = None
                    room = C - loads[b]
                    for i in cand_for_station[b]:
                        s = assign.get(i)
                        if s is None or s == b:
                            continue
                        ti = proc[i]
                        if ti > room or loads[s] - ti < target:
                            continue
                        if not movable_to(i, b, assign):
                            continue
                        b2 = loads[b] + ti
                        s2 = loads[s] - ti
                        key = (max(0, b2 - upper), -min(s2, b2), abs(b2 - target), -s2)
                        if best_key is None or key < best_key:
                            best_key = key
                            best = (i, s)
                    if best is None:
                        break
                    i, s = best
                    assign[i] = b
                    loads[s] -= proc[i]
                    loads[b] += proc[i]
                    changed = True
                    if time.time() > improve_deadline:
                        return assign
            if not changed:
                break

        # If there is a unique bottleneck, raise it even when the ideal target is blocked.
        for _ in range(2 * max(1, len(tasks))):
            if time.time() > improve_deadline:
                break
            min_load = min(loads[1:]) if m >= 1 else 0
            min_stations = [s for s in range(1, m + 1) if loads[s] == min_load]
            if len(min_stations) != 1:
                break
            b = min_stations[0]
            second = min([loads[s] for s in range(1, m + 1) if s != b] or [10**18])
            best = None
            best_new_min = min_load
            best_tie = None
            for i in cand_for_station[b]:
                s = assign.get(i)
                if s is None or s == b:
                    continue
                ti = proc[i]
                if loads[b] + ti > C:
                    continue
                s2 = loads[s] - ti
                if s2 < min_load:
                    continue
                if not movable_to(i, b, assign):
                    continue
                b2 = loads[b] + ti
                new_min = min(second, s2, b2)
                tie = (min(s2, b2), s2, -abs(loads[s] - loads[b]))
                if new_min > best_new_min or (new_min == best_new_min and best is not None and tie > best_tie):
                    best_new_min = new_min
                    best = (i, s)
                    best_tie = tie
            if best is None or best_new_min <= min_load:
                break
            i, s = best
            assign[i] = b
            loads[s] -= proc[i]
            loads[b] += proc[i]
        return assign

    # Fast feasible incumbents; if one hits the valid lower bound, it is provably optimal.
    heur_budget = min(1.0, max(0.05, 0.15 * time_limit)) if time_limit > 0 else 0.0
    heur_deadline = min(deadline, time.time() + heur_budget)
    candidates = []
    a = construct_forward()
    if a is not None:
        candidates.append(a)
    b = construct_backward()
    if b is not None:
        candidates.append(b)
    for cand in candidates:
        if time.time() > heur_deadline:
            break
        cand = dict(cand)
        cand = improve_balance(cand, heur_deadline)
        if is_feasible(cand):
            record_solution(cand)
            if state["obj"] <= lb_obj + 1e-9:
                _, sol = make_solution(state["assign"])
                with open(args.solution_path, "w") as f:
                    json.dump(sol, f)
                return

    def fallback_assignment():
        assign = {}
        for i in order:
            s = E[i]
            for p in preds[i]:
                if p in assign:
                    s = max(s, assign[p])
            assign[i] = max(1, min(m, s))
        return assign

    # Solve the exact compact MIP with Gurobi for the remaining time.
    if gp is not None and state["obj"] > lb_obj + 1e-9:
        remaining = deadline - time.time()
        if remaining > 0.05:
            model = gp.Model("salbp_balance")
            model.setParam("Seed", 0)
            model.setParam("MIPGap", 1e-4)
            model.setParam("NumericFocus", 0)
            model.setParam("Threads", 1)
            model.setParam("LogToConsole", 0)
            model.setParam("OutputFlag", 0)

            x_vars = {}
            x_list = []
            x_keys = []
            x_by_task = {i: [] for i in tasks}
            tasks_for_station = [[] for _ in range(m + 1)]
            for i in tasks:
                for s in allowed[i]:
                    v = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{s}")
                    x_vars[(i, s)] = v
                    x_list.append(v)
                    x_keys.append((i, s))
                    x_by_task[i].append((v, s))
                    tasks_for_station[s].append((i, proc[i]))

            for i in tasks:
                if x_by_task[i]:
                    model.addConstr(gp.quicksum(v for v, _ in x_by_task[i]) == 1, name=f"assign_{i}")
                else:
                    # Should not occur for schema-valid instances; keep the model defined.
                    model.addConstr(1 == 0, name=f"assign_empty_{i}")

            stations = list(range(1, m + 1))
            load = model.addVars(stations, lb=0, ub=C, vtype=GRB.INTEGER, name="load")
            for s in stations:
                model.addConstr(
                    load[s] == gp.quicksum(ti * x_vars[(i, s)] for i, ti in tasks_for_station[s]),
                    name=f"loaddef_{s}",
                )

            best_ub = C
            if state["assign"] is not None:
                best_ub = int(min(C, max(lb_obj, round(state["obj"]))))
            D = model.addVar(lb=int(lb_obj), ub=int(best_ub), vtype=GRB.INTEGER, name="max_idle")
            for s in stations:
                model.addConstr(load[s] + D >= C, name=f"balance_{s}")

            pos_expr = {}
            for i in tasks:
                pos_expr[i] = gp.quicksum(s * x_vars[(i, s)] for s in allowed[i]) if allowed[i] else gp.LinExpr()
            for i in tasks:
                for j in succs[i]:
                    model.addConstr(pos_expr[i] <= pos_expr[j], name=f"prec_{i}_{j}")

            model.setObjective(D, GRB.MINIMIZE)
            model.update()

            if state["assign"] is not None:
                for v in x_list:
                    v.Start = 0.0
                for i, s in state["assign"].items():
                    if (i, s) in x_vars:
                        x_vars[(i, s)].Start = 1.0
                hloads = compute_loads(state["assign"])
                for s in stations:
                    load[s].Start = hloads[s]
                D.Start = state["obj"]

            def assignment_from_vals(vals):
                best_v = {i: -1.0 for i in tasks}
                best_s = {i: (allowed[i][0] if allowed[i] else max(1, min(m, E[i]))) for i in tasks}
                for (i, s), val in zip(x_keys, vals):
                    if val > best_v[i] + 1e-9:
                        best_v[i] = val
                        best_s[i] = s
                return best_s

            def callback(cb_model, where):
                if where == GRB.Callback.MIPSOL:
                    try:
                        vals = cb_model.cbGetSolution(x_list)
                        record_solution(assignment_from_vals(vals), require_feasible=True)
                    except Exception:
                        pass

            model.setParam("TimeLimit", max(0.0, deadline - time.time() - 0.05))
            model.optimize(callback)
            if model.SolCount > 0:
                try:
                    vals = model.getAttr("X", x_list)
                    record_solution(assignment_from_vals(vals), require_feasible=True)
                except Exception:
                    pass

    if state["assign"] is None:
        state["assign"] = fallback_assignment()
    _, sol = make_solution(state["assign"])
    with open(args.solution_path, "w") as f:
        json.dump(sol, f)


if __name__ == "__main__":
    main()