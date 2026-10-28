import argparse
import json
import time
import random

HAS_GUROBI = False
try:
    import gurobipy as gp
    from gurobipy import GRB
    HAS_GUROBI = True
except ImportError:
    pass

try:
    from solution_logger import SolutionLogger
except Exception:
    SolutionLogger = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    return parser.parse_args()


def load_instance(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def build_and_solve(instance, time_limit, logger=None):
    groups = instance.get("groups", [])
    capacities = instance.get("capacities", [])
    m_resources = len(capacities)

    env = gp.Env(empty=True)
    env.setParam("Seed", 0)
    env.setParam("Threads", 1)
    env.setParam("OutputFlag", 0)
    env.start()

    model = gp.Model(env=env)
    model.setParam("MIPGap", 1e-4)
    model.setParam("NumericFocus", 0)
    model.setParam("TimeLimit", float(time_limit))

    var_info = []
    group_vars = {}

    for g in groups:
        gid = g["group_id"]
        gvars = []
        for it in g["items"]:
            iid = it["item_id"]
            profit = it["profit"]
            weights = it["weights"]
            x = model.addVar(vtype=GRB.BINARY, obj=profit, name=f"x_{gid}_{iid}")
            var_info.append({
                "var": x,
                "group_id": gid,
                "item_id": iid,
                "profit": profit,
                "weights": weights,
            })
            gvars.append(x)
        group_vars[gid] = gvars

    for gid, gvars in group_vars.items():
        model.addConstr(gp.quicksum(gvars) == 1, name=f"one_per_group_{gid}")

    if m_resources > 0:
        for r in range(m_resources):
            terms = []
            for d in var_info:
                w = d["weights"][r] if r < len(d["weights"]) else 0
                if w != 0:
                    terms.append(w * d["var"])
            if terms:
                model.addConstr(gp.quicksum(terms) <= capacities[r], name=f"resource_{r}")

    model.ModelSense = GRB.MAXIMIZE

    if logger is not None:
        model._vars = [d["var"] for d in var_info]
        model._var_info = var_info
        model._logger = logger
        
        def gurobi_callback(model, where):
            if where == GRB.Callback.MIPSOL:
                obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                x = model.cbGetSolution(model._vars)
                selected = []
                for d, val in zip(model._var_info, x):
                    if val is not None and val > 0.5:
                        selected.append({
                            "group_id": d["group_id"],
                            "item_id": d["item_id"],
                            "profit": d["profit"],
                            "weights": d["weights"],
                        })
                if model._logger and obj is not None:
                    solution_dict = {
                        "objective_value": float(obj),
                        "selected_items": selected,
                    }
                    model._logger.log_solution(float(obj), solution_dict)
                    
        model.optimize(gurobi_callback)
    else:
        model.optimize()

    best_obj = None
    selected = []

    if model.SolCount > 0:
        try:
            best_obj = model.ObjVal
        except Exception:
            best_obj = None
        for d in var_info:
            try:
                val = d["var"].Xn if hasattr(d["var"], "Xn") else d["var"].X
            except Exception:
                val = d["var"].X
            if val is not None and val > 0.5:
                selected.append({
                    "group_id": d["group_id"],
                    "item_id": d["item_id"],
                    "profit": d["profit"],
                    "weights": d["weights"],
                })

    if best_obj is None:
        best_obj = 0.0
        selected = []

    return float(best_obj), selected, True


def solve_heuristic(instance, time_limit, logger=None):
    start_time = time.time()
    
    groups = instance['groups']
    groups_by_id = {g['group_id']: g for g in groups}
    capacities = instance['capacities']
    m = len(capacities)
    
    def get_violation(used):
        v = 0
        for r in range(m):
            if used[r] > capacities[r]:
                v += used[r] - capacities[r]
        return v
        
    def repair(sol, used, prof, time_limit, start_time):
        violation = get_violation(used)
        while violation > 0 and time.time() - start_time < time_limit:
            best_delta_v = 0
            best_g = None
            best_j = None
            best_delta_p = -1e18
            
            for g in groups:
                gid = g['group_id']
                i = sol[gid]
                item_i = g['items'][i]
                for j in range(len(g['items'])):
                    if i == j: continue
                    item_j = g['items'][j]
                    delta_v = 0
                    for r in range(m):
                        old_r = used[r] - item_i['weights'][r] + item_j['weights'][r]
                        old_viol = max(0, used[r] - capacities[r])
                        new_viol = max(0, old_r - capacities[r])
                        delta_v += old_viol - new_viol
                    delta_p = item_j['profit'] - item_i['profit']
                    
                    if delta_v > best_delta_v:
                        best_delta_v = delta_v
                        best_g = gid
                        best_j = j
                        best_delta_p = delta_p
                    elif delta_v == best_delta_v and best_delta_v > 0 and delta_p > best_delta_p:
                        best_g = gid
                        best_j = j
                        best_delta_p = delta_p
                        
            if best_g is not None:
                g = groups_by_id[best_g]
                i = sol[best_g]
                j = best_j
                item_i = g['items'][i]
                item_j = g['items'][j]
                for r in range(m):
                    used[r] += item_j['weights'][r] - item_i['weights'][r]
                prof += best_delta_p
                sol[best_g] = j
                violation = get_violation(used)
            else:
                break
        return sol, used, prof, violation == 0

    def construct_solution(strategy):
        sol = {}
        used = [0] * m
        prof = 0
        for g in groups:
            if strategy == "min_weight":
                idx = min(range(len(g['items'])), key=lambda i: sum(g['items'][i]['weights']))
            elif strategy == "max_ratio":
                def ratio(item):
                    w = sum(item['weights'])
                    if w == 0: return item['profit'] if item['profit'] > 0 else 1e-6
                    return item['profit'] / w
                idx = max(range(len(g['items'])), key=lambda i: ratio(g['items'][i]))
            else: # max profit
                idx = max(range(len(g['items'])), key=lambda i: g['items'][i]['profit'])
                
            sol[g['group_id']] = idx
            item = g['items'][idx]
            for r in range(m):
                used[r] += item['weights'][r]
            prof += item['profit']
        return sol, used, prof

    best_sol = None
    best_prof = -1
    best_used = None
    
    for strat in ["min_weight", "max_ratio", "max_profit"]:
        if time.time() - start_time >= time_limit:
            break
        sol, used, prof = construct_solution(strat)
        sol, used, prof, feasible = repair(sol, used, prof, time_limit, start_time)
        
        if feasible and prof > best_prof:
            best_sol = dict(sol)
            best_prof = prof
            best_used = list(used)
            
            selected = []
            for gg in groups:
                item = gg['items'][sol[gg['group_id']]]
                selected.append({
                    "group_id": gg['group_id'],
                    "item_id": item['item_id'],
                    "profit": item['profit'],
                    "weights": item['weights'],
                })
            if logger:
                solution_dict = {"objective_value": float(best_prof), "selected_items": selected}
                logger.log_solution(float(best_prof), solution_dict)

    if best_sol is not None:
        sol = dict(best_sol)
        used = list(best_used)
        prof = best_prof
        
        while time.time() - start_time < time_limit:
            best_gain = 0
            best_g = None
            best_j = None
            
            for g in groups:
                gid = g['group_id']
                i = sol[gid]
                item_i = g['items'][i]
                for j in range(len(g['items'])):
                    if i == j: continue
                    item_j = g['items'][j]
                    gain = item_j['profit'] - item_i['profit']
                    if gain > best_gain:
                        feasible = True
                        for r in range(m):
                            if used[r] - item_i['weights'][r] + item_j['weights'][r] > capacities[r]:
                                feasible = False
                                break
                        if feasible:
                            best_gain = gain
                            best_g = gid
                            best_j = j
                            
            if best_g is not None:
                g = groups_by_id[best_g]
                i = sol[best_g]
                j = best_j
                item_i = g['items'][i]
                item_j = g['items'][j]
                for r in range(m):
                    used[r] += item_j['weights'][r] - item_i['weights'][r]
                prof += best_gain
                sol[best_g] = j
                
                if prof > best_prof:
                    best_prof = prof
                    best_sol = dict(sol)
                    best_used = list(used)
                    selected = []
                    for gg in groups:
                        item = gg['items'][sol[gg['group_id']]]
                        selected.append({
                            "group_id": gg['group_id'],
                            "item_id": item['item_id'],
                            "profit": item['profit'],
                            "weights": item['weights'],
                        })
                    if logger:
                        solution_dict = {"objective_value": float(best_prof), "selected_items": selected}
                        logger.log_solution(float(best_prof), solution_dict)
            else:
                # Perturbation
                if time.time() - start_time < time_limit:
                    g = random.choice(groups)
                    gid = g['group_id']
                    i = sol[gid]
                    possible_j = [j for j in range(len(g['items'])) if j != i]
                    if not possible_j: continue
                    j = random.choice(possible_j)
                    item_i = g['items'][i]
                    item_j = g['items'][j]
                    for r in range(m):
                        used[r] += item_j['weights'][r] - item_i['weights'][r]
                    prof += item_j['profit'] - item_i['profit']
                    sol[gid] = j
                    
                    sol, used, prof, feasible = repair(sol, used, prof, time_limit, start_time)
                    if feasible and prof > best_prof:
                        best_prof = prof
                        best_sol = dict(sol)
                        best_used = list(used)
                        selected = []
                        for gg in groups:
                            item = gg['items'][sol[gg['group_id']]]
                            selected.append({
                                "group_id": gg['group_id'],
                                "item_id": item['item_id'],
                                "profit": item['profit'],
                                "weights": item['weights'],
                            })
                        if logger:
                            solution_dict = {"objective_value": float(best_prof), "selected_items": selected}
                            logger.log_solution(float(best_prof), solution_dict)
                else:
                    break

    if best_sol is not None:
        selected = []
        for g in groups:
            item = g['items'][best_sol[g['group_id']]]
            selected.append({
                "group_id": g['group_id'],
                "item_id": item['item_id'],
                "profit": item['profit'],
                "weights": item['weights'],
            })
        return float(best_prof), selected, True
    else:
        return 0.0, [], False


def main():
    args = parse_args()
    instance = load_instance(args.instance_path)

    logger = None
    if args.log_path and SolutionLogger is not None:
        logger = SolutionLogger(args.log_path, sense="maximize")

    start_time = time.time()
    remaining_time = max(1.0, args.time_limit - (time.time() - start_time))

    if HAS_GUROBI:
        obj, selected, status = build_and_solve(instance, remaining_time, logger=logger)
    else:
        obj, selected, status = solve_heuristic(instance, remaining_time, logger=logger)

    solution = {
        "objective_value": float(obj),
        "selected_items": selected,
    }

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()