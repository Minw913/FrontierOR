import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    logger = None
    if args.log_path:
        try:
            from solution_logger import SolutionLogger
            logger = SolutionLogger(args.log_path, sense="maximize")
        except Exception:
            logger = None

    with open(args.instance_path) as f:
        data = json.load(f)

    n = data['n']
    m = data['m']
    capacity = data['capacity']
    weights = data['weights']
    values = data['values']
    pairwise = data['pairwise_values']

    def compute_objective(assignment):
        total = 0
        items_by_k = {}
        for i, k in assignment.items():
            total += values[i]
            items_by_k.setdefault(k, []).append(i)
        for k, items in items_by_k.items():
            for a in range(len(items)):
                for b in range(a + 1, len(items)):
                    total += pairwise[items[a]][items[b]]
        return total

    def greedy_warmstart():
        order = sorted(range(n), key=lambda i: -(values[i] / max(weights[i], 1)))
        assignment = {}
        knap_w = [0] * m
        knap_items = [[] for _ in range(m)]
        for i in order:
            best_k = -1
            best_gain = 0.0
            for k in range(m):
                if knap_w[k] + weights[i] <= capacity:
                    gain = values[i]
                    for j in knap_items[k]:
                        gain += pairwise[i][j]
                    if best_k == -1 or gain > best_gain:
                        best_gain = gain
                        best_k = k
            if best_k != -1 and best_gain > 0:
                assignment[i] = best_k
                knap_w[best_k] += weights[i]
                knap_items[best_k].append(i)
        return assignment

    warm = greedy_warmstart()
    warm_obj = compute_objective(warm)
    if logger:
        logger.log_solution(float(warm_obj), {str(i): k for i, k in warm.items()})

    import gurobipy as gp
    from gurobipy import GRB

    env = gp.Env(empty=True)
    env.setParam('Seed', 0)
    env.setParam('Threads', 1)
    env.setParam('OutputFlag', 0)
    env.start()

    model = gp.Model(env=env)
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.TimeLimit = float(max(args.time_limit, 1))

    x = {}
    for i in range(n):
        for k in range(m):
            x[i, k] = model.addVar(vtype=GRB.BINARY, name=f'x_{i}_{k}')

    for i in range(n):
        for k in range(m):
            x[i, k].Start = 1.0 if warm.get(i, -1) == k else 0.0

    obj = gp.LinExpr(0)
    for i in range(n):
        obj += values[i] * gp.quicksum(x[i, k] for k in range(m))

    for i in range(n):
        for j in range(i + 1, n):
            pij = pairwise[i][j]
            if pij == 0:
                continue
            for k in range(m):
                yv = model.addVar(vtype=GRB.BINARY, name=f'y_{i}_{j}_{k}')
                obj += pij * yv
                model.addConstr(yv <= x[i, k])
                model.addConstr(yv <= x[j, k])
                model.addConstr(yv >= x[i, k] + x[j, k] - 1)

    model.setObjective(obj, GRB.MAXIMIZE)

    for i in range(n):
        model.addConstr(gp.quicksum(x[i, k] for k in range(m)) <= 1)

    for k in range(m):
        model.addConstr(gp.quicksum(weights[i] * x[i, k] for i in range(n)) <= capacity)

    x_vars_flat = [x[i, k] for i in range(n) for k in range(m)]

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            try:
                obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if logger:
                    vals = model.cbGetSolution(x_vars_flat)
                    assignment = {}
                    idx = 0
                    for i in range(n):
                        for k in range(m):
                            if vals[idx] > 0.5:
                                assignment[str(i)] = k
                            idx += 1
                    logger.log_solution(float(obj_val), assignment)
            except Exception:
                pass

    model.optimize(callback)

    if model.SolCount == 0:
        result = {
            "objective_value": float(warm_obj),
            "assignment": {str(i): k for i, k in warm.items()}
        }
    else:
        obj_val = model.ObjVal
        assignment = {}
        for i in range(n):
            for k in range(m):
                try:
                    if x[i, k].Xn > 0.5:
                        assignment[str(i)] = k
                except Exception:
                    if x[i, k].X > 0.5:
                        assignment[str(i)] = k
        # Prefer the better of (MIP incumbent, warm heuristic)
        mip_obj = compute_objective({int(k): v for k, v in assignment.items()})
        if mip_obj >= warm_obj:
            result = {"objective_value": float(obj_val), "assignment": assignment}
        else:
            result = {
                "objective_value": float(warm_obj),
                "assignment": {str(i): k for i, k in warm.items()}
            }

    with open(args.solution_path, 'w') as f:
        json.dump(result, f, indent=2)

    if logger:
        logger.log_solution(float(result["objective_value"]), result["assignment"])


if __name__ == '__main__':
    main()