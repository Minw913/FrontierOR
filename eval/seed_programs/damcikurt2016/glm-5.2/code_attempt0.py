import json
import argparse
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def get_descendants(node_id, max_depth, children):
    """Return list of descendant node ids within max_depth edges (BFS)."""
    res = []
    queue = [(node_id, 0)]
    while queue:
        curr, depth = queue.pop(0)
        if depth > 0:
            res.append(curr)
        if depth < max_depth:
            for child in children.get(curr, []):
                queue.append((child, depth + 1))
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path) as f:
        data = json.load(f)

    gen = data['generator']
    tree = data['scenario_tree']

    C_lower = gen['C_lower_MW']
    C_upper = gen['C_upper_MW']
    V_plus = gen['V_plus_MW_per_h']
    V_minus = gen['V_minus_MW_per_h']
    U_bar = gen['U_bar_startup_cost_dollars']
    U_lower = gen['U_lower_shutdown_cost_dollars']
    fuel_a = gen['fuel_a_dollars_per_h']
    fuel_b = gen['fuel_b_dollars_per_MWh']

    L = tree['L_min_up_time']
    ell = tree['ell_min_down_time']
    y0 = tree['initial_generator_status_y0']
    x0 = tree['initial_generation_x0_MW']

    nodes = tree['nodes']
    node_dict = {n['id']: n for n in nodes}
    all_ids = [n['id'] for n in nodes]
    non_root_ids = [n['id'] for n in nodes if n['parent_id'] is not None]
    children = {n['id']: n['children_ids'] for n in nodes}

    # ------------------------------------------------------------------
    # Gurobi model
    # ------------------------------------------------------------------
    env = gp.Env()
    model = gp.Model(env=env)
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = max(1, args.time_limit - 2)
    model.Params.OutputFlag = 0

    y = {}
    x = {}
    u = {}
    for nid in all_ids:
        y[nid] = model.addVar(vtype=GRB.BINARY, name=f"y_{nid}")
        x[nid] = model.addVar(lb=0, ub=C_upper, name=f"x_{nid}")
    for nid in non_root_ids:
        u[nid] = model.addVar(vtype=GRB.BINARY, name=f"u_{nid}")

    model.update()

    # Generation limits
    for nid in all_ids:
        model.addConstr(x[nid] <= C_upper * y[nid])
        model.addConstr(x[nid] >= C_lower * y[nid])

    # Ramp limits and startup logic
    for nid in all_ids:
        n = node_dict[nid]
        if n['parent_id'] is None:
            # Root: parent is the initial state
            model.addConstr(x[nid] - x0 <= V_plus * y0 + C_upper * (1 - y0))
            model.addConstr(x0 - x[nid] <= V_minus * y[nid] + C_upper * (1 - y[nid]))
        else:
            pid = n['parent_id']
            # Startup definition
            model.addConstr(u[nid] <= y[nid])
            model.addConstr(u[nid] <= 1 - y[pid])
            model.addConstr(u[nid] >= y[nid] - y[pid])
            # Ramp up
            model.addConstr(x[nid] - x[pid] <= V_plus * y[pid] + C_upper * (1 - y[pid]))
            # Ramp down
            model.addConstr(x[pid] - x[nid] <= V_minus * y[nid] + C_upper * (1 - y[nid]))

    # Minimum up time
    if L > 1:
        for nid in non_root_ids:
            desc = get_descendants(nid, L - 1, children)
            if desc:
                model.addConstrs(
                    (y[j] >= u[nid] for j in desc),
                    name=f"minup_{nid}"
                )

    # Minimum down time
    if ell > 1:
        for nid in non_root_ids:
            desc = get_descendants(nid, ell - 1, children)
            if desc:
                pid = node_dict[nid]['parent_id']
                model.addConstrs(
                    (y[j] + y[pid] - y[nid] + u[nid] <= 1 for j in desc),
                    name=f"mindown_{nid}"
                )

    # Objective: expected profit
    obj = gp.LinExpr()
    for nid in all_ids:
        n = node_dict[nid]
        prob = n['probability']
        price = n['electricity_price_dollars_per_MWh']
        obj += prob * (price * x[nid] - fuel_a * y[nid] - fuel_b * x[nid])
    for nid in non_root_ids:
        n = node_dict[nid]
        pid = n['parent_id']
        prob = n['probability']
        # startup cost
        obj -= prob * U_bar * u[nid]
        # shutdown cost (v = y[pid] - y[nid] + u[nid])
        obj -= prob * U_lower * (y[pid] - y[nid] + u[nid])

    model.setObjective(obj, GRB.MAXIMIZE)

    # ------------------------------------------------------------------
    # Logger and callback
    # ------------------------------------------------------------------
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    y_vars = [y[i] for i in all_ids]
    x_vars = [x[i] for i in all_ids]
    u_vars = [u[i] for i in non_root_ids] if non_root_ids else []

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                y_vals = model.cbGetSolution(y_vars)
                x_vals = model.cbGetSolution(x_vars)
                u_vals = model.cbGetSolution(u_vars) if u_vars else []
                sol = {
                    "objective_value": float(obj_val),
                    "x": {str(i): float(x_vals[idx]) for idx, i in enumerate(all_ids)},
                    "y": {str(i): float(y_vals[idx]) for idx, i in enumerate(all_ids)},
                    "u": {str(i): float(u_vals[idx]) for idx, i in enumerate(non_root_ids)} if u_vals else {},
                    "sparse_defaults": {"x": 0, "y": 0, "u": 0}
                }
                logger.log_solution(float(obj_val), sol)

    if logger:
        model.setCallback(callback)

    model.optimize()

    # ------------------------------------------------------------------
    # Extract and write solution
    # ------------------------------------------------------------------
    if model.SolCount > 0:
        obj_val = model.ObjVal
        x_dict = {str(nid): float(x[nid].X) for nid in all_ids}
        y_dict = {str(nid): int(round(y[nid].X)) for nid in all_ids}
        u_dict = {str(nid): int(round(u[nid].X)) for nid in non_root_ids}
        solution = {
            "objective_value": float(obj_val),
            "x": x_dict,
            "y": y_dict,
            "u": u_dict,
            "sparse_defaults": {"x": 0, "y": 0, "u": 0}
        }
    else:
        # Fallback: trivial all-off solution
        solution = {
            "objective_value": 0.0,
            "x": {str(nid): 0.0 for nid in all_ids},
            "y": {str(nid): 0 for nid in all_ids},
            "u": {str(nid): 0 for nid in non_root_ids},
            "sparse_defaults": {"x": 0, "y": 0, "u": 0}
        }
        if logger:
            logger.log_solution(0.0, solution)

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()