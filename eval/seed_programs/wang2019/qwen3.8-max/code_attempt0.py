import argparse
import json
from collections import deque
from gurobipy import *


def forward_reachable(start, out_arcs):
    """BFS to find all nodes reachable from start via outgoing arcs."""
    visited = {start}
    q = deque([start])
    while q:
        u = q.popleft()
        for a in out_arcs.get(u, []):
            v = a['to']
            if v not in visited:
                visited.add(v)
                q.append(v)
    return visited


def backward_reachable(target, in_arcs):
    """BFS to find all nodes that can reach target via incoming arcs."""
    visited = {target}
    q = deque([target])
    while q:
        u = q.popleft()
        for a in in_arcs.get(u, []):
            v = a['from']
            if v not in visited:
                visited.add(v)
                q.append(v)
    return visited


def extract_cb_solution(model, x, y, arcs, commodities, fleet_types, obj_val):
    flows = {}
    vehicles = {}
    for k in commodities:
        kid = k['id']
        for a in arcs:
            key = f"x_{kid}_{a['from']}_{a['to']}"
            if (kid, a['from'], a['to']) in x:
                flows[key] = model.cbGetSolution(x[kid, a['from'], a['to']])
            else:
                flows[key] = 0.0
    for m in fleet_types:
        mid = m['id']
        for a in arcs:
            key = f"y_{mid}_{a['from']}_{a['to']}"
            vehicles[key] = int(round(model.cbGetSolution(y[mid, a['from'], a['to']])))
    return {"objective_value": obj_val, "flows": flows, "vehicles": vehicles}


def extract_final_solution(model, x, y, arcs, commodities, fleet_types):
    flows = {}
    vehicles = {}
    for k in commodities:
        kid = k['id']
        for a in arcs:
            key = f"x_{kid}_{a['from']}_{a['to']}"
            if (kid, a['from'], a['to']) in x:
                flows[key] = x[kid, a['from'], a['to']].X
            else:
                flows[key] = 0.0
    for m in fleet_types:
        mid = m['id']
        for a in arcs:
            key = f"y_{mid}_{a['from']}_{a['to']}"
            vehicles[key] = int(round(y[mid, a['from'], a['to']].X))
    return {"objective_value": model.ObjVal, "flows": flows, "vehicles": vehicles}


def build_default_solution(arcs, commodities, fleet_types):
    flows = {}
    vehicles = {}
    for k in commodities:
        for a in arcs:
            flows[f"x_{k['id']}_{a['from']}_{a['to']}"] = 0.0
    for m in fleet_types:
        for a in arcs:
            vehicles[f"y_{m['id']}_{a['from']}_{a['to']}"] = 0
    return {"objective_value": 0.0, "flows": flows, "vehicles": vehicles}


def main():
    parser = argparse.ArgumentParser(description="Service Network Design with Fleet Management")
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Read instance
    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    nodes = data['nodes']
    arcs = data['arcs']
    commodities = data['commodities']
    fleet_types = data['fleet_types']

    # Build adjacency lists
    out_arcs = {n: [] for n in nodes}
    in_arcs = {n: [] for n in nodes}
    for a in arcs:
        out_arcs[a['from']].append(a)
        in_arcs[a['to']].append(a)

    # Precompute useful arcs per commodity (pruning)
    commodity_useful_arcs = {}
    for k in commodities:
        fwd = forward_reachable(k['origin'], out_arcs)
        bwd = backward_reachable(k['destination'], in_arcs)
        useful = set()
        for a in arcs:
            if a['from'] in fwd and a['to'] in bwd:
                useful.add((a['from'], a['to']))
        commodity_useful_arcs[k['id']] = useful

    # Create Gurobi model
    model = Model("service_network_design")
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('OutputFlag', 0)

    # Decision variables: x[k, from, to] = flow of commodity k on arc (from, to)
    x = {}
    for k in commodities:
        kid = k['id']
        for a in arcs:
            if (a['from'], a['to']) in commodity_useful_arcs[kid]:
                x[kid, a['from'], a['to']] = model.addVar(
                    lb=0.0, vtype=GRB.CONTINUOUS,
                    name=f"x_{kid}_{a['from']}_{a['to']}"
                )

    # Decision variables: y[m, from, to] = number of vehicles of type m on arc (from, to)
    y = {}
    total_demand = sum(k['demand'] for k in commodities)
    for m in fleet_types:
        mid = m['id']
        cap = m['capacity']
        ub_y = (total_demand + cap - 1) // cap if cap > 0 else 0
        for a in arcs:
            y[mid, a['from'], a['to']] = model.addVar(
                lb=0, ub=ub_y, vtype=GRB.INTEGER,
                name=f"y_{mid}_{a['from']}_{a['to']}"
            )

    model.update()

    # Objective: minimize total flow cost + total fixed vehicle cost
    obj = LinExpr()
    for a in arcs:
        af, at, fc = a['from'], a['to'], a['flow_cost']
        for k in commodities:
            if (k['id'], af, at) in x:
                obj += fc * x[k['id'], af, at]
    for m in fleet_types:
        fxc = m['fixed_cost']
        for a in arcs:
            obj += fxc * y[m['id'], a['from'], a['to']]
    model.setObjective(obj, GRB.MINIMIZE)

    # Flow conservation constraints for each commodity at each node
    for k in commodities:
        kid = k['id']
        for n in nodes:
            expr = LinExpr()
            for a in out_arcs[n]:
                if (kid, a['from'], a['to']) in x:
                    expr += x[kid, a['from'], a['to']]
            for a in in_arcs[n]:
                if (kid, a['from'], a['to']) in x:
                    expr -= x[kid, a['from'], a['to']]

            if n == k['origin']:
                rhs = float(k['demand'])
            elif n == k['destination']:
                rhs = -float(k['demand'])
            else:
                rhs = 0.0

            model.addConstr(expr == rhs, name=f"flow_bal_{kid}_{n}")

    # Capacity constraints: total flow on each arc <= total vehicle capacity
    for a in arcs:
        af, at = a['from'], a['to']
        flow_expr = LinExpr()
        for k in commodities:
            if (k['id'], af, at) in x:
                flow_expr += x[k['id'], af, at]

        cap_expr = LinExpr()
        for m in fleet_types:
            cap_expr += m['capacity'] * y[m['id'], af, at]

        model.addConstr(flow_expr <= cap_expr, name=f"cap_{af}_{at}")

    # Vehicle balance constraints: for each fleet type, vehicles form circulations
    for m in fleet_types:
        mid = m['id']
        for n in nodes:
            expr = LinExpr()
            for a in in_arcs[n]:
                expr += y[mid, a['from'], a['to']]
            for a in out_arcs[n]:
                expr -= y[mid, a['from'], a['to']]
            model.addConstr(expr == 0, name=f"veh_bal_{mid}_{n}")

    model.update()

    # Callback to log incumbent solutions
    best_obj = [float('inf')]

    def my_callback(mdl, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = mdl.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj_val < best_obj[0] - 1e-9:
                best_obj[0] = obj_val
                if logger:
                    sol = extract_cb_solution(mdl, x, y, arcs, commodities, fleet_types, obj_val)
                    logger.log_solution(obj_val, sol)

    # Solve
    model.optimize(my_callback)

    # Extract and write final solution
    if model.SolCount > 0:
        solution = extract_final_solution(model, x, y, arcs, commodities, fleet_types)
    else:
        solution = build_default_solution(arcs, commodities, fleet_types)

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()