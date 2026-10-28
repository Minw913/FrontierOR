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

    with open(args.instance_path) as f:
        instance = json.load(f)

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    from gurobipy import Model, GRB, quicksum

    nodes = instance['nodes']
    arcs = instance['arcs']
    commodities = instance['commodities']
    fleet_types = instance['fleet_types']

    out_arcs = {n: [] for n in nodes}
    in_arcs = {n: [] for n in nodes}
    adj_out = {n: set() for n in nodes}
    adj_in = {n: set() for n in nodes}
    for a in arcs:
        out_arcs[a['from']].append(a)
        in_arcs[a['to']].append(a)
        adj_out[a['from']].add(a['to'])
        adj_in[a['to']].add(a['from'])

    arc_map = {a['id']: a for a in arcs}

    # Reachability caches
    fwd_cache = {}

    def reachable_from(start):
        if start in fwd_cache:
            return fwd_cache[start]
        visited = set()
        stack = [start]
        while stack:
            n = stack.pop()
            if n in visited:
                continue
            visited.add(n)
            for nb in adj_out.get(n, []):
                if nb not in visited:
                    stack.append(nb)
        fwd_cache[start] = visited
        return visited

    bwd_cache = {}

    def can_reach(end):
        if end in bwd_cache:
            return bwd_cache[end]
        visited = set()
        stack = [end]
        while stack:
            n = stack.pop()
            if n in visited:
                continue
            visited.add(n)
            for nb in adj_in.get(n, []):
                if nb not in visited:
                    stack.append(nb)
        bwd_cache[end] = visited
        return visited

    # Usable arcs per commodity (on some path from origin to destination)
    commodity_arcs = {}
    for c in commodities:
        fwd = reachable_from(c['origin'])
        bwd = can_reach(c['destination'])
        commodity_arcs[c['id']] = [a for a in arcs if a['from'] in fwd and a['to'] in bwd]

    m = Model("freight")
    m.Params.Seed = 0
    m.Params.MIPGap = 1e-4
    m.Params.NumericFocus = 0
    m.Params.Threads = 1
    m.Params.TimeLimit = max(args.time_limit - 1, 1)
    m.Params.OutputFlag = 0

    # Flow variables x[commodity_id, arc_id]
    flow_pairs = []
    for c in commodities:
        for a in commodity_arcs[c['id']]:
            flow_pairs.append((c['id'], a['id']))

    x = {}
    for cid, aid in flow_pairs:
        x[cid, aid] = m.addVar(lb=0, obj=arc_map[aid]['flow_cost'],
                               name=f"x_{cid}_{aid}")

    # Vehicle variables y[fleet_type_id, arc_id]
    y = {}
    for t in fleet_types:
        for a in arcs:
            y[t['id'], a['id']] = m.addVar(vtype=GRB.INTEGER, lb=0,
                                           obj=t['fixed_cost'],
                                           name=f"y_{t['id']}_{a['id']}")

    m.update()

    # Flow balance constraints
    for c in commodities:
        usable_ids = set(a['id'] for a in commodity_arcs[c['id']])
        for n in nodes:
            outflow = quicksum(x[c['id'], a['id']] for a in out_arcs[n] if a['id'] in usable_ids)
            inflow = quicksum(x[c['id'], a['id']] for a in in_arcs[n] if a['id'] in usable_ids)
            if n == c['origin']:
                rhs = c['demand']
            elif n == c['destination']:
                rhs = -c['demand']
            else:
                rhs = 0
            m.addConstr(outflow - inflow == rhs, name=f"fb_{c['id']}_{n}")

    # Capacity constraints
    for a in arcs:
        flow_terms = [x[c['id'], a['id']] for c in commodities if (c['id'], a['id']) in x]
        if flow_terms:
            cap_terms = [t['capacity'] * y[t['id'], a['id']] for t in fleet_types]
            m.addConstr(quicksum(flow_terms) <= quicksum(cap_terms), name=f"cap_{a['id']}")

    # Vehicle balance constraints (circulation per fleet type)
    for t in fleet_types:
        for n in nodes:
            out_v = quicksum(y[t['id'], a['id']] for a in out_arcs[n])
            in_v = quicksum(y[t['id'], a['id']] for a in in_arcs[n])
            m.addConstr(out_v == in_v, name=f"vb_{t['id']}_{n}")

    # Prepare variable lists for callback
    var_list_x = [x[cid, aid] for cid, aid in flow_pairs]
    var_info_x = [(cid, arc_map[aid]['from'], arc_map[aid]['to']) for cid, aid in flow_pairs]

    var_list_y = []
    var_info_y = []
    for t in fleet_types:
        for a in arcs:
            var_list_y.append(y[t['id'], a['id']])
            var_info_y.append((t['id'], a['from'], a['to']))

    best = [float('inf')]
    best_sol = [None]

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj < best[0] - 1e-9:
                best[0] = obj
                xvals = model.cbGetSolution(var_list_x)
                yvals = model.cbGetSolution(var_list_y)
                flows = {}
                for (cid, fr, to), val in zip(var_info_x, xvals):
                    if val > 1e-6:
                        flows[f"x_{cid}_{fr}_{to}"] = float(val)
                vehicles = {}
                for (tid, fr, to), val in zip(var_info_y, yvals):
                    if val >= 0.5:
                        vehicles[f"y_{tid}_{fr}_{to}"] = int(round(val))
                sol_dict = {"objective_value": float(obj), "flows": flows, "vehicles": vehicles}
                best_sol[0] = sol_dict
                if logger:
                    logger.log_solution(float(obj), sol_dict)

    m.optimize(callback)

    # Build final solution from Gurobi's best incumbent if available
    sol_dict = best_sol[0]
    if m.SolCount > 0:
        obj_val = m.ObjVal
        flows = {}
        for cid, aid in flow_pairs:
            val = x[cid, aid].X
            if val is not None and val > 1e-6:
                a = arc_map[aid]
                flows[f"x_{cid}_{a['from']}_{a['to']}"] = float(val)
        vehicles = {}
        for t in fleet_types:
            for a in arcs:
                val = y[t['id'], a['id']].X
                if val is not None and val >= 0.5:
                    vehicles[f"y_{t['id']}_{a['from']}_{a['to']}"] = int(round(val))
        sol_dict = {"objective_value": float(obj_val), "flows": flows, "vehicles": vehicles}

    if sol_dict is None:
        sol_dict = {"objective_value": 0.0, "flows": {}, "vehicles": {}}

    with open(args.solution_path, 'w') as f:
        json.dump(sol_dict, f, indent=2)


if __name__ == '__main__':
    main()