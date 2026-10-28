import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
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

    source = instance['source_node']
    target = instance['target_node']
    nodes = instance['nodes']
    arcs = instance['arcs']
    linear_costs = instance['linear_costs']
    quadratic_costs = instance['quadratic_costs']

    num_arcs = len(arcs)
    arc_list = list(range(num_arcs))

    # Build adjacency and node mapping
    out_arcs = {node['id']: [] for node in nodes}
    in_arcs = {node['id']: [] for node in nodes}
    arc_from_to = {}
    for a in arcs:
        aid = a['id']
        u = a['from_node']
        v = a['to_node']
        out_arcs[u].append(aid)
        in_arcs[v].append(aid)
        arc_from_to[aid] = (u, v)

    # Create model
    model = gp.Model("QuadraticPath")
    model.setParam("Seed", 0)
    model.setParam("MIPGap", 1e-4)
    model.setParam("NumericFocus", 0)
    model.setParam("Threads", 1)
    model.Params.TimeLimit = args.time_limit
    model.Params.LogToConsole = 0

    # Binary variables: x[a] = 1 if arc a is used
    x = model.addVars(arc_list, vtype=GRB.BINARY, name="x")

    # Linear part
    linear_expr = gp.quicksum(linear_costs[a] * x[a] for a in arc_list)

    # Quadratic part: since quadratic_costs is symmetric, we model full xQx
    # Use gurobipy's quad terms
    qexpr = gp.QuadExpr()
    for i in arc_list:
        for j in arc_list:
            if quadratic_costs[i][j] != 0:
                qexpr.add(x[i] * x[j] * quadratic_costs[i][j])

    objective = linear_expr + qexpr
    model.setObjective(objective, GRB.MINIMIZE)

    # Flow conservation
    for node in nodes:
        nid = node['id']
        if nid == source:
            model.addConstr(gp.quicksum(x[a] for a in out_arcs[nid]) == 1, f"out_source_{nid}")
            model.addConstr(gp.quicksum(x[a] for a in in_arcs[nid]) == 0, f"in_source_{nid}")
        elif nid == target:
            model.addConstr(gp.quicksum(x[a] for a in in_arcs[nid]) == 1, f"in_target_{nid}")
            model.addConstr(gp.quicksum(x[a] for a in out_arcs[nid]) == 0, f"out_target_{nid}")
        else:
            inflow = gp.quicksum(x[a] for a in in_arcs[nid])
            outflow = gp.quicksum(x[a] for a in out_arcs[nid])
            model.addConstr(inflow == outflow, f"flow_{nid}")

    # Callback to log incumbent solutions
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if logger:
                logger.log(float(obj))

    # Optimize
    start_time = time.time()
    model.optimize(callback)

    # Extract solution
    if model.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED):
        if model.SolCount > 0:
            best_obj = model.ObjVal
            selected = []
            for a in arc_list:
                if x[a].X > 0.5:
                    arc_info = arcs[a]
                    selected.append({
                        "id": arc_info['id'],
                        "from_node": arc_info['from_node'],
                        "to_node": arc_info['to_node']
                    })
            # Sort by from_node for nicer output (optional)
            selected.sort(key=lambda e: e['from_node'])

            solution = {
                "objective_value": float(best_obj),
                "solution_arcs": selected
            }

            with open(args.solution_path, 'w') as f:
                json.dump(solution, f, indent=2)
        else:
            # No solution found
            solution = {
                "objective_value": 0.0,
                "solution_arcs": []
            }
            with open(args.solution_path, 'w') as f:
                json.dump(solution, f, indent=2)
    else:
        # Infeasible or other status
        solution = {
            "objective_value": 0.0,
            "solution_arcs": []
        }
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()