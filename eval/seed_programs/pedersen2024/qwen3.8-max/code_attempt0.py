import json
import argparse
import sys
from gurobipy import *

def main():
    parser = argparse.ArgumentParser(description='Wind Farm Network Optimization')
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Read instance
    with open(args.instance_path) as f:
        instance = json.load(f)

    params = instance['parameters']
    nodes = instance['nodes']
    edges = instance['edges']
    quota = params['quota']
    alpha = params['alpha']
    n = params['total_nodes']

    # Classify nodes
    node_map = {nd['id']: nd for nd in nodes}
    substations = [nd for nd in nodes if nd['type'] == 'substation']
    turbines = [nd for nd in nodes if nd['type'] == 'potential_turbine']
    steiners = [nd for nd in nodes if nd['type'] == 'steiner']

    num_sub = len(substations)
    root_id = substations[0]['id'] if substations else None

    # Build adjacency info for flow constraints
    from_edges = {nd['id']: [] for nd in nodes}  # edge IDs where this node is 'from'
    to_edges = {nd['id']: [] for nd in nodes}    # edge IDs where this node is 'to'
    for e in edges:
        from_edges[e['from']].append(e['id'])
        to_edges[e['to']].append(e['id'])

    # Create Gurobi model
    model = Model("wind_farm")
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('OutputFlag', 0)

    # Decision variables
    x = {}
    for nd in turbines:
        x[nd['id']] = model.addVar(vtype=GRB.BINARY, name=f"x_{nd['id']}")

    z = {}
    for nd in steiners:
        z[nd['id']] = model.addVar(vtype=GRB.BINARY, name=f"z_{nd['id']}")

    y = {}
    for e in edges:
        y[e['id']] = model.addVar(vtype=GRB.BINARY, name=f"y_{e['id']}")

    # Flow variables: f_fwd[eid] = flow from 'from' to 'to', f_bwd[eid] = flow from 'to' to 'from'
    f_fwd = {}
    f_bwd = {}
    for e in edges:
        f_fwd[e['id']] = model.addVar(lb=0, ub=n, name=f"ff_{e['id']}")
        f_bwd[e['id']] = model.addVar(lb=0, ub=n, name=f"fb_{e['id']}")

    model.update()

    # Helper: returns expression indicating if node is active
    def active(nid):
        nd = node_map[nid]
        if nd['type'] == 'substation':
            return 1
        elif nd['type'] == 'potential_turbine':
            return x[nid]
        else:
            return z[nid]

    # Edge-node incidence: edge can only be selected if both endpoints are active
    for e in edges:
        model.addConstr(y[e['id']] <= active(e['from']), f"inc1_{e['id']}")
        model.addConstr(y[e['id']] <= active(e['to']), f"inc2_{e['id']}")

    # Tree property: |E_selected| = |V_active| - 1
    model.addConstr(
        quicksum(y[e['id']] for e in edges) ==
        (num_sub - 1) + quicksum(x[t['id']] for t in turbines) + quicksum(z[s['id']] for s in steiners),
        "tree_property"
    )

    # Flow conservation constraints
    for nd in nodes:
        nid = nd['id']
        # Outgoing flow from nid
        out_flow = quicksum(f_fwd[eid] for eid in from_edges[nid]) + \
                   quicksum(f_bwd[eid] for eid in to_edges[nid])
        # Incoming flow to nid
        in_flow = quicksum(f_fwd[eid] for eid in to_edges[nid]) + \
                  quicksum(f_bwd[eid] for eid in from_edges[nid])

        if nid == root_id:
            # Root sends out (total_active - 1) units
            rhs = (num_sub - 1) + quicksum(x[t['id']] for t in turbines) + \
                  quicksum(z[s['id']] for s in steiners)
            model.addConstr(out_flow - in_flow == rhs, "flow_root")
        else:
            # Non-root: net inflow = 1 if active, 0 if inactive
            model.addConstr(in_flow - out_flow == active(nid), f"flow_{nid}")

    # Flow capacity: flow only on selected edges
    for e in edges:
        model.addConstr(f_fwd[e['id']] <= (n - 1) * y[e['id']], f"capf_{e['id']}")
        model.addConstr(f_bwd[e['id']] <= (n - 1) * y[e['id']], f"capb_{e['id']}")

    # Energy quota constraint
    if turbines:
        model.addConstr(
            quicksum(nd['profit'] * x[nd['id']] for nd in turbines) >= quota,
            "quota"
        )
    elif quota > 0:
        # Infeasible: no turbines but positive quota
        model.addConstr(0 >= quota, "infeasible_quota")

    # Objective: minimize alpha * financial_cost + (1-alpha) * scenic_impact
    obj = LinExpr()
    for e in edges:
        combined = alpha * e['cost'] + (1.0 - alpha) * e['scenic_impact']
        obj += combined * y[e['id']]
    for nd in turbines:
        combined = alpha * nd['cost'] + (1.0 - alpha) * nd['scenic_impact']
        obj += combined * x[nd['id']]

    model.setObjective(obj, GRB.MINIMIZE)

    # Callback to log incumbent solutions
    best_logged_obj = [float('inf')]

    def callback(m, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = m.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj_val < best_logged_obj[0] - 1e-9:
                best_logged_obj[0] = obj_val
                try:
                    x_vals = m.cbGetSolution([x[t['id']] for t in turbines]) if turbines else []
                    y_vals = m.cbGetSolution([y[e['id']] for e in edges]) if edges else []

                    sel_turbines = [turbines[i]['id'] for i in range(len(turbines)) if x_vals[i] > 0.5]
                    sel_edges = [{"from": edges[i]['from'], "to": edges[i]['to']}
                                 for i in range(len(edges)) if y_vals[i] > 0.5]

                    solution = {
                        "objective_value": obj_val,
                        "selected_turbines": sel_turbines,
                        "selected_edges": sel_edges
                    }

                    if logger:
                        logger.log_solution(obj_val, solution)
                except Exception:
                    if logger:
                        logger.log(obj_val)

    # Optimize
    model.optimize(callback)

    # Extract final solution
    if model.SolCount > 0:
        selected_turbines = [nd['id'] for nd in turbines if x[nd['id']].X > 0.5]
        selected_edges = [{"from": e['from'], "to": e['to']} for e in edges if y[e['id']].X > 0.5]
        objective_value = model.ObjVal

        solution = {
            "objective_value": objective_value,
            "selected_turbines": selected_turbines,
            "selected_edges": selected_edges
        }

        # Log final if not already logged via callback
        if logger and objective_value >= best_logged_obj[0] - 1e-9:
            logger.log_solution(objective_value, solution)
    else:
        # No feasible solution found
        solution = {
            "objective_value": 0.0,
            "selected_turbines": [],
            "selected_edges": []
        }

    # Write solution
    with open(args.solution_path, 'w') as f_out:
        json.dump(solution, f_out, indent=2)


if __name__ == '__main__':
    main()