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
        data = json.load(f)

    num_nodes = data['num_nodes']
    num_arcs = data['num_arcs']
    num_commodities = data['num_commodities']
    num_fleet_types = data['num_fleet_types']
    nodes = data['nodes']
    arcs = data['arcs']
    commodities = data['commodities']
    fleet_types = data['fleet_types']

    # Build mappings
    arc_list = [(a['from'], a['to']) for a in arcs]
    arc_id_to_idx = {a['id']: i for i, a in enumerate(arcs)}
    arc_from_to_to_flow_cost = {(a['from'], a['to']): a['flow_cost'] for a in arcs}

    comm_list = [(c['id'], c['origin'], c['destination'], c['demand']) for c in commodities]
    fleet_list = [(f['id'], f['capacity'], f['fixed_cost']) for f in fleet_types]

    # Create model
    model = gp.Model("ServiceNetworkDesign")
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('Threads', 1)
    model.setParam('MIPGap', 0.01)  # Allow 1% gap for efficiency
    model.setParam('LogToConsole', 0)

    # Variables
    # Flow: x[k][i][j] for commodity k on arc (i,j)
    x = {}
    for k_idx in range(num_commodities):
        k_id, _, _, _ = comm_list[k_idx]
        x[k_idx] = {}
        for a_idx in range(num_arcs):
            i, j = arc_list[a_idx]
            x[k_idx][a_idx] = model.addVar(
                lb=0.0,
                vtype=GRB.CONTINUOUS,
                name=f"x_{k_id}_{i}_{j}"
            )

    # Vehicles: y[f][a] integer
    y = {}
    for f_idx in range(num_fleet_types):
        f_id, _, _ = fleet_list[f_idx]
        y[f_idx] = {}
        for a_idx in range(num_arcs):
            i, j = arc_list[a_idx]
            y[f_idx][a_idx] = model.addVar(
                lb=0.0,
                vtype=GRB.INTEGER,
                name=f"y_{f_id}_{i}_{j}"
            )

    # Objective: flow costs + fixed vehicle costs
    obj = gp.LinExpr()
    # Flow costs - note that flow_cost is same for all commodities per arc as per schema
    for k_idx in range(num_commodities):
        _, _, _, demand = comm_list[k_idx]
        for a_idx in range(num_arcs):
            i, j = arc_list[a_idx]
            flow_cost = arc_from_to_to_flow_cost[(i, j)]
            obj.addTerms(flow_cost * demand, x[k_idx][a_idx])  # since we scale by demand later? No, we use fraction actually.

    # Correction: better to model as fraction of demand to avoid large numbers, but since demands are given, we use absolute.
    # Actually the schema has per-commodity unit flow cost? Wait, in description it says "each combination of commodity and arc has a known unit flow cost"
    # But in the instance_schema.json it shows only one "flow_cost" per arc. So we assume it's the same for all commodities.
    # To match description, but data only has per arc. We follow data.

    # Fixed costs
    for f_idx in range(num_fleet_types):
        _, _, fixed_cost = fleet_list[f_idx]
        for a_idx in range(num_arcs):
            obj.addTerms(fixed_cost, y[f_idx][a_idx])

    model.setObjective(obj, GRB.MINIMIZE)

    # Capacity constraints
    for a_idx in range(num_arcs):
        i, j = arc_list[a_idx]
        cap_expr = gp.LinExpr()
        for f_idx in range(num_fleet_types):
            _, capacity, _ = fleet_list[f_idx]
            cap_expr.addTerms(capacity, y[f_idx][a_idx])
        flow_expr = gp.LinExpr()
        for k_idx in range(num_commodities):
            _, _, _, demand = comm_list[k_idx]
            flow_expr.addTerms(demand, x[k_idx][a_idx])
        model.addConstr(flow_expr <= cap_expr, name=f"cap_{i}_{j}")

    # Flow conservation for each commodity
    for k_idx in range(num_commodities):
        k_id, origin, destination, demand = comm_list[k_idx]
        for n_idx in range(num_nodes):
            n = nodes[n_idx]
            inflow = gp.LinExpr()
            outflow = gp.LinExpr()
            # Find incoming and outgoing arcs
            for a_idx in range(num_arcs):
                i, j = arc_list[a_idx]
                if j == n:
                    inflow.addTerms(1.0, x[k_idx][a_idx])
                if i == n:
                    outflow.addTerms(1.0, x[k_idx][a_idx])
            if n == origin:
                model.addConstr(outflow - inflow == 1.0, name=f"flow_bal_{k_id}_{n}")
            elif n == destination:
                model.addConstr(outflow - inflow == -1.0, name=f"flow_bal_{k_id}_{n}")
            else:
                model.addConstr(outflow - inflow == 0.0, name=f"flow_bal_{k_id}_{n}")

    # Vehicle circulation constraints for each fleet type
    for f_idx in range(num_fleet_types):
        f_id, _, _ = fleet_list[f_idx]
        for n_idx in range(num_nodes):
            n = nodes[n_idx]
            inflow = gp.LinExpr()
            outflow = gp.LinExpr()
            for a_idx in range(num_arcs):
                i, j = arc_list[a_idx]
                if j == n:
                    inflow.addTerms(1.0, y[f_idx][a_idx])
                if i == n:
                    outflow.addTerms(1.0, y[f_idx][a_idx])
            model.addConstr(inflow == outflow, name=f"veh_bal_{f_id}_{n}")

    # Callback to log incumbent solutions
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if logger:
                logger.log(float(obj_val))

    # Optimize
    model.optimize(callback)

    # Extract solution
    if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED]:
        objective_value = model.objVal if model.objVal < GRB.INFINITY else 0.0

        flows = {}
        for k_idx in range(num_commodities):
            k_id, o, d, _ = comm_list[k_idx]
            for a_idx in range(num_arcs):
                i, j = arc_list[a_idx]
                val = x[k_idx][a_idx].X
                if abs(val) > 1e-6:  # only record non-zero
                    key = f"x_{k_id}_{i}_{j}"
                    flows[key] = float(val)

        vehicles = {}
        for f_idx in range(num_fleet_types):
            f_id, _, _ = fleet_list[f_idx]
            for a_idx in range(num_arcs):
                i, j = arc_list[a_idx]
                val = y[f_idx][a_idx].X
                if abs(val) > 1e-6:
                    key = f"y_{f_id}_{i}_{j}"
                    vehicles[key] = int(round(val))

        solution = {
            "objective_value": float(objective_value),
            "flows": flows,
            "vehicles": vehicles
        }

        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
    else:
        # Infeasible or other - write empty
        solution = {
            "objective_value": 0.0,
            "flows": {},
            "vehicles": {}
        }
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()