import argparse
import json
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

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    arcs = instance['arcs']
    seen_arcs = {}
    for arc in arcs:
        key = (arc['from'], arc['to'])
        if key in seen_arcs:
            raise Exception(f"Duplicate arc found: {key}")
        seen_arcs[key] = arc['flow_cost']

    arcs_list = list(seen_arcs.keys())
    node_set = set(instance['nodes'])
    commodities = instance['commodities']
    fleet_types = instance['fleet_types']

    model = gp.Model("FreightNetwork")
    model.setParam('OutputFlag', 0)
    model.setParam('Threads', 1)

    x = {}
    for k in commodities:
        k_id = k['id']
        for (i, j) in arcs_list:
            var_name = f"x_{k_id}_{i}_{j}"
            x[k_id, i, j] = model.addVar(lb=0.0, name=var_name)

    y = {}
    for t in fleet_types:
        t_id = t['id']
        for (i, j) in arcs_list:
            var_name = f"y_{t_id}_{i}_{j}"
            y[t_id, i, j] = model.addVar(vtype=GRB.INTEGER, lb=0, name=var_name)

    model.update()

    for k in commodities:
        k_id = k['id']
        origin = k['origin']
        dest = k['destination']
        demand = k['demand']
        for node in node_set:
            out_arcs = [(i, j) for (i, j) in arcs_list if i == node]
            in_arcs = [(i, j) for (i, j) in arcs_list if j == node]
            expr = gp.LinExpr()
            for (i, j) in out_arcs:
                expr += x[k_id, i, j]
            for (i, j) in in_arcs:
                expr -= x[k_id, i, j]
            if node == origin:
                model.addConstr(expr == demand, name=f"flow_bal_k{k_id}_n{node}")
            elif node == dest:
                model.addConstr(expr == -demand, name=f"flow_bal_k{k_id}_n{node}")
            else:
                model.addConstr(expr == 0, name=f"flow_bal_k{k_id}_n{node}")

    for (i, j) in arcs_list:
        total_flow = gp.LinExpr()
        for k in commodities:
            k_id = k['id']
            total_flow += x[k_id, i, j]
        total_capacity = gp.LinExpr()
        for t in fleet_types:
            t_id = t['id']
            cap = t['capacity']
            total_capacity += cap * y[t_id, i, j]
        model.addConstr(total_flow <= total_capacity, name=f"cap_{i}_{j}")

    for t in fleet_types:
        t_id = t['id']
        for node in node_set:
            in_arcs = [(i, j) for (i, j) in arcs_list if j == node]
            out_arcs = [(i, j) for (i, j) in arcs_list if i == node]
            expr_in = gp.LinExpr()
            for (i, j) in in_arcs:
                expr_in += y[t_id, i, j]
            expr_out = gp.LinExpr()
            for (i, j) in out_arcs:
                expr_out += y[t_id, i, j]
            model.addConstr(expr_in == expr_out, name=f"veh_bal_t{t_id}_n{node}")

    flow_cost = gp.LinExpr()
    for (i, j) in arcs_list:
        cost_ij = seen_arcs[(i, j)]
        for k in commodities:
            k_id = k['id']
            flow_cost += cost_ij * x[k_id, i, j]

    fixed_cost = gp.LinExpr()
    for t in fleet_types:
        t_id = t['id']
        fixed_cost_t = t['fixed_cost']
        for (i, j) in arcs_list:
            fixed_cost += fixed_cost_t * y[t_id, i, j]

    model.setObjective(flow_cost + fixed_cost, GRB.MINIMIZE)
    model.setParam('TimeLimit', args.time_limit)

    if logger is not None:
        class CallbackState:
            def __init__(self, logger):
                self.logger = logger
                self.best_obj = None
            
            def __call__(self, model, where):
                if where == GRB.Callback.MIPSOL:
                    obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    if self.best_obj is None or obj < self.best_obj:
                        self.best_obj = obj
                        self.logger.log(obj)
        
        cb = CallbackState(logger)
        model.optimize(cb)
    else:
        model.optimize()

    if model.SolCount > 0:
        objective_value = model.objVal
        flows_dict = {}
        for k in commodities:
            k_id = k['id']
            for (i, j) in arcs_list:
                key = f"x_{k_id}_{i}_{j}"
                flows_dict[key] = x[k_id, i, j].X
        
        vehicles_dict = {}
        for t in fleet_types:
            t_id = t['id']
            for (i, j) in arcs_list:
                key = f"y_{t_id}_{i}_{j}"
                vehicles_dict[key] = int(round(y[t_id, i, j].X))
    else:
        objective_value = 1e20
        flows_dict = {}
        for k in commodities:
            k_id = k['id']
            for (i, j) in arcs_list:
                key = f"x_{k_id}_{i}_{j}"
                flows_dict[key] = 0.0
        
        vehicles_dict = {}
        for t in fleet_types:
            t_id = t['id']
            for (i, j) in arcs_list:
                key = f"y_{t_id}_{i}_{j}"
                vehicles_dict[key] = 0

    solution = {
        "objective_value": objective_value,
        "flows": flows_dict,
        "vehicles": vehicles_dict
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == "__main__":
    main()