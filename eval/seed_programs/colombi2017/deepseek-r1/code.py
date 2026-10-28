import gurobipy as gp
from gurobipy import GRB
import json
import argparse
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    logger = None
    if args.log_path:
        logger = SolutionLogger(args.log_path, sense="maximize")

    num_nodes = data['num_nodes']
    depot = data['depot']
    arcs_list = data['arcs']
    profitable_arcs_list = data['profitable_arcs']
    VI_nodes = data['VI_nodes']
    strong_incompatibilities = data['strong_incompatibilities']
    weak_incompatibilities = data['weak_incompatibilities']

    arc_cost = {}
    outgoing_arcs_dict = {i: [] for i in range(num_nodes)}
    incoming_arcs_dict = {i: [] for i in range(num_nodes)}
    for arc in arcs_list:
        i, j, cost = arc
        arc_cost[(i, j)] = cost
        outgoing_arcs_dict[i].append((j, cost))
        incoming_arcs_dict[j].append((i, cost))

    profitable_arcs_from_node = {i: [] for i in range(num_nodes)}
    for arc in profitable_arcs_list:
        i, j, profit = arc
        profitable_arcs_from_node[i].append((j, profit))

    model = gp.Model()
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', args.time_limit)

    x = {}
    for arc in arcs_list:
        i, j, cost = arc
        x[(i, j)] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"x_{i}_{j}")

    y = {}
    for arc in profitable_arcs_list:
        i, j, profit = arc
        y[(i, j)] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{j}")

    z = {}
    for node in VI_nodes:
        z[node] = model.addVar(vtype=GRB.BINARY, name=f"z_{node}")

    weak_edges_list = []
    for edge in weak_incompatibilities:
        i, j, penalty = edge
        w_var = model.addVar(vtype=GRB.BINARY, name=f"w_{i}_{j}")
        weak_edges_list.append((i, j, penalty, w_var))

    for node in range(num_nodes):
        outgoing_vars = [x[(node, j)] for (j, cost) in outgoing_arcs_dict[node]]
        incoming_vars = [x[(i, node)] for (i, cost) in incoming_arcs_dict[node]]
        model.addConstr(gp.quicksum(outgoing_vars) == gp.quicksum(incoming_vars), f"flow_cons_{node}")

    for arc in profitable_arcs_list:
        i, j, profit = arc
        model.addConstr(y[(i, j)] <= x[(i, j)], f"serve_once_{i}_{j}")

    for node in VI_nodes:
        if profitable_arcs_from_node[node]:
            expr = gp.quicksum(y[(node, j)] for (j, profit) in profitable_arcs_from_node[node])
            model.addConstr(z[node] <= expr, f"z_upper_{node}")
            for (j, profit) in profitable_arcs_from_node[node]:
                model.addConstr(z[node] >= y[(node, j)], f"z_lower_{node}_{j}")
        else:
            model.addConstr(z[node] == 0, f"z_zero_{node}")

    for edge in strong_incompatibilities:
        i, j = edge
        model.addConstr(z[i] + z[j] <= 1, f"strong_{i}_{j}")

    for (i, j, penalty, w_var) in weak_edges_list:
        model.addConstr(z[i] + z[j] <= 1 + w_var, f"weak_{i}_{j}")

    total_profit = gp.quicksum(y[(i, j)] * profit for (i, j, profit) in profitable_arcs_list)
    total_travel_cost = gp.quicksum(x[(i, j)] * cost for (i, j, cost) in arcs_list)
    total_penalty = gp.quicksum(w_var * penalty for (i, j, penalty, w_var) in weak_edges_list)
    model.setObjective(total_profit - total_travel_cost - total_penalty, GRB.MAXIMIZE)

    model._arcs = arcs_list
    model._profitable_arcs = profitable_arcs_list
    model._arc_cost = arc_cost
    model._x = x
    model._y = y
    model._logger = logger

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            arcs_list = model._arcs
            profitable_arcs_list = model._profitable_arcs
            arc_cost = model._arc_cost
            x_dict = model._x
            y_dict = model._y
            logger = model._logger

            served_arcs = []
            for arc in profitable_arcs_list:
                i, j, profit = arc
                y_val = model.cbGetSolution(y_dict[(i, j)])
                if y_val >= 0.5:
                    cost_val = arc_cost[(i, j)]
                    served_arcs.append({'from': i, 'to': j, 'profit': profit, 'cost': cost_val})

            tour_arcs = []
            for arc in arcs_list:
                i, j, cost_val = arc
                x_val = model.cbGetSolution(x_dict[(i, j)])
                if x_val >= 1:
                    tour_arcs.append({'from': i, 'to': j, 'count': x_val, 'cost': cost_val})

            solution_dict = {
                "objective_value": obj,
                "served_arcs": served_arcs,
                "tour_arcs": tour_arcs
            }

            if logger:
                logger.log_solution(obj, solution_dict)

    if logger:
        model.optimize(callback)
    else:
        model.optimize()

    if model.SolCount > 0:
        obj_val = model.objVal
        served_arcs = []
        for arc in profitable_arcs_list:
            i, j, profit = arc
            if y[(i, j)].X >= 0.5:
                cost_val = arc_cost[(i, j)]
                served_arcs.append({'from': i, 'to': j, 'profit': profit, 'cost': cost_val})

        tour_arcs = []
        for arc in arcs_list:
            i, j, cost_val = arc
            count = x[(i, j)].X
            if count >= 1:
                tour_arcs.append({'from': i, 'to': j, 'count': count, 'cost': cost_val})

        solution_dict = {
            "objective_value": obj_val,
            "served_arcs": served_arcs,
            "tour_arcs": tour_arcs
        }

        with open(args.solution_path, 'w') as f:
            json.dump(solution_dict, f)

if __name__ == '__main__':
    main()