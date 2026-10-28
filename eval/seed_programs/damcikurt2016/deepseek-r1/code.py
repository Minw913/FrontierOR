import os
os.environ['GUROBI_PYTHON_SKIP_DBUS'] = '1'

import json
import argparse
import gurobipy as gp
from collections import deque
from solution_logger import SolutionLogger

def compute_min_set(node, horizon):
    min_set = []
    start_time = node['time']
    queue = deque([node])
    while queue:
        current = queue.popleft()
        if current['time'] <= start_time + horizon:
            min_set.append(current)
            for child in current['children']:
                queue.append(child)
    return min_set

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    gen = instance['generator']
    tree = instance['scenario_tree']
    nodes_data = tree['nodes']

    C_lower = gen['C_lower_MW']
    C_upper = gen['C_upper_MW']
    V_plus = gen['V_plus_MW_per_h']
    V_minus = gen['V_minus_MW_per_h']
    L_min_up_time = tree['L_min_up_time']
    ell_min_down_time = tree['ell_min_down_time']
    startup_cost = gen['U_bar_startup_cost_dollars']
    shutdown_cost = gen['U_lower_shutdown_cost_dollars']
    fuel_a = gen['fuel_a_dollars_per_h']
    fuel_b = gen['fuel_b_dollars_per_MWh']
    initial_status = tree['initial_generator_status_y0']
    initial_generation = tree['initial_generation_x0_MW']

    node_dict = {}
    root_id = None
    for node in nodes_data:
        node_id = node['id']
        node_dict[node_id] = {
            'id': node_id,
            'time': node['time'],
            'parent_id': node['parent_id'],
            'children_ids': node['children_ids'],
            'probability': node['probability'],
            'price': node['electricity_price_dollars_per_MWh'],
            'children': [],
            'parent': None
        }
        if node['parent_id'] is None:
            root_id = node_id

    for node_id, node in node_dict.items():
        parent_id = node['parent_id']
        if parent_id is not None:
            node['parent'] = node_dict[parent_id]
        for child_id in node['children_ids']:
            node['children'].append(node_dict[child_id])
    root = node_dict[root_id]

    min_up_sets = {}
    min_down_sets = {}
    for node_id, node in node_dict.items():
        min_up_sets[node_id] = compute_min_set(node, L_min_up_time - 1)
        min_down_sets[node_id] = compute_min_set(node, ell_min_down_time - 1)

    model = gp.Model()
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('LogToConsole', 0)

    y = {}
    x = {}
    u = {}
    for node_id, node in node_dict.items():
        y[node_id] = model.addVar(vtype=gp.GRB.BINARY, name=f"y_{node_id}")
        x[node_id] = model.addVar(lb=0, ub=C_upper, name=f"x_{node_id}")
        if node_id != root_id:
            u[node_id] = model.addVar(vtype=gp.GRB.BINARY, name=f"u_{node_id}")

    obj = 0
    root_coeff = -fuel_a - startup_cost * (1 - initial_status)
    obj += root['probability'] * ((root['price'] - fuel_b) * x[root_id] + root_coeff * y[root_id])

    for node_id, node in node_dict.items():
        if node_id == root_id:
            continue
        parent_id = node['parent_id']
        coeff_y = -fuel_a + shutdown_cost
        term = (node['price'] - fuel_b) * x[node_id] + coeff_y * y[node_id] - shutdown_cost * y[parent_id] - (startup_cost + shutdown_cost) * u[node_id]
        obj += node['probability'] * term

    model.setObjective(obj, gp.GRB.MAXIMIZE)

    for node_id, node in node_dict.items():
        model.addConstr(x[node_id] >= C_lower * y[node_id], name=f"gen_lb_{node_id}")
        model.addConstr(x[node_id] <= C_upper * y[node_id], name=f"gen_ub_{node_id}")

    model.addConstr(x[root_id] - initial_generation <= V_plus * initial_status + C_upper * (1 - initial_status), name="ramp_up_root")
    model.addConstr(initial_generation - x[root_id] <= V_minus * y[root_id] + C_upper * (1 - y[root_id]), name="ramp_down_root")

    for node_id, node in node_dict.items():
        if node_id == root_id:
            continue
        parent_id = node['parent_id']
        model.addConstr(u[node_id] >= y[node_id] - y[parent_id], name=f"u_lb_{node_id}")
        model.addConstr(u[node_id] <= y[node_id], name=f"u_ub1_{node_id}")
        model.addConstr(u[node_id] <= 1 - y[parent_id], name=f"u_ub2_{node_id}")

        for j_node in min_up_sets[node_id]:
            j_id = j_node['id']
            model.addConstr(y[j_id] >= u[node_id], name=f"min_up_{node_id}_{j_id}")

        for j_node in min_down_sets[node_id]:
            j_id = j_node['id']
            model.addConstr(y[j_id] <= 1 - (y[parent_id] - y[node_id] + u[node_id]), name=f"min_down_{node_id}_{j_id}")

        parent = node['parent']
        model.addConstr(x[node_id] - x[parent_id] <= V_plus * y[parent_id] + C_upper * (1 - y[parent_id]), name=f"ramp_up_{node_id}")
        model.addConstr(x[parent_id] - x[node_id] <= V_minus * y[node_id] + C_upper * (1 - y[node_id]), name=f"ramp_down_{node_id}")

    if initial_status == 0:
        for j_node in min_up_sets[root_id]:
            j_id = j_node['id']
            model.addConstr(y[j_id] >= y[root_id], name=f"min_up_root_{j_id}")
    else:
        for j_node in min_down_sets[root_id]:
            j_id = j_node['id']
            model.addConstr(y[j_id] <= y[root_id], name=f"min_down_root_{j_id}")

    def callback(model, where):
        if where == gp.GRB.Callback.MIPSOL:
            objval = model.cbGet(gp.GRB.Callback.MIPSOL_OBJ)
            x_vals = {}
            y_vals = {}
            u_vals = {}
            for node_id in node_dict:
                x_vals[str(node_id)] = model.cbGetSolution(x[node_id])
                y_vals[str(node_id)] = model.cbGetSolution(y[node_id])
                if node_id != root_id:
                    u_vals[str(node_id)] = model.cbGetSolution(u[node_id])
            solution_dict = {
                "objective_value": objval,
                "x": x_vals,
                "y": y_vals,
                "u": u_vals
            }
            if logger:
                logger.log_solution(objval, solution_dict)

    model.optimize(callback if logger else None)

    solution = {
        "objective_value": 0.0,
        "x": {},
        "y": {},
        "u": {}
    }

    if model.SolCount > 0:
        solution['objective_value'] = model.ObjVal
        for node_id in node_dict:
            solution['x'][str(node_id)] = x[node_id].X
            solution['y'][str(node_id)] = y[node_id].X
            if node_id != root_id:
                solution['u'][str(node_id)] = u[node_id].X
    else:
        for node_id in node_dict:
            solution['x'][str(node_id)] = 0.0
            solution['y'][str(node_id)] = 0.0
            if node_id != root_id:
                solution['u'][str(node_id)] = 0.0

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()