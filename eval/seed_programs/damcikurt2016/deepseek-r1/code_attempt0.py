import json
import sys
import gurobipy as gp
from collections import deque
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    gen = instance['generator']
    tree = instance['scenario_tree']
    params = instance['parameters']

    C_lower = gen['C_lower']
    C_upper = gen['C_upper']
    V_plus = gen['V_plus']
    V_minus = gen['V_minus']
    L_min_up_time = gen['L_min_up_time']
    ell_min_down_time = gen['ell_min_down_time']
    startup_cost = gen['U_bar_startup_cost_dollars']
    shutdown_cost = gen['U_lower_shutdown_cost_dollars']
    fuel_a = gen['fuel_a_dollars_per_h']
    fuel_b = gen['fuel_b_dollars_per_MWh']
    initial_status = gen['initial_generator_status_y0']
    initial_generation = gen['initial_generation_x0_MW']

    nodes = tree['nodes']
    root_id = tree['root_node_id']
    time_periods = tree['time_periods']

    node_dict = {}
    for node in nodes:
        node_id = node['node_id']
        node_dict[node_id] = {
            'id': node_id,
            'time': node['time_period'],
            'parent_id': node['parent_node_id'],
            'children_ids': node['child_node_ids'],
            'probability': node['probability'],
            'price': node['electricity_price_dollars_per_MWh'],
            'children': [],
            'parent': None
        }

    for node_id, node in node_dict.items():
        if node['parent_id'] is not None:
            node['parent'] = node_dict[node['parent_id']]
        for child_id in node['children_ids']:
            node['children'].append(node_dict[child_id])

    root = node_dict[root_id]

    min_up_dict = {}
    min_down_dict = {}
    for node_id, node in node_dict.items():
        min_up_dict[node_id] = []
        queue = deque([(node, 0)])
        while queue:
            current, steps = queue.popleft()
            min_up_dict[node_id].append(current)
            if steps < L_min_up_time - 1:
                for child in current['children']:
                    queue.append((child, steps + 1))

        min_down_dict[node_id] = []
        queue = deque([(node, 0)])
        while queue:
            current, steps = queue.popleft()
            min_down_dict[node_id].append(current)
            if steps < ell_min_down_time - 1:
                for child in current['children']:
                    queue.append((child, steps + 1))

    min_up_nodes_root = []
    min_down_nodes_root = []
    if initial_status == 0:
        queue = deque([root])
        while queue:
            current = queue.popleft()
            if current['time'] <= L_min_up_time:
                min_up_nodes_root.append(current)
                for child in current['children']:
                    queue.append(child)
    else:
        queue = deque([root])
        while queue:
            current = queue.popleft()
            if current['time'] <= ell_min_down_time:
                min_down_nodes_root.append(current)
                for child in current['children']:
                    queue.append(child)

    model = gp.Model()
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)

    y = {}
    x = {}
    u = {}
    for node_id, node in node_dict.items():
        y[node_id] = model.addVar(vtype=gp.GRB.BINARY, name=f"y_{node_id}")
        x[node_id] = model.addVar(lb=0, ub=C_upper, name=f"x_{node_id}")
        if node_id != root_id:
            u[node_id] = model.addVar(vtype=gp.GRB.BINARY, name=f"u_{node_id}")

    constant_term = -initial_status * shutdown_cost
    obj = constant_term

    root_coeff = -fuel_a - (1 - initial_status) * startup_cost + initial_status * shutdown_cost - len(root['children_ids']) * shutdown_cost
    obj += (root['price'] - fuel_b) * x[root_id] + root_coeff * y[root_id]

    for node_id, node in node_dict.items():
        if node_id == root_id:
            continue
        coeff_y = -fuel_a + shutdown_cost * (1 - len(node['children_ids']))
        obj += (node['price'] - fuel_b) * x[node_id] + coeff_y * y[node_id] - (startup_cost + shutdown_cost) * u[node_id]

    model.setObjective(obj, gp.GRB.MAXIMIZE)

    for node_id, node in node_dict.items():
        model.addConstr(x[node_id] >= C_lower * y[node_id], name=f"gen_lb_{node_id}")
        model.addConstr(x[node_id] <= C_upper * y[node_id], name=f"gen_ub_{node_id}")

    if initial_status == 1:
        model.addConstr(x[root_id] - initial_generation <= V_plus * y[root_id] + C_upper * (1 - y[root_id]), name="ramp_up_root")
        model.addConstr(initial_generation - x[root_id] <= V_minus * y[root_id] + C_upper * (1 - y[root_id]), name="ramp_down_root")

    for node_id, node in node_dict.items():
        if node_id == root_id:
            continue
        parent_id = node['parent_id']
        model.addConstr(u[node_id] >= y[node_id] - y[parent_id], name=f"u_lb_{node_id}")
        model.addConstr(u[node_id] <= y[node_id], name=f"u_ub1_{node_id}")
        model.addConstr(u[node_id] <= 1 - y[parent_id], name=f"u_ub2_{node_id}")

        for j_node in min_up_dict[node_id]:
            j_id = j_node['id']
            model.addConstr(y[j_id] >= u[node_id], name=f"min_up_{node_id}_{j_id}")

        for j_node in min_down_dict[node_id]:
            j_id = j_node['id']
            model.addConstr(y[j_id] <= 1 - (y[parent_id] - y[node_id] + u[node_id]), name=f"min_down_{node_id}_{j_id}")

        parent_id = node['parent_id']
        model.addConstr(x[node_id] - x[parent_id] <= V_plus + C_upper * ((1 - y[parent_id]) + (1 - y[node_id])), name=f"ramp_up_{node_id}")
        model.addConstr(x[parent_id] - x[node_id] <= V_minus + C_upper * ((1 - y[parent_id]) + (1 - y[node_id])), name=f"ramp_down_{node_id}")

    if initial_status == 0:
        for j_node in min_up_nodes_root:
            j_id = j_node['id']
            model.addConstr(y[j_id] >= y[root_id], name=f"min_up_root_{j_id}")
    else:
        for j_node in min_down_nodes_root:
            j_id = j_node['id']
            model.addConstr(y[j_id] <= y[root_id], name=f"min_down_root_{j_id}")

    model.optimize()

    if model.status == gp.GRB.OPTIMAL:
        solution = {
            "objective_value": model.objVal,
            "x": {},
            "y": {},
            "u": {}
        }
        for node_id in node_dict:
            solution['x'][str(node_id)] = x[node_id].X
            solution['y'][str(node_id)] = y[node_id].X
            if node_id != root_id:
                solution['u'][str(node_id)] = u[node_id].X

        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()