import json
import argparse
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, default=300)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path) as f:
        data = json.load(f)

    num_customers = data['num_customers']
    num_vehicles = data['num_vehicles']
    vehicle_capacity = data['vehicle_capacity']
    planning_horizon = data['planning_horizon']
    depot = data['depot']
    depot_end = data['depot_end']
    customers = data['customers']
    distance_matrix = data['distance_matrix']
    rho = data['inconvenience_cost_function']['rho']

    model_index_to_id = {}
    for idx, cust in enumerate(customers):
        model_index = idx + 1
        model_index_to_id[model_index] = cust['id']

    nodes = [0] + [i + 1 for i in range(num_customers)] + [num_customers + 1]

    a, b, d, st, pref = {}, {}, {}, {}, {}
    a[0] = depot['time_window'][0]
    b[0] = depot['time_window'][1]
    d[0] = 0
    st[0] = 0
    pref[0] = 0

    a[num_customers + 1] = depot_end['time_window'][0]
    b[num_customers + 1] = depot_end['time_window'][1]
    d[num_customers + 1] = 0
    st[num_customers + 1] = 0
    pref[num_customers + 1] = 0

    for i in range(1, num_customers + 1):
        cust = customers[i - 1]
        a[i] = cust['time_window'][0]
        b[i] = cust['time_window'][1]
        d[i] = cust['demand']
        st[i] = cust['service_time']
        pref[i] = cust['preferred_time']

    model = gp.Model("VRP_with_TimeWindows")
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', args.time_limit)

    arcs = []
    for i in nodes:
        if i == num_customers + 1:
            continue
        for j in nodes:
            if j == i or j == 0:
                continue
            arcs.append((i, j))

    x = model.addVars(arcs, vtype=GRB.BINARY, name="x")
    s = model.addVars(nodes, lb=0, ub=planning_horizon, name="s")
    u_nodes = nodes.copy()
    u_nodes.remove(num_customers + 1)
    u = model.addVars(u_nodes, lb=0, ub=vehicle_capacity, name="u")

    model.addConstr(sum(x[0, j] for (i, j) in arcs if i == 0) <= num_vehicles, "depot_outflow")
    model.addConstr(sum(x[i, num_customers + 1] for (i, j) in arcs if j == num_customers + 1) <= num_vehicles, "depot_end_inflow")

    for k in range(1, num_customers + 1):
        model.addConstr(sum(x[i, k] for (i, j) in arcs if j == k) == 1, f"inflow_{k}")
        model.addConstr(sum(x[k, j] for (i, j) in arcs if i == k) == 1, f"outflow_{k}")

    for (i, j) in arcs:
        M = b[i] + st[i] + distance_matrix[i][j] - a[j]
        model.addConstr(s[i] + st[i] + distance_matrix[i][j] <= s[j] + M * (1 - x[i, j]), f"time_{i}_{j}")

    for i in nodes:
        model.addConstr(s[i] >= a[i], f"tw_lb_{i}")
        model.addConstr(s[i] <= b[i], f"tw_ub_{i}")

    model.addConstr(u[0] == 0, "u0")
    for (i, j) in arcs:
        if j == num_customers + 1:
            continue
        model.addConstr(u[j] >= u[i] + d[j] - vehicle_capacity * (1 - x[i, j]), f"cap_{i}_{j}")

    routing_cost = gp.quicksum(distance_matrix[i][j] * x[i, j] for (i, j) in arcs)
    inconvenience_cost = rho * gp.quicksum((s[i] - pref[i]) * (s[i] - pref[i]) for i in range(1, num_customers + 1))
    total_cost = routing_cost + inconvenience_cost
    model.setObjective(total_cost, GRB.MINIMIZE)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    def mycallback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            x_vals = model.cbGetSolution(x)
            s_vals = model.cbGetSolution(s)

            next_node = {}
            for (i, j) in arcs:
                if x_vals[i, j] > 0.5:
                    next_node[i] = j

            routes = []
            for j in nodes:
                if (0, j) in arcs and x_vals[0, j] > 0.5:
                    route = [0]
                    current = j
                    while current != num_customers + 1:
                        route.append(current)
                        current = next_node.get(current, None)
                        if current is None:
                            break
                    route.append(num_customers + 1)
                    routes.append(route)

            schedule_dict = {}
            for i in range(1, num_customers + 1):
                customer_id = model_index_to_id[i]
                schedule_dict[str(customer_id)] = s_vals[i]

            solution_dict = {
                "objective_value": obj,
                "routes": routes,
                "schedule": schedule_dict
            }

            if logger:
                logger.log_solution(obj, solution_dict)

    model.optimize(mycallback if logger else None)

    if model.SolCount > 0:
        x_vals = model.getAttr('x', x)
        s_vals = model.getAttr('x', s)

        next_node = {}
        for (i, j) in arcs:
            if x_vals[i, j] > 0.5:
                next_node[i] = j

        routes = []
        for j in nodes:
            if (0, j) in arcs and x_vals[0, j] > 0.5:
                route = [0]
                current = j
                while current != num_customers + 1:
                    route.append(current)
                    current = next_node.get(current, None)
                    if current is None:
                        break
                route.append(num_customers + 1)
                routes.append(route)

        schedule_dict = {}
        for i in range(1, num_customers + 1):
            customer_id = model_index_to_id[i]
            schedule_dict[str(customer_id)] = s_vals[i]

        solution_dict = {
            "objective_value": model.ObjVal,
            "routes": routes,
            "schedule": schedule_dict
        }

        with open(args.solution_path, 'w') as f:
            json.dump(solution_dict, f)
    else:
        routes = []
        schedule_dict = {}
        routing_cost_val = 0
        for i in range(1, num_customers + 1):
            route = [0, i, num_customers + 1]
            routes.append(route)
            schedule_dict[str(model_index_to_id[i])] = pref[i]
            routing_cost_val += distance_matrix[0][i] + distance_matrix[i][num_customers + 1]

        inconvenience_cost_val = 0
        total_cost_val = routing_cost_val + inconvenience_cost_val
        solution_dict = {
            "objective_value": total_cost_val,
            "routes": routes,
            "schedule": schedule_dict
        }

        with open(args.solution_path, 'w') as f:
            json.dump(solution_dict, f)

if __name__ == "__main__":
    main()