import gurobipy as gp
from gurobipy import GRB
import json
import argparse
import os
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
    
    n = instance['num_delivery_sites']
    m = instance['num_customers']
    vehicle_capacity = instance['vehicle_capacity']
    num_vehicles = instance['num_vehicles']
    depot = instance['depot']
    delivery_sites = instance['delivery_sites']
    customers = instance['customers']
    routing_cost_matrix = instance['routing_cost_matrix']
    assignment_cost_matrix = instance['assignment_cost_matrix']
    
    model = gp.Model()
    model.setParam('OutputFlag', 1 if args.log_path else 0)
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('Threads', 1)
    
    y = model.addVars(n, vtype=GRB.BINARY, name="y")
    z = model.addVars(m, n, vtype=GRB.BINARY, name="z")
    x = {}
    for k in range(num_vehicles):
        for i in range(n + 1):
            for j in range(n + 1):
                if i != j:
                    x[k, i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{k}_{i}_{j}")
    q = model.addVars(m, n, num_vehicles, vtype=GRB.BINARY, name="q")
    u = model.addVars(num_vehicles, n + 1, lb=0, ub=vehicle_capacity, name="u")
    
    for c in range(m):
        model.addConstr(gp.quicksum(z[c, j] for j in range(n)) == 1, name=f"assign_{c}")
    
    for c in range(m):
        for j in range(n):
            model.addConstr(z[c, j] <= y[j], name=f"open_{c}_{j}")
    
    for j in range(n):
        total_demand = gp.quicksum(customers[c]['demand'] * z[c, j] for c in range(m))
        model.addConstr(total_demand <= vehicle_capacity, name=f"sitecap_{j}")
    
    for k in range(num_vehicles):
        model.addConstr(gp.quicksum(x[k, 0, j] for j in range(1, n + 1)) == 1, name=f"depot_out_{k}")
        model.addConstr(gp.quicksum(x[k, j, 0] for j in range(1, n + 1)) == 1, name=f"depot_in_{k}")
        for j in range(n):
            in_flow = gp.quicksum(x[k, i, j + 1] for i in range(n + 1) if i != j + 1)
            out_flow = gp.quicksum(x[k, j + 1, i] for i in range(n + 1) if i != j + 1)
            model.addConstr(in_flow == out_flow, name=f"flow_cons_{k}_{j}")
    
    for j in range(n):
        total_visit = gp.quicksum(
            gp.quicksum(x[k, i, j + 1] for i in range(n + 1) if i != j + 1) 
            for k in range(num_vehicles)
        )
        model.addConstr(total_visit <= 1, name=f"site_visit_{j}")
    
    for c in range(m):
        for j in range(n):
            total_visit_j = gp.quicksum(
                gp.quicksum(x[k, i, j + 1] for i in range(n + 1) if i != j + 1)
                for k in range(num_vehicles)
            )
            model.addConstr(z[c, j] <= total_visit_j, name=f"link_assign_routing_{c}_{j}")
    
    for k in range(num_vehicles):
        for c in range(m):
            for j in range(n):
                w_jk = gp.quicksum(x[k, i, j + 1] for i in range(n + 1) if i != j + 1)
                model.addConstr(q[c, j, k] <= z[c, j], name=f"q1_{c}_{j}_{k}")
                model.addConstr(q[c, j, k] <= w_jk, name=f"q2_{c}_{j}_{k}")
                model.addConstr(q[c, j, k] >= z[c, j] + w_jk - 1, name=f"q3_{c}_{j}_{k}")
    
    for k in range(num_vehicles):
        total_demand_k = gp.quicksum(customers[c]['demand'] * q[c, j, k] for c in range(m) for j in range(n))
        model.addConstr(total_demand_k <= vehicle_capacity, name=f"vehcap_{k}")
    
    for k in range(num_vehicles):
        model.addConstr(u[k, 0] == 0, name=f"depot_u_{k}")
    
    for k in range(num_vehicles):
        for j in range(n):
            w_jk = gp.quicksum(x[k, i, j + 1] for i in range(n + 1) if i != j + 1)
            d_j = gp.quicksum(customers[c]['demand'] * z[c, j] for c in range(m))
            model.addConstr(u[k, j + 1] >= d_j - vehicle_capacity * (1 - w_jk), name=f"mtz_lb_{k}_{j}")
    
    for k in range(num_vehicles):
        for i in range(n + 1):
            for j in range(1, n + 1):
                if i == j:
                    continue
                d_j0 = gp.quicksum(customers[c]['demand'] * z[c, j - 1] for c in range(m))
                model.addConstr(u[k, j] >= u[k, i] + d_j0 - vehicle_capacity * (1 - x[k, i, j]), name=f"mtz_arc_{k}_{i}_{j}")
    
    for j in range(n):
        total_visit_j = gp.quicksum(
            gp.quicksum(x[k, i, j + 1] for i in range(n + 1) if i != j + 1)
            for k in range(num_vehicles)
        )
        total_assigned = gp.quicksum(z[c, j] for c in range(m))
        model.addConstr(total_assigned >= total_visit_j, name=f"assign_visit_{j}")
    
    routing_cost = gp.quicksum(
        routing_cost_matrix[i][j] * x[k, i, j]
        for k in range(num_vehicles)
        for i in range(n + 1)
        for j in range(n + 1)
        if i != j
    )
    assignment_cost = gp.quicksum(
        assignment_cost_matrix[c][j] * z[c, j]
        for c in range(m)
        for j in range(n)
    )
    total_cost = routing_cost + assignment_cost
    model.setObjective(total_cost, GRB.MINIMIZE)
    
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj)
    
    try:
        if logger:
            model.optimize(callback)
        else:
            model.optimize()
    except Exception as e:
        print(f"Error during optimization: {e}")
        pass
    
    solution = {
        "objective_value": None,
        "routes": {},
        "assignments": {},
        "customer_tours": {}
    }
    
    if model.SolCount > 0:
        model.setParam(GRB.Param.SolutionNumber, 0)
        solution['objective_value'] = model.ObjVal
        
        routes_dict = {}
        for k in range(num_vehicles):
            tour_sequence = []
            current = 0
            tour_sequence.append(current)
            visited = set([0])
            while True:
                next_node = None
                for j in range(n + 1):
                    if j == current:
                        continue
                    if (k, current, j) in x and abs(x[k, current, j].Xn - 1.0) < 1e-5:
                        next_node = j
                        break
                if next_node is None:
                    break
                if next_node == 0 and len(tour_sequence) > 1:
                    tour_sequence.append(next_node)
                    break
                tour_sequence.append(next_node)
                if next_node in visited:
                    break
                visited.add(next_node)
                current = next_node
            if len(tour_sequence) < 2:
                continue
            node_ids = []
            for idx in tour_sequence:
                if idx == 0:
                    node_ids.append(depot['id'])
                else:
                    site_index = idx - 1
                    node_ids.append(delivery_sites[site_index]['id'])
            arcs = []
            for i in range(len(tour_sequence) - 1):
                from_idx = tour_sequence[i]
                to_idx = tour_sequence[i + 1]
                from_id = depot['id'] if from_idx == 0 else delivery_sites[from_idx - 1]['id']
                to_id = depot['id'] if to_idx == 0 else delivery_sites[to_idx - 1]['id']
                arcs.append([from_id, to_id])
            routes_dict[str(k)] = {
                "arcs": arcs,
                "sequence": node_ids
            }
        solution['routes'] = routes_dict
        
        assignments_dict = {}
        customer_tours_dict = {}
        for c in range(m):
            c_id = customers[c]['id']
            assigned_site_index = None
            for j in range(n):
                if abs(z[c, j].Xn - 1.0) < 1e-5:
                    assigned_site_index = j
                    break
            if assigned_site_index is None:
                continue
            site_id = delivery_sites[assigned_site_index]['id']
            assignments_dict[str(c_id)] = site_id
            
            vehicle_k = None
            for k in range(num_vehicles):
                found = False
                for i in range(n + 1):
                    if i == assigned_site_index + 1:
                        continue
                    if (k, i, assigned_site_index + 1) in x and abs(x[k, i, assigned_site_index + 1].Xn - 1.0) < 1e-5:
                        found = True
                        break
                if found:
                    vehicle_k = k
                    break
            if vehicle_k is None:
                continue
            customer_tours_dict[str(c_id)] = {
                "vehicle": str(vehicle_k),
                "site": site_id
            }
        solution['assignments'] = assignments_dict
        solution['customer_tours'] = customer_tours_dict
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()