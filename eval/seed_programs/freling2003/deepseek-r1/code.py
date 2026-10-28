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
        data = json.load(f)
    
    num_facilities = data['parameters']['num_facilities']
    num_customers = data['parameters']['num_customers']
    num_periods = data['parameters']['num_periods']
    seasonal_factors = data['seasonal_factors']
    demands = data['demands']
    holding_costs = data['holding_costs']
    capacities = data['capacities']
    transportation_costs = data['transportation_costs']
    
    demand_per_customer_period = []
    for i in range(num_customers):
        d_i = demands[i]
        demand_i = [d_i * s for s in seasonal_factors]
        demand_per_customer_period.append(demand_i)
    
    total_transport_cost = [[0.0] * num_customers for _ in range(num_facilities)]
    for j in range(num_facilities):
        for i in range(num_customers):
            total = 0.0
            for t in range(num_periods):
                total += transportation_costs[j][i][t] * demand_per_customer_period[i][t]
            total_transport_cost[j][i] = total

    m = gp.Model("Logistics_Distribution")
    m.setParam("Threads", 1)
    m.setParam('OutputFlag', 0)
    m.setParam('TimeLimit', args.time_limit)
    
    x = m.addVars(num_facilities, num_customers, vtype=GRB.BINARY, name="x")
    I = m.addVars(num_facilities, num_periods, lb=0.0, name="I")
    
    for i in range(num_customers):
        m.addConstr(gp.quicksum(x[j, i] for j in range(num_facilities)) == 1, name=f"assign_customer_{i}")
    
    for j in range(num_facilities):
        demand_j0 = gp.quicksum(x[j, i] * demand_per_customer_period[i][0] for i in range(num_customers))
        m.addConstr(demand_j0 + I[j, 0] <= capacities[j][0], name=f"flow_{j}_0")
        for t in range(1, num_periods):
            demand_jt = gp.quicksum(x[j, i] * demand_per_customer_period[i][t] for i in range(num_customers))
            m.addConstr(demand_jt + I[j, t] - I[j, t-1] <= capacities[j][t], name=f"flow_{j}_{t}")
    
    transport_cost = gp.quicksum(total_transport_cost[j][i] * x[j, i] for j in range(num_facilities) for i in range(num_customers))
    inventory_cost = gp.quicksum(holding_costs[j][t] * I[j, t] for j in range(num_facilities) for t in range(num_periods))
    m.setObjective(transport_cost + inventory_cost, GRB.MINIMIZE)
    
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    def make_callback(logger_obj):
        def callback(model, where):
            if where == GRB.Callback.MIPSOL:
                obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if logger_obj:
                    logger_obj.log(obj)
        return callback

    if logger:
        callback_fn = make_callback(logger)
        m.optimize(callback_fn)
    else:
        m.optimize()
    
    if m.SolCount > 0:
        obj_value = m.ObjVal
        assignment_dict = {}
        for i in range(num_customers):
            for j in range(num_facilities):
                if x[j, i].X > 0.5:
                    assignment_dict[str(i)] = j
                    break
        
        inventory_dict = {}
        for j in range(num_facilities):
            inv_list = [I[j, t].X for t in range(num_periods)]
            inventory_dict[str(j)] = inv_list
        
        solution = {
            "objective_value": obj_value,
            "assignment": assignment_dict,
            "inventory": inventory_dict
        }
    else:
        assignment_dict = {}
        for i in range(num_customers):
            min_cost = float('inf')
            best_j = 0
            for j in range(num_facilities):
                if total_transport_cost[j][i] < min_cost:
                    min_cost = total_transport_cost[j][i]
                    best_j = j
            assignment_dict[str(i)] = best_j
        
        D_jt = [[0.0] * num_periods for _ in range(num_facilities)]
        for i in range(num_customers):
            j = assignment_dict[str(i)]
            for t in range(num_periods):
                D_jt[j][t] += demand_per_customer_period[i][t]
        
        inventory_dict = {}
        total_inv_cost = 0.0
        for j in range(num_facilities):
            cum_demand = 0.0
            cum_cap = 0.0
            inv_list = []
            for t in range(num_periods):
                cum_demand += D_jt[j][t]
                cum_cap += capacities[j][t]
                inv_level = cum_cap - cum_demand
                if inv_level < 0:
                    inv_level = 0.0
                total_inv_cost += holding_costs[j][t] * inv_level
                inv_list.append(inv_level)
            inventory_dict[str(j)] = inv_list
        
        total_trans_cost = 0.0
        for i in range(num_customers):
            j = assignment_dict[str(i)]
            total_trans_cost += total_transport_cost[j][i]
        
        obj_value = total_trans_cost + total_inv_cost
        solution = {
            "objective_value": obj_value,
            "assignment": assignment_dict,
            "inventory": inventory_dict
        }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == '__main__':
    main()