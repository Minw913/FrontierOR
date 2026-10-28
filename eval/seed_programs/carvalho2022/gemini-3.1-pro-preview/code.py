import sys
import json
import argparse
import math
import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None

def main():
    parser = argparse.ArgumentParser(description="Solve Manufacturing Scheduling Problem")
    parser.add_argument('--instance_path', type=str, required=True, help='Path to instance JSON')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to output solution JSON')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds')
    parser.add_argument('--log_path', type=str, required=False, help='Path to logging JSONL')
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    # Logging utility
    logger = None
    if getattr(args, 'log_path', None):
        if SolutionLogger is not None:
            logger = SolutionLogger(args.log_path, sense="minimize")

    def make_callback(sol_logger):
        def callback(model, where):
            if where == GRB.Callback.MIPSOL:
                try:
                    obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    if sol_logger:
                        sol_logger.log(obj)
                except Exception:
                    pass
        return callback

    # Dimensions
    n = data['dimensions']['n']
    m = data['dimensions']['m']
    p = data['dimensions']['p']

    # Parameters
    demands = data['demands']
    inventory_costs = data['inventory_costs']
    processing_time = data['processing_time']
    setup_times = data['setup_times']
    setup_costs = data['setup_costs']
    max_production = data['max_production']
    max_setups_per_item = data['max_setups_per_item']
    minimum_lot_sizes = data['minimum_lot_sizes']
    machine_eligibility = data['machine_eligibility']
    machine_capacities = data['machine_capacities']

    env = gp.Env()
    env.setParam("Threads", 1)  # Enforce single core based on requirements
    model = gp.Model(env=env)

    # Time limit and optimization focus
    model.setParam("TimeLimit", args.time_limit)
    model.setParam("MIPFocus", 1)       # Focus on finding feasible incumbent solutions
    model.setParam("Heuristics", 0.2)   # Shift ratio to heuristic methods

    # Decision variables
    x = {} # production
    for i in range(n):
        for k in range(m):
            for t in range(p):
                for d in range(t, p):
                    x[i, k, t, d] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"x_{i}_{k}_{t}_{d}")

    y = {} # setups
    for k in range(m):
        for t in range(p):
            for i in range(n):
                for j in range(n):
                    if i != j:
                        y[i, j, k, t] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"y_{i}_{j}_{k}_{t}")

    z = {} # carryover
    for i in range(n):
        for k in range(m):
            for t in range(p + 1):
                z[i, k, t] = model.addVar(vtype=GRB.BINARY, name=f"z_{i}_{k}_{t}")

    # Sub-cycle elimination via single-commodity flow per machine and period
    f_flow = {}
    g_flow = {}
    for k in range(m):
        for t in range(p):
            for i in range(n):
                g_flow[i, k, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"g_{i}_{k}_{t}")
                for j in range(n):
                    if i != j:
                        f_flow[i, j, k, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"f_{i}_{j}_{k}_{t}")

    # Objective function
    obj = gp.LinExpr()
    
    # Inventory Holding Costs
    for i in range(n):
        for k in range(m):
            for t in range(p):
                for d in range(t, p):
                    holding_periods = d - t
                    if holding_periods > 0 and inventory_costs[i] > 0:
                        obj.add(x[i, k, t, d], inventory_costs[i] * holding_periods)

    # Sequence-dependent Setup Costs
    for k in range(m):
        for t in range(p):
            for i in range(n):
                for j in range(n):
                    if i != j:
                        if setup_costs[i][j][k] != 0:
                            obj.add(y[i, j, k, t], setup_costs[i][j][k])

    model.setObjective(obj, GRB.MINIMIZE)

    # CONSTRAINTS

    # 1. Demand completion
    for i in range(n):
        for d in range(p):
            model.addConstr(
                gp.quicksum(x[i, k, t, d] for k in range(m) for t in range(d + 1)) == demands[i][d],
                name=f"demand_i{i}_d{d}"
            )

    # 2. Maximum production from period t onward
    for i in range(n):
        for t in range(p):
            model.addConstr(
                gp.quicksum(x[i, k, tau, d] for k in range(m) for tau in range(t, p) for d in range(tau, p)) <= max_production[i][t],
                name=f"max_prod_i{i}_t{t}"
            )

    # 3. Capacity constraints
    for k in range(m):
        for t in range(p):
            lhs = gp.LinExpr()
            for i in range(n):
                for d in range(t, p):
                    if processing_time > 0:
                        lhs.add(x[i, k, t, d], processing_time)
            for i in range(n):
                for j in range(n):
                    if i != j:
                        if setup_times[i][j][k] > 0:
                            lhs.add(y[i, j, k, t], setup_times[i][j][k])
            model.addConstr(lhs <= machine_capacities[k][t], name=f"cap_k{k}_t{t}")

    # 4. State configuration and transitions (Degree Balance)
    for k in range(m):
        for t in range(p):
            # Only exact one initial configuring state
            model.addConstr(gp.quicksum(z[i, k, t] for i in range(n)) == 1, name=f"z_sum_k{k}_t{t}")

            for i in range(n):
                enters = z[i, k, t] + gp.quicksum(y[j, i, k, t] for j in range(n) if j != i)
                leaves = z[i, k, t + 1] + gp.quicksum(y[i, j, k, t] for j in range(n) if j != i)
                model.addConstr(enters == leaves, name=f"bal_i{i}_k{k}_t{t}")

    # 5. Production setup dependency and minimum lot size requirements
    for i in range(n):
        for k in range(m):
            for t in range(p):
                sum_x = gp.quicksum(x[i, k, t, d] for d in range(t, p))
                sum_y = gp.quicksum(y[j, i, k, t] for j in range(n) if j != i)

                # Link continuous production to valid state setups (carry-overs + changeovers)
                bound_cap = float('inf')
                if processing_time > 0:
                    bound_cap = math.floor(machine_capacities[k][t] / processing_time)
                big_m = min(sum(demands[i][d] for d in range(t, p)), max_production[i][t])
                if processing_time > 0:
                    big_m = min(big_m, bound_cap)
                    
                model.addConstr(sum_x <= int(big_m) * (z[i, k, t] + sum_y), name=f"setup_valid_i{i}_k{k}_t{t}")

                # Guarantee minimum lot sizes for newly configured changeovers (excludes strict carryovers)
                model.addConstr(sum_x >= minimum_lot_sizes[i] * sum_y, name=f"min_lot_i{i}_k{k}_t{t}")

    # 6. Maximum setups restriction (and machine ineligibility bindings)
    for i in range(n):
        for k in range(m):
            for t in range(p):
                model.addConstr(
                    gp.quicksum(y[j, i, k, t] for j in range(n) if j != i) <= max_setups_per_item[i][k][t],
                    name=f"max_setups_i{i}_k{k}_t{t}"
                )
                
                # Prevent any production variable from existing securely if physically blocked
                if max_setups_per_item[i][k][t] == 0:
                    for d in range(t, p):
                        x[i, k, t, d].ub = 0

            if machine_eligibility[i][k] == 0:
                for t in range(p + 1):
                    z[i, k, t].ub = 0
                for t in range(p):
                    for j in range(n):
                        if i != j:
                            y[i, j, k, t].ub = 0
                            y[j, i, k, t].ub = 0
                    for d in range(t, p):
                        x[i, k, t, d].ub = 0

    # 7. Disconnected Sub-cycle elimination (Flow logic)
    for k in range(m):
        for t in range(p):
            m_flow = sum(max_setups_per_item[i][k][t] for i in range(n))
            if m_flow == 0:
                continue

            sum_total_y = gp.quicksum(y[i, j, k, t] for i in range(n) for j in range(n) if i != j)
            sum_total_g = gp.quicksum(g_flow[i, k, t] for i in range(n))
            
            model.addConstr(sum_total_g == sum_total_y, name=f"flow_g_sum_k{k}_t{t}")

            for i in range(n):
                model.addConstr(g_flow[i, k, t] <= m_flow * z[i, k, t], name=f"flow_g_ub_i{i}_k{k}_t{t}")
                
                flow_in = gp.quicksum(f_flow[j, i, k, t] for j in range(n) if j != i) + g_flow[i, k, t]
                flow_out = gp.quicksum(f_flow[i, j, k, t] for j in range(n) if j != i)
                flow_consumed = gp.quicksum(y[j, i, k, t] for j in range(n) if j != i)
                
                model.addConstr(flow_in == flow_out + flow_consumed, name=f"flow_bal_i{i}_k{k}_t{t}")

                for j in range(n):
                    if i != j:
                        model.addConstr(f_flow[i, j, k, t] <= m_flow * y[i, j, k, t], name=f"flow_f_ub_i{i}j{j}k{k}t{t}")

    # Start Optimization
    model.optimize(make_callback(logger))

    # Retrieve output and Format
    if model.SolCount > 0:
        production = {}
        for i in range(n):
            for k in range(m):
                for t in range(p):
                    for d in range(p):
                        if d >= t:
                            production[f"x_{i}_{k}_{t}_{d}"] = float(x[i, k, t, d].X)
                        else:
                            production[f"x_{i}_{k}_{t}_{d}"] = 0.0

        setups = {}
        for from_i in range(n):
            for to_i in range(n):
                for k in range(m):
                    for t in range(p):
                        if from_i == to_i:
                            setups[f"y_{from_i}_{to_i}_{k}_{t}"] = 0
                        else:
                            setups[f"y_{from_i}_{to_i}_{k}_{t}"] = int(round(y[from_i, to_i, k, t].X))

        carryover = {}
        for i in range(n):
            for k in range(m):
                for t in range(p):
                    carryover[f"z_{i}_{k}_{t}"] = int(round(z[i, k, t].X))

        solution = {
            "objective_value": float(model.ObjVal),
            "production": production,
            "setups": setups,
            "carryover": carryover
        }

        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
    else:
        print("MIP Solver could not find a feasible solution within the given time limit.", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()