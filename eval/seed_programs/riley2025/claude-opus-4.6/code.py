import argparse
import json
import time
import sys
from itertools import product

def solve(instance_path, solution_path, time_limit, log_path):
    from solution_logger import SolutionLogger
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None

    with open(instance_path, 'r') as f:
        data = json.load(f)

    I = data['num_demand_sites']
    J = data['num_production_sites']
    S = data['num_scenarios']
    T = data['num_time_periods']
    M = data['num_module_sizes']
    K = data['units_per_module_size']
    u_max = float(data['u_max'])
    
    scenario_probs = data['scenario_probabilities']
    
    capacities = [float(data['module_sizes'][m]['capacity']) for m in range(M)]
    capital_costs_per_unit = [float(data['module_sizes'][m]['capital_cost_per_unit']) for m in range(M)]
    
    # production_costs[i][j]
    prod_costs = data['production_costs']
    # relocation_costs[j][jp][m]
    reloc_costs = data['relocation_costs']
    # penalty_costs[i][t]
    pen_costs = data['penalty_costs']
    
    # scenarios demands[s][i][t]
    demands = []
    for sc in data['scenarios']:
        demands.append(sc['demands'])
    
    import gurobipy as gp
    from gurobipy import GRB
    
    start_time = time.time()
    
    model = gp.Model("modular_supply_chain")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = max(1, time_limit - 5)  # reserve some time for output
    
    # First stage: purchase decisions
    # y[m,k] binary
    y = {}
    for m in range(M):
        for k in range(K):
            y[m, k] = model.addVar(vtype=GRB.BINARY, name=f"y[{m},{k}]",
                                    obj=capital_costs_per_unit[m])
    
    # z[m,s] = number of units of size m purchased (linked to y), integer
    z = {}
    for m in range(M):
        for s in range(S):
            z[m, s] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=K, name=f"z[{m},{s}]")
    
    # Second stage variables per scenario
    # x[i,j,t,s] fraction of demand at site i served from production site j
    x = {}
    for s in range(S):
        for t in range(T):
            for i in range(I):
                for j in range(J):
                    x[i, j, t, s] = model.addVar(lb=0.0, ub=1.0, name=f"x[{i},{j},{t},{s}]")
    
    # q[i,t,s] unmet fraction
    q = {}
    for s in range(S):
        for t in range(T):
            for i in range(I):
                q[i, t, s] = model.addVar(lb=0.0, ub=1.0, name=f"q[{i},{t},{s}]")
    
    # v[j,m,t,s] units at production site j
    v = {}
    for s in range(S):
        for t in range(T):
            for j in range(J):
                for m in range(M):
                    v[j, m, t, s] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=K,
                                                   name=f"v[{j},{m},{t},{s}]")
    
    # v_depot[m,t,s] units at depot
    v_depot = {}
    for s in range(S):
        for t in range(T):
            for m in range(M):
                v_depot[m, t, s] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=K,
                                                  name=f"v_depot[{m},{t},{s}]")
    
    # w[j,jp,m,t,s] relocation between production sites (j != jp)
    w = {}
    for s in range(S):
        for t in range(T):
            for j in range(J):
                for jp in range(J):
                    if j != jp:
                        for m in range(M):
                            w[j, jp, m, t, s] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=K,
                                                               name=f"w[{j},{jp},{m},{t},{s}]")
    
    # w_fd[j,m,t,s] depot-to-production
    w_fd = {}
    for s in range(S):
        for t in range(T):
            for j in range(J):
                for m in range(M):
                    w_fd[j, m, t, s] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=K,
                                                      name=f"w_fd[{j},{m},{t},{s}]")
    
    # w_td[j,m,t,s] production-to-depot
    w_td = {}
    for s in range(S):
        for t in range(T):
            for j in range(J):
                for m in range(M):
                    w_td[j, m, t, s] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=K,
                                                      name=f"w_td[{j},{m},{t},{s}]")
    
    model.update()
    
    # Objective: capital costs (already in obj of y vars) + expected operating costs
    # Operating cost per scenario s:
    # 1) demand-weighted production cost
    # 2) relocation cost
    # 3) unmet demand penalty
    
    op_cost = gp.LinExpr()
    for s in range(S):
        prob = scenario_probs[s]
        # Production/serving cost
        for t in range(T):
            for i in range(I):
                d_ist = float(demands[s][i][t])
                if d_ist > 0:
                    for j in range(J):
                        coeff = prob * float(prod_costs[i][j]) * d_ist
                        if coeff != 0:
                            op_cost.addTerms(coeff, x[i, j, t, s])
                    # Penalty cost
                    pen_coeff = prob * float(pen_costs[i][t]) * d_ist
                    if pen_coeff != 0:
                        op_cost.addTerms(pen_coeff, q[i, t, s])
        
        # Relocation costs between production sites
        for t in range(T):
            for j in range(J):
                for jp in range(J):
                    if j != jp:
                        for m in range(M):
                            rc = prob * float(reloc_costs[j][jp][m])
                            if rc != 0:
                                op_cost.addTerms(rc, w[j, jp, m, t, s])
        
        # Relocation costs depot-to-production and production-to-depot
        # The problem says relocation costs are for ordered pairs of production sites.
        # Depot relocations: we need to check if there's a cost. The problem description
        # mentions "all ordered pairs of sites (including moves between the depot and production sites)"
        # but the relocation_costs are only [J][J][M]. 
        # Looking more carefully: "For every ordered pair of production sites, every module size, 
        # and every time period, a per-unit relocation cost is given"
        # But the relocation cost description says relocations happen between production sites only
        # for the cost matrix. Depot moves might be free or we need to handle differently.
        # Actually re-reading: the cost matrix is [num_production_sites][num_production_sites][num_module_sizes]
        # and only covers production-to-production moves. Depot moves seem to have no explicit cost
        # in the data, suggesting they might be zero cost. Let's assume depot moves are free.
    
    model.setObjective(model.getObjective() + op_cost, GRB.MINIMIZE)
    
    # Constraints
    
    # Link z to y: z[m,s] = sum_k y[m,k] for all s
    for m in range(M):
        total_y = gp.quicksum(y[m, k] for k in range(K))
        for s in range(S):
            model.addConstr(z[m, s] == total_y, name=f"link_z[{m},{s}]")
    
    # Symmetry breaking: y[m,k] >= y[m,k+1]
    for m in range(M):
        for k in range(K - 1):
            model.addConstr(y[m, k] >= y[m, k + 1], name=f"sym[{m},{k}]")
    
    # Demand satisfaction: sum_j x[i,j,t,s] + q[i,t,s] = 1
    for s in range(S):
        for t in range(T):
            for i in range(I):
                model.addConstr(
                    gp.quicksum(x[i, j, t, s] for j in range(J)) + q[i, t, s] == 1.0,
                    name=f"dem_sat[{i},{t},{s}]"
                )
    
    # Depot inventory: v_depot[m,t,s] = z[m,s] + sum over tau=0..t of (sum_j w_td[j,m,tau,s] - sum_j w_fd[j,m,tau,s])
    for s in range(S):
        for m in range(M):
            for t in range(T):
                net_flow = gp.LinExpr()
                for tau in range(t + 1):
                    for j in range(J):
                        net_flow += w_td[j, m, tau, s]
                        net_flow -= w_fd[j, m, tau, s]
                model.addConstr(
                    v_depot[m, t, s] == z[m, s] + net_flow,
                    name=f"depot_inv[{m},{t},{s}]"
                )
    
    # Production site inventory: v[j,m,t,s] = sum over tau=0..t of net inflow
    # net inflow at tau = (w_fd[j,m,tau,s] - w_td[j,m,tau,s]) + sum_{jp!=j} (w[jp,j,m,tau,s] - w[j,jp,m,tau,s])
    for s in range(S):
        for j in range(J):
            for m in range(M):
                for t in range(T):
                    net_flow = gp.LinExpr()
                    for tau in range(t + 1):
                        net_flow += w_fd[j, m, tau, s]
                        net_flow -= w_td[j, m, tau, s]
                        for jp in range(J):
                            if jp != j:
                                net_flow += w[jp, j, m, tau, s]
                                net_flow -= w[j, jp, m, tau, s]
                    model.addConstr(
                        v[j, m, t, s] == net_flow,
                        name=f"site_inv[{j},{m},{t},{s}]"
                    )
    
    # Capacity constraint at production site
    for s in range(S):
        for t in range(T):
            for j in range(J):
                total_prod = gp.LinExpr()
                for i in range(I):
                    d_ist = float(demands[s][i][t])
                    if d_ist > 0:
                        total_prod += d_ist * x[i, j, t, s]
                
                # Capacity from modules
                cap = gp.quicksum(capacities[m] * v[j, m, t, s] for m in range(M))
                model.addConstr(total_prod <= cap, name=f"cap[{j},{t},{s}]")
                
                # Global max production limit
                model.addConstr(total_prod <= u_max, name=f"umax[{j},{t},{s}]")
    
    model.update()
    
    # Callback to log solutions
    best_obj = [float('inf')]
    
    def callback(model_cb, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model_cb.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj_val < best_obj[0] - 1e-6:
                best_obj[0] = obj_val
                if logger:
                    logger.log(obj_val)
    
    model.optimize(callback)
    
    # Extract solution
    if model.SolCount == 0:
        # No solution found - output empty
        sol = {
            "objective_value": float('inf'),
            "model_variables": {},
            "y": {}
        }
        with open(solution_path, 'w') as f:
            json.dump(sol, f)
        return
    
    obj_val = model.ObjVal
    
    model_vars = {}
    y_dict = {}
    
    # Extract y
    for m in range(M):
        for k in range(K):
            val = y[m, k].X
            if abs(val) > 1e-8:
                key = f"y[{m},{k}]"
                model_vars[key] = round(val)
                y_dict[f"y_{m}_{k}"] = round(val)
    
    # Extract z
    for m in range(M):
        for s in range(S):
            val = z[m, s].X
            if abs(val) > 1e-8:
                model_vars[f"z[{m},{s}]"] = round(val)
    
    # Extract x
    for (i, j, t, s), var in x.items():
        val = var.X
        if abs(val) > 1e-8:
            model_vars[f"x[{i},{j},{t},{s}]"] = val
    
    # Extract q
    for (i, t, s), var in q.items():
        val = var.X
        if abs(val) > 1e-8:
            model_vars[f"q[{i},{t},{s}]"] = val
    
    # Extract v
    for (j, m, t, s), var in v.items():
        val = var.X
        if abs(val) > 1e-8:
            model_vars[f"v[{j},{m},{t},{s}]"] = round(val)
    
    # Extract v_depot
    for (m, t, s), var in v_depot.items():
        val = var.X
        if abs(val) > 1e-8:
            model_vars[f"v_depot[{m},{t},{s}]"] = round(val)
    
    # Extract w
    for (j, jp, m, t, s), var in w.items():
        val = var.X
        if abs(val) > 1e-8:
            model_vars[f"w[{j},{jp},{m},{t},{s}]"] = round(val)
    
    # Extract w_fd
    for (j, m, t, s), var in w_fd.items():
        val = var.X
        if abs(val) > 1e-8:
            model_vars[f"w_fd[{j},{m},{t},{s}]"] = round(val)
    
    # Extract w_td
    for (j, m, t, s), var in w_td.items():
        val = var.X
        if abs(val) > 1e-8:
            model_vars[f"w_td[{j},{m},{t},{s}]"] = round(val)
    
    sol = {
        "objective_value": obj_val,
        "model_variables": model_vars,
        "y": y_dict
    }
    
    if logger:
        logger.log_solution(obj_val, sol)
    
    with open(solution_path, 'w') as f:
        json.dump(sol, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()
    
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)