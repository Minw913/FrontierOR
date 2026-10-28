import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    # Extract data
    I = data['num_demand_sites']
    J = data['num_production_sites']
    S = data['num_scenarios']
    T = data['num_time_periods']
    M = data['num_module_sizes']
    K = data['units_per_module_size']
    u_max = data['u_max']
    
    scenario_probs = data['scenario_probabilities']
    module_sizes = data['module_sizes']
    scenarios = data['scenarios']
    
    cap = [module_sizes[m]['capacity'] for m in range(M)]
    capital_cost = [module_sizes[m]['capital_cost_per_unit'] for m in range(M)]
    
    production_costs = data['production_costs']
    relocation_costs = data['relocation_costs']
    penalty_costs = data['penalty_costs']
    
    demands = []
    for scen in scenarios:
        demands.append(scen['demands'])
    
    # Create model
    model = gp.Model("ModularSupplyChain")
    
    # Set parameters
    model.setParam(GRB.Param.Seed, 0)
    model.setParam(GRB.Param.MIPGap, 1e-4)
    model.setParam(GRB.Param.NumericFocus, 0)
    model.setParam(GRB.Param.Threads, 1)
    model.setParam(GRB.Param.TimeLimit, args.time_limit)
    
    # Variables
    y = model.addVars(M, K, vtype=GRB.BINARY, name="y")
    z = model.addVars(M, S, vtype=GRB.CONTINUOUS, lb=0, ub=K, name="z")
    x = model.addVars(I, J, T, S, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="x")
    q = model.addVars(I, T, S, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="q")
    v = model.addVars(J, M, T, S, vtype=GRB.INTEGER, lb=0, ub=K, name="v")
    v_depot = model.addVars(M, T, S, vtype=GRB.INTEGER, lb=0, ub=K, name="v_depot")
    w = model.addVars(J, J, M, T, S, vtype=GRB.INTEGER, lb=0, ub=K, name="w")
    w_fd = model.addVars(J, M, T, S, vtype=GRB.INTEGER, lb=0, ub=K, name="w_fd")
    w_td = model.addVars(J, M, T, S, vtype=GRB.INTEGER, lb=0, ub=K, name="w_td")
    
    # Objective
    capital_cost_expr = gp.quicksum(capital_cost[m] * y[m, k] for m in range(M) for k in range(K))
    
    operating_cost = 0
    for s in range(S):
        prob = scenario_probs[s]
        # Production + transport cost
        prod_trans = gp.quicksum(
            production_costs[i][j] * demands[s][i][t] * x[i, j, t, s]
            for i in range(I) for j in range(J) for t in range(T)
        )
        # Penalty
        penalty = gp.quicksum(
            penalty_costs[i][t] * demands[s][i][t] * q[i, t, s]
            for i in range(I) for t in range(T)
        )
        # Relocation cost
        reloc = gp.quicksum(
            relocation_costs[j][jp][m] * w[j, jp, m, t, s]
            for j in range(J) for jp in range(J) if j != jp
            for m in range(M) for t in range(T)
        )
        reloc += gp.quicksum(
            relocation_costs[j][j][m] * w_fd[j, m, t, s]  # assume index 0 is depot? Wait, schema uses production indices
            for j in range(J) for m in range(M) for t in range(T)
        )
        reloc += gp.quicksum(
            relocation_costs[j][j][m] * w_td[j, m, t, s]
            for j in range(J) for m in range(M) for t in range(T)
        )
        operating_cost += prob * (prod_trans + penalty + reloc)
    
    model.setObjective(capital_cost_expr + operating_cost, GRB.MINIMIZE)
    
    # Constraints
    
    # 1. Demand satisfaction
    for i in range(I):
        for t in range(T):
            for s in range(S):
                model.addConstr(
                    gp.quicksum(x[i, j, t, s] for j in range(J)) + q[i, t, s] == 1,
                    name=f"demand_balance_{i}_{t}_{s}"
                )
    
    # 2. Capacity constraints
    for j in range(J):
        for t in range(T):
            for s in range(S):
                # Module capacity
                model.addConstr(
                    gp.quicksum(
                        demands[s][i][t] * x[i, j, t, s] for i in range(I)
                    ) <= gp.quicksum(cap[m] * v[j, m, t, s] for m in range(M)),
                    name=f"module_capacity_{j}_{t}_{s}"
                )
                # Global production limit
                model.addConstr(
                    gp.quicksum(
                        demands[s][i][t] * x[i, j, t, s] for i in range(I)
                    ) <= u_max,
                    name=f"global_limit_{j}_{t}_{s}"
                )
    
    # 3. Inventory balance - depot and sites (cumulative flows)
    for m in range(M):
        for s in range(S):
            # Initial condition: all at depot
            total_purchased = gp.quicksum(y[m, k] for k in range(K))
            model.addConstr(v_depot[m, 0, s] == total_purchased, name=f"init_depot_{m}_{s}")
            for j in range(J):
                model.addConstr(v[j, m, 0, s] == 0, name=f"init_site_{j}_{m}_{s}")
            
            for t in range(T):
                # Depot balance (cumulative)
                inflow_depot = gp.quicksum(w_td[j, m, t, s] for j in range(J))
                outflow_depot = gp.quicksum(w_fd[j, m, t, s] for j in range(J))
                if t == 0:
                    model.addConstr(
                        v_depot[m, t, s] == total_purchased + inflow_depot - outflow_depot,
                        name=f"depot_balance_{m}_{t}_{s}"
                    )
                else:
                    model.addConstr(
                        v_depot[m, t, s] == v_depot[m, t-1, s] + inflow_depot - outflow_depot,
                        name=f"depot_balance_{m}_{t}_{s}"
                    )
                
                # Site balances
                for j in range(J):
                    inflow = w_fd[j, m, t, s] + gp.quicksum(w[jp, j, m, t, s] for jp in range(J) if jp != j)
                    outflow = w_td[j, m, t, s] + gp.quicksum(w[j, jp, m, t, s] for jp in range(J) if jp != j)
                    if t == 0:
                        model.addConstr(
                            v[j, m, t, s] == inflow - outflow,
                            name=f"site_balance_{j}_{m}_{t}_{s}"
                        )
                    else:
                        model.addConstr(
                            v[j, m, t, s] == v[j, m, t-1, s] + inflow - outflow,
                            name=f"site_balance_{j}_{m}_{t}_{s}"
                        )
    
    # 4. z variables: total purchased per module size per scenario (for consistency)
    for m in range(M):
        for s in range(S):
            model.addConstr(
                z[m, s] == gp.quicksum(y[m, k] for k in range(K)),
                name=f"z_link_{m}_{s}"
            )
            # Also link to total units
            model.addConstr(
                gp.quicksum(v_depot[m, T-1, s] for s in [s]) + gp.quicksum(v[j, m, T-1, s] for j in range(J)) == z[m, s],
                name=f"total_conservation_{m}_{s}"
            )
    
    # 5. Relocation bounds (only between different sites for w)
    for j in range(J):
        for jp in range(J):
            if j == jp:
                for m in range(M):
                    for t in range(T):
                        for s in range(S):
                            model.addConstr(w[j, jp, m, t, s] == 0, name=f"no_self_reloc_{j}_{jp}_{m}_{t}_{s}")
    
    # Callback for logging incumbents
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                # Extract solution
                sol = {}
                try:
                    # y
                    for m in range(M):
                        for k in range(K):
                            val = model.cbGetSolution(y[m, k])
                            if abs(val) > 1e-6:
                                sol[f"y[{m},{k}]"] = int(round(val))
                    
                    # z
                    for m_val in range(M):
                        for s_val in range(S):
                            val = model.cbGetSolution(z[m_val, s_val])
                            if abs(val) > 1e-6:
                                sol[f"z[{m_val},{s_val}]"] = float(val)
                    
                    # x, q, v, v_depot, w, w_fd, w_td
                    for i_val in range(I):
                        for j_val in range(J):
                            for t_val in range(T):
                                for s_val in range(S):
                                    val = model.cbGetSolution(x[i_val, j_val, t_val, s_val])
                                    if abs(val) > 1e-6:
                                        sol[f"x[{i_val},{j_val},{t_val},{s_val}]"] = float(val)
                    
                    for i_val in range(I):
                        for t_val in range(T):
                            for s_val in range(S):
                                val = model.cbGetSolution(q[i_val, t_val, s_val])
                                if abs(val) > 1e-6:
                                    sol[f"q[{i_val},{t_val},{s_val}]"] = float(val)
                    
                    for j_val in range(J):
                        for m_val in range(M):
                            for t_val in range(T):
                                for s_val in range(S):
                                    val = model.cbGetSolution(v[j_val, m_val, t_val, s_val])
                                    if abs(val) > 1e-6:
                                        sol[f"v[{j_val},{m_val},{t_val},{s_val}]"] = int(round(val))
                    
                    for m_val in range(M):
                        for t_val in range(T):
                            for s_val in range(S):
                                val = model.cbGetSolution(v_depot[m_val, t_val, s_val])
                                if abs(val) > 1e-6:
                                    sol[f"v_depot[{m_val},{t_val},{s_val}]"] = int(round(val))
                    
                    for j_val in range(J):
                        for jp_val in range(J):
                            for m_val in range(M):
                                for t_val in range(T):
                                    for s_val in range(S):
                                        val = model.cbGetSolution(w[j_val, jp_val, m_val, t_val, s_val])
                                        if abs(val) > 1e-6:
                                            sol[f"w[{j_val},{jp_val},{m_val},{t_val},{s_val}]"] = int(round(val))
                    
                    for j_val in range(J):
                        for m_val in range(M):
                            for t_val in range(T):
                                for s_val in range(S):
                                    val = model.cbGetSolution(w_fd[j_val, m_val, t_val, s_val])
                                    if abs(val) > 1e-6:
                                        sol[f"w_fd[{j_val},{m_val},{t_val},{s_val}]"] = int(round(val))
                                    val = model.cbGetSolution(w_td[j_val, m_val, t_val, s_val])
                                    if abs(val) > 1e-6:
                                        sol[f"w_td[{j_val},{m_val},{t_val},{s_val}]"] = int(round(val))
                    
                    if logger:
                        logger.log_solution(obj, sol)
                except:
                    if logger:
                        logger.log(obj)
    
    model.optimize(callback)
    
    # Build solution dictionary
    solution_dict = {}
    if model.SolCount > 0:
        # Extract values
        for m in range(M):
            for k in range(K):
                val = y[m, k].X
                if abs(val) > 1e-6:
                    solution_dict[f"y[{m},{k}]"] = int(round(val))
        
        for m_val in range(M):
            for s_val in range(S):
                val = z[m_val, s_val].X
                if abs(val) > 1e-6:
                    solution_dict[f"z[{m_val},{s_val}]"] = float(val)
        
        for i_val in range(I):
            for j_val in range(J):
                for t_val in range(T):
                    for s_val in range(S):
                        val = x[i_val, j_val, t_val, s_val].X
                        if abs(val) > 1e-6:
                            solution_dict[f"x[{i_val},{j_val},{t_val},{s_val}]"] = float(val)
        
        for i_val in range(I):
            for t_val in range(T):
                for s_val in range(S):
                    val = q[i_val, t_val, s_val].X
                    if abs(val) > 1e-6:
                        solution_dict[f"q[{i_val},{t_val},{s_val}]"] = float(val)
        
        for j_val in range(J):
            for m_val in range(M):
                for t_val in range(T):
                    for s_val in range(S):
                        val = v[j_val, m_val, t_val, s_val].X
                        if abs(val) > 1e-6:
                            solution_dict[f"v[{j_val},{m_val},{t_val},{s_val}]"] = int(round(val))
        
        for m_val in range(M):
            for t_val in range(T):
                for s_val in range(S):
                    val = v_depot[m_val, t_val, s_val].X
                    if abs(val) > 1e-6:
                        solution_dict[f"v_depot[{m_val},{t_val},{s_val}]"] = int(round(val))
        
        for j_val in range(J):
            for jp_val in range(J):
                for m_val in range(M):
                    for t_val in range(T):
                        for s_val in range(S):
                            val = w[j_val, jp_val, m_val, t_val, s_val].X
                            if abs(val) > 1e-6:
                                solution_dict[f"w[{j_val},{jp_val},{m_val},{t_val},{s_val}]"] = int(round(val))
        
        for j_val in range(J):
            for m_val in range(M):
                for t_val in range(T):
                    for s_val in range(S):
                        val = w_fd[j_val, m_val, t_val, s_val].X
                        if abs(val) > 1e-6:
                            solution_dict[f"w_fd[{j_val},{m_val},{t_val},{s_val}]"] = int(round(val))
                        val = w_td[j_val, m_val, t_val, s_val].X
                        if abs(val) > 1e-6:
                            solution_dict[f"w_td[{j_val},{m_val},{t_val},{s_val}]"] = int(round(val))
    
    objective = model.ObjVal if model.SolCount > 0 else 0.0
    
    output = {
        "objective_value": float(objective),
        "model_variables": solution_dict,
        "y": {f"y_{m}_{k}": solution_dict.get(f"y[{m},{k}]", 0) for m in range(M) for k in range(K)}
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(output, f, indent=2)

if __name__ == "__main__":
    main()