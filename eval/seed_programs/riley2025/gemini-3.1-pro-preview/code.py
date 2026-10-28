import argparse
import json
import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True, help="Path to the instance JSON file.")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to write the solution JSON file.")
    parser.add_argument("--time_limit", type=int, default=3600, help="Maximum runtime in seconds.")
    parser.add_argument("--log_path", type=str, default="", help="Path to log intermediate solutions.")
    args = parser.parse_args()

    logger = None
    if args.log_path and SolutionLogger is not None:
        logger = SolutionLogger(args.log_path, sense="minimize")

    # Load instance data
    with open(args.instance_path, 'r') as f:
        data = json.load(f)
        
    I = data["num_demand_sites"]
    J = data["num_production_sites"]
    S = data["num_scenarios"]
    T = data["num_time_periods"]
    M = data["num_module_sizes"]
    K = data["units_per_module_size"]
    u_max = data["u_max"]
    prob = data["scenario_probabilities"]

    capacity = {m["id"]: m["capacity"] for m in data["module_sizes"]}
    default_costs = {m["id"]: m["capital_cost_per_unit"] for m in data["module_sizes"]}

    # Read specific unit costs if present, fallback to default size costs
    unit_costs = {}
    for m in range(M):
        for k in range(K):
            unit_costs[m,k] = default_costs[m]

    if "module_units" in data:
        for mu in data["module_units"]:
            m_id = mu["module_size_id"]
            k_id = mu["unit_id"]
            if (m_id, k_id) in unit_costs:
                unit_costs[m_id, k_id] = mu["capital_cost"]

    prod_costs = data["production_costs"]
    reloc_costs = data["relocation_costs"]
    pen_costs = data["penalty_costs"]

    dem = [None] * S
    for scen in data["scenarios"]:
        dem[scen["id"]] = scen["demands"]
        
    # Initialize Gurobi model
    model = gp.Model()
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    if args.time_limit:
        model.setParam('TimeLimit', args.time_limit)

    # 1. Variables
    # First stage: module purchases
    y = {}
    for m in range(M):
        for k in range(K):
            y[m,k] = model.addVar(vtype=GRB.BINARY, name=f"y[{m},{k}]")

    # Second stage: continuous operational variables
    x = {}
    for i in range(I):
        for j in range(J):
            for t in range(T):
                for s in range(S):
                    x[i,j,t,s] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f"x[{i},{j},{t},{s}]")

    q = {}
    for i in range(I):
        for t in range(T):
            for s in range(S):
                q[i,t,s] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f"q[{i},{t},{s}]")

    # Second stage: integer relocation & inventory variables
    v = {}
    for j in range(J):
        for m in range(M):
            for t in range(T):
                for s in range(S):
                    v[j,m,t,s] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=K, name=f"v[{j},{m},{t},{s}]")

    v_depot = {}
    for m in range(M):
        for t in range(T):
            for s in range(S):
                v_depot[m,t,s] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=K, name=f"v_depot[{m},{t},{s}]")

    w = {}
    for j in range(J):
        for jp in range(J):
            if j != jp:
                for m in range(M):
                    for t in range(T):
                        for s in range(S):
                            w[j,jp,m,t,s] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=K, name=f"w[{j},{jp},{m},{t},{s}]")

    w_fd = {}
    w_td = {}
    for j in range(J):
        for m in range(M):
            for t in range(T):
                for s in range(S):
                    w_fd[j,m,t,s] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=K, name=f"w_fd[{j},{m},{t},{s}]")
                    w_td[j,m,t,s] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=K, name=f"w_td[{j},{m},{t},{s}]")

    # 2. Objective Function
    cap_cost_expr = gp.quicksum(unit_costs[m,k] * y[m,k] for m in range(M) for k in range(K))
    op_cost_expr = gp.quicksum(
        prob[s] * (
            gp.quicksum(prod_costs[i][j] * dem[s][i][t] * x[i,j,t,s] for i in range(I) for j in range(J) for t in range(T)) +
            gp.quicksum(reloc_costs[j][jp][m] * w[j,jp,m,t,s] for j in range(J) for jp in range(J) if j != jp for m in range(M) for t in range(T)) +
            gp.quicksum(pen_costs[i][t] * dem[s][i][t] * q[i,t,s] for i in range(I) for t in range(T))
        )
        for s in range(S)
    )
    model.setObjective(cap_cost_expr + op_cost_expr, GRB.MINIMIZE)

    # 3. Constraints
    # Demand satisfaction
    for i in range(I):
        for t in range(T):
            for s in range(S):
                model.addConstr(gp.quicksum(x[i,j,t,s] for j in range(J)) + q[i,t,s] == 1, name=f"dem_sum_{i}_{t}_{s}")

    # Depot inventory balance
    for m in range(M):
        for s in range(S):
            for t in range(T):
                if t == 0:
                    model.addConstr(
                        v_depot[m,t,s] == gp.quicksum(y[m,k] for k in range(K)) +
                                          gp.quicksum(w_td[j,m,t,s] for j in range(J)) -
                                          gp.quicksum(w_fd[j,m,t,s] for j in range(J)),
                        name=f"depot_bal_{m}_{t}_{s}"
                    )
                else:
                    model.addConstr(
                        v_depot[m,t,s] == v_depot[m,t-1,s] +
                                          gp.quicksum(w_td[j,m,t,s] for j in range(J)) -
                                          gp.quicksum(w_fd[j,m,t,s] for j in range(J)),
                        name=f"depot_bal_{m}_{t}_{s}"
                    )

    # Production site inventory balance
    for j in range(J):
        for m in range(M):
            for s in range(S):
                for t in range(T):
                    inflow_w = gp.quicksum(w[jp,j,m,t,s] for jp in range(J) if jp != j)
                    outflow_w = gp.quicksum(w[j,jp,m,t,s] for jp in range(J) if jp != j)
                    if t == 0:
                        model.addConstr(
                            v[j,m,t,s] == w_fd[j,m,t,s] - w_td[j,m,t,s] + inflow_w - outflow_w,
                            name=f"prod_bal_{j}_{m}_{t}_{s}"
                        )
                    else:
                        model.addConstr(
                            v[j,m,t,s] == v[j,m,t-1,s] + w_fd[j,m,t,s] - w_td[j,m,t,s] + inflow_w - outflow_w,
                            name=f"prod_bal_{j}_{m}_{t}_{s}"
                        )

    # Production capacities and global limit
    for j in range(J):
        for t in range(T):
            for s in range(S):
                dem_weighted = gp.quicksum(dem[s][i][t] * x[i,j,t,s] for i in range(I))
                cap_avail = gp.quicksum(capacity[m] * v[j,m,t,s] for m in range(M))
                
                model.addConstr(dem_weighted <= cap_avail, name=f"cap_{j}_{t}_{s}")
                model.addConstr(dem_weighted <= u_max, name=f"umax_{j}_{t}_{s}")

    # Redundant but tightening constraint on total active unit limit
    for m in range(M):
        for s in range(S):
            for t in range(T):
                model.addConstr(
                    v_depot[m,t,s] + gp.quicksum(v[j,m,t,s] for j in range(J)) == gp.quicksum(y[m,k] for k in range(K)),
                    name=f"tot_units_{m}_{t}_{s}"
                )

    # Symmetry breaking: prefer purchasing cheaper units, and fix order for identically priced units
    for m in range(M):
        units = sorted([(unit_costs[m,k], k) for k in range(K)])
        for idx in range(K - 1):
            _, k1 = units[idx]
            _, k2 = units[idx+1]
            model.addConstr(y[m,k1] >= y[m,k2], name=f"sym_{m}_{k1}_{k2}")

    # Helper to construct solution dictionary according to schemas
    def extract_solution(model, get_val, obj):
        sol = {"model_variables": {}, "y": {}}
        
        # Binary variables mapping
        for (m,k), var in y.items():
            val = get_val(var)
            sol["y"][f"y_{m}_{k}"] = int(round(val))
            if val > 0.5:
                sol["model_variables"][var.VarName] = 1.0
                
        # Continuous variables 
        for var_dict in [x, q]:
            for key, var in var_dict.items():
                val = get_val(var)
                if val > 1e-5:
                    sol["model_variables"][var.VarName] = val
                    
        # Integer variables (cast strictly to standard float output per json structure mappings)
        for var_dict in [v, v_depot, w, w_fd, w_td]:
            for key, var in var_dict.items():
                val = get_val(var)
                if val > 0.5:
                    sol["model_variables"][var.VarName] = float(round(val))
                    
        sol["objective_value"] = obj
        return sol

    # Callbacks for recording best feasible intermediates
    def cb(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                try:
                    sol = extract_solution(model, lambda v: model.cbGetSolution(v), obj)
                    logger.log_solution(obj, sol)
                except Exception:
                    logger.log(obj)

    # Optimize Model
    model.optimize(cb)

    # Output optimal / best feasible solution
    if model.SolCount > 0:
        best_sol = extract_solution(model, lambda v: v.X, model.ObjVal)
        with open(args.solution_path, 'w') as f:
            json.dump(best_sol, f, indent=2)

if __name__ == "__main__":
    main()