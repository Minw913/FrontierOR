import argparse
import json
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def solve():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True, help="Path to the JSON instance file.")
    parser.add_argument("--solution_path", type=str, required=True, help="Path where the output JSON will be written.")
    parser.add_argument("--time_limit", type=int, default=300, help="Maximum runtime in seconds.")
    parser.add_argument("--log_path", type=str, default="", help="Path for logging intermediate solutions.")
    args = parser.parse_args()

    # Initialize Logger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Load instance data
    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    N = data["N"]
    s_lb = data["first_stage"]["variable_lower_bounds"]
    holding = data["second_stage"]["holding_costs"]
    penalty = data["second_stage"]["penalty_costs"]
    
    trans_mean = data["second_stage"]["transshipment_costs_mean"]
    # Provide a default deterministic baseline just in case
    trans_std = data["second_stage"].get("transshipment_costs_std", [[0.0] * N for _ in range(N)])
    rand_links = data["second_stage"].get("random_cost_links", [])
    
    num_scen = data["second_stage"]["demand_distribution"]["num_scenarios"]
    probs = data["second_stage"]["demand_distribution"]["probabilities"]
    demands = data["second_stage"]["demand_distribution"]["scenarios"]

    # Extract random seed from the dataset optimally, falling back to 42
    seed = 42
    for k in ["seed", "random_seed"]:
        if k in data:
            seed = data[k]
            break
        elif k in data.get("second_stage", {}):
            seed = data["second_stage"][k]
            break
    
    np.random.seed(seed)

    # Pre-compute random transshipment cost realizations per scenario
    C = np.zeros((num_scen, N, N))
    for w in range(num_scen):
        for i in range(N):
            for j in range(N):
                C[w, i, j] = trans_mean[i][j]

        # Explicitly sample normal distribution for probabilistic edges
        for link in rand_links:
            i, j = link
            val = np.random.normal(trans_mean[i][j], trans_std[i][j])
            C[w, i, j] = val  # The prompt denotes "drawn from a normal distribution" explicitly.

    # Formulate optimization model
    m = gp.Model("InventoryTransshipment")
    if args.time_limit:
        m.Params.TimeLimit = args.time_limit
    
    # Configure solver parameters per prompt restrictions
    m.Params.Threads = 1       # Single CPU core restriction
    m.Params.Method = 2        # Barrier Method typically solves block LP significantly faster
    m.Params.OutputFlag = 0

    # Decision variables
    s = m.addVars(N, lb=s_lb, vtype=GRB.CONTINUOUS, name="s")
    e = m.addVars(num_scen, N, lb=0.0, vtype=GRB.CONTINUOUS, name="e")
    f = m.addVars(num_scen, N, lb=0.0, vtype=GRB.CONTINUOUS, name="f")
    q = m.addVars(num_scen, N, lb=0.0, vtype=GRB.CONTINUOUS, name="q")
    r = m.addVars(num_scen, N, lb=0.0, vtype=GRB.CONTINUOUS, name="r")
    t = m.addVars(num_scen, N, N, lb=0.0, vtype=GRB.CONTINUOUS, name="t")

    # Disable self-loops along the identity trace for bounds robustness
    for w in range(num_scen):
        for i in range(N):
            t[w, i, i].ub = 0.0

    # Objective Function Initialization
    obj_expr = gp.LinExpr()
    
    # Create balances mathematically as expressed sequentially in descriptions
    for w in range(num_scen):
        prob = probs[w]
        
        # Scenario operational boundaries validation
        dem_sum = sum(demands[w])

        for i in range(N):
            # Accumulate cost to objective structure
            obj_expr.addTerms(prob * holding[i], e[w, i])
            obj_expr.addTerms(prob * penalty[i], r[w, i])
            
            # (Rule 1) Base inventory consumed
            # stock locally + aggregate stock transshipped out + stock preserved == Initial stocking decision
            m.addLConstr(f[w, i] + gp.quicksum(t[w, i, j] for j in range(N) if i != j) + e[w, i] == s[i])
            
            # (Rule 2) Market demand fulfillment
            # stock locally + aggregate stock replenished from outside + lost sales backlog == Ground demand realized
            m.addLConstr(f[w, i] + gp.quicksum(t[w, j, i] for j in range(N) if i != j) + r[w, i] == demands[w][i])
            
            # (Rule 3) Structural inventory boundary limit replenishment requirement formulation
            # stock preserved + additional replenishment threshold limit == Defined original inventory decision
            m.addLConstr(e[w, i] + q[w, i] == s[i])
            
            for j in range(N):
                if i != j:
                    obj_expr.addTerms(prob * C[w, i, j], t[w, i, j])

        # (Rule 4) Network systemic verification balance rule across entire macro-system
        m.addLConstr(gp.quicksum(r[w, i] for i in range(N)) + gp.quicksum(q[w, i] for i in range(N)) == dem_sum)

    # Execute Solution process 
    m.setObjective(obj_expr, GRB.MINIMIZE)
    m.optimize()

    # Solution Extraction Logic ensuring adherence safely to bounds
    if m.SolCount > 0:
        if logger:
            logger.log(m.ObjVal)

        scenarios_out = []
        for w in range(num_scen):
            e_val = [max(0.0, e[w, i].X) for i in range(N)]
            f_val = [max(0.0, f[w, i].X) for i in range(N)]
            q_val = [max(0.0, q[w, i].X) for i in range(N)]
            r_val = [max(0.0, r[w, i].X) for i in range(N)]
            
            t_val = [[0.0] * N for _ in range(N)]
            for i in range(N):
                for j in range(N):
                    if i != j:
                        t_val[i][j] = max(0.0, t[w, i, j].X)
                        
            scenarios_out.append({
                "e": e_val,
                "f": f_val,
                "q": q_val,
                "r": r_val,
                "t": t_val
            })

        solution_out = {
            "objective_value": m.ObjVal,
            "s": [max(0.0, s[i].X) for i in range(N)],
            "scenarios": scenarios_out
        }

    else:
        # Fallback pseudo-zero baseline response if unresolved safely bypassing internal breakdown
        scenarios_out = [{"e": [0.0]*N, "f": [0.0]*N, "q": [0.0]*N, "r": [0.0]*N, "t": [[0.0]*N for _ in range(N)]} for _ in range(num_scen)]
        solution_out = {
            "objective_value": float('inf'),
            "s": [max(0.0, x) for x in s_lb],
            "scenarios": scenarios_out
        }

    # Stream out standardized JSON matching required schema outputs precisely 
    with open(args.solution_path, "w") as f:
        json.dump(solution_out, f, indent=4)

if __name__ == "__main__":
    solve()