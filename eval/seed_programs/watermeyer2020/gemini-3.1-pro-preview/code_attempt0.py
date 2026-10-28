import sys
import json
import argparse
import gurobipy as gp
from gurobipy import GRB

# Attempt to import solution_logger, mock it if not found locally
try:
    from solution_logger import SolutionLogger
except ImportError:
    class SolutionLogger:
        def __init__(self, log_path, sense="minimize"):
            pass
        def log(self, objective_value):
            pass

def solve():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    # Load instance data
    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    d_bar = data["d_bar"]
    num_activities = data["num_activities"]
    activities_data = data["activities"]
    temporal_constraints = data["temporal_constraints"]
    resources = data["resources"]

    # Parse processing times and resource demands
    p = [0] * num_activities
    q = {r: [0] * num_activities for r in resources}

    for act_id_str, act_data in activities_data.items():
        act_id = int(act_id_str)
        p[act_id] = act_data.get("processing_time", 0)
        r_demands = act_data.get("resource_demands", {})
        for r_id_str, demand in r_demands.items():
            if r_id_str in q:
                q[r_id_str][act_id] = demand

    # Initialize Floyd-Warshall to compute all-pairs longest paths (Earliest/Latest Start Times)
    inf = float('inf')
    FW = [[-inf] * num_activities for _ in range(num_activities)]
    for i in range(num_activities):
        FW[i][i] = 0

    # Apply given temporal constraints S_j - S_i >= delta
    for c in temporal_constraints:
        u = c['i']
        v = c['j']
        w = c['delta']
        FW[u][v] = max(FW[u][v], w)

    # Universally enforce sequence bounds to ensure valid time window horizons up to predefined max duration
    for i in range(num_activities):
        # S_i - S_0 >= 0 (All start times >= 0)
        FW[0][i] = max(FW[0][i], 0)
        # S_0 - S_i >= -d_bar => S_i <= d_bar (All start times <= maximum project duration)
        FW[i][0] = max(FW[i][0], -d_bar)

    # Compute longest distances
    for k in range(num_activities):
        for i in range(num_activities):
            for j in range(num_activities):
                if FW[i][k] != -inf and FW[k][j] != -inf:
                    path_len = FW[i][k] + FW[k][j]
                    if path_len > FW[i][j]:
                        FW[i][j] = path_len

    # Identify temporal cycle infeasibilities
    for i in range(num_activities):
        if FW[i][i] > 0:
            raise ValueError(f"Infeasible temporal constraints (positive cycle detected at activity {i}).")

    # Time boundaries (E: Early, L: Late start times)
    E = [0] * num_activities
    L = [0] * num_activities
    for i in range(num_activities):
        E[i] = FW[0][i]
        L[i] = -FW[i][0]

    # Initialize Mathematical Model
    model = gp.Model()
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', args.time_limit)

    # Step variables dictionary Y_{i,t} = 1 iff S_i <= t
    Y = {}
    for i in range(num_activities):
        for t in range(E[i], L[i]):
            Y[(i, t)] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{t}")

    # Helper function to access step functions resolving bounds intelligently 
    def get_Y(idx, t_val):
        if t_val < E[idx]: return 0
        if t_val >= L[idx]: return 1
        return Y[(idx, t_val)]

    # Temporal Sequence Constraints Formulation mapping to Y_j_t <= Y_i_{t-delta}
    for c in temporal_constraints:
        i = c['i']
        j = c['j']
        delta = c['delta']
        for t in range(E[j], L[j]):
            y_j = get_Y(j, t)
            y_i = get_Y(i, t - delta)
            if isinstance(y_i, int) and y_i == 1:
                continue
            model.addConstr(y_j <= y_i)

    # Monotonic step limits configuration
    for i in range(num_activities):
        for t in range(E[i], L[i] - 1):
            model.addConstr(Y[(i, t)] <= Y[(i, t+1)])

    # Partially Renewable Resource Constraints Formulation using algebraic telescope optimizations
    for r_id_str, r_data in resources.items():
        capacity = r_data["capacity"]
        periods = set(r_data["periods"])
        
        expr = gp.LinExpr(0)
        for i in range(num_activities):
            demand = q[r_id_str][i]
            if demand == 0:
                continue
            
            # Formulating constant base for S_i = L_i bound tracking
            A_L = sum(demand for k in range(L[i] + 1, L[i] + p[i] + 1) if k in periods)
            expr.addConstant(A_L)
            
            # Efficient summation mapped specifically back into decision domain bounds
            for k in periods:
                t1 = k - 1
                if E[i] <= t1 < L[i]:
                    expr.addTerms(demand, Y[(i, t1)])
                
                t2 = k - p[i] - 1
                if E[i] <= t2 < L[i]:
                    expr.addTerms(-demand, Y[(i, t2)])
                    
        model.addConstr(expr <= capacity, name=f"res_{r_id_str}")

    # Objective Function: Minimize S_{n_end} => Minimize(L - Ys[n_end])
    n_end = num_activities - 1 # project end activity index
    obj_expr = gp.LinExpr(L[n_end])
    for t in range(E[n_end], L[n_end]):
        obj_expr.addTerms(-1.0, Y[(n_end, t)])

    model.setObjective(obj_expr, GRB.MINIMIZE)

    # Initialize incumbent logging hook mechanism
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    def cb(mdl, where):
        if where == GRB.Callback.MIPSOL:
            obj = mdl.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(int(round(obj)))

    # Execute search phase
    model.optimize(cb)

    # Extraction and Schema mapping Phase
    if model.SolCount > 0:
        obj_val = int(round(model.ObjVal))
        start_times = {}
        
        for i in range(num_activities):
            s_i = L[i]
            for t in range(E[i], L[i]):
                if Y[(i, t)].X > 0.5:
                    s_i -= 1
            start_times[str(i)] = s_i

        solution = {
            "objective_value": obj_val,
            "start_times": start_times
        }
        
        # Deploy standardized output protocol format execution
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f)

if __name__ == "__main__":
    solve()