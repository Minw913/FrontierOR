import sys
import json
import argparse
import itertools
import random

# Gurobipy is required for solving the optimization problem
try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    print("Gurobipy is not installed.")
    sys.exit(1)

# SolutionLogger for logging intermediate solutions
try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None

def solve():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True, help="Path to the instance JSON file.")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to write the output solution JSON file.")
    parser.add_argument("--time_limit", type=int, default=3600, help="Maximum runtime in seconds.")
    parser.add_argument("--log_path", type=str, default=None, help="Path for convergence logging.")
    args = parser.parse_args()
    
    # Read the instance JSON file
    with open(args.instance_path, "r") as f:
        instance = json.load(f)
        
    # Initialize solution logger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path and SolutionLogger else None
    
    # Initialize random seed. Fallback to 0 if not present in the instance.
    seed = instance.get("seed")
    if seed is None:
        seed = 0
    random.seed(seed)
    
    # Helper to deterministically enumerate admissible shift-level QC sequences
    def get_seqs(prof):
        seqs = []
        min_qc = prof["min_qc"]
        max_qc = prof["max_qc"]
        min_s = prof["min_handling_shifts"]
        max_s = prof["max_handling_shifts"]
        # Enumerate all sequences for each valid shift length and sort lexicographically
        for L in range(min_s, max_s + 1):
            for s in itertools.product(range(min_qc, max_qc + 1), repeat=L):
                seqs.append(s)
        return seqs

    # Generate and sample QC profiles for big and small vessels
    big_prof = instance["qc_profiles"]["big"]
    small_prof = instance["qc_profiles"]["small"]
    
    all_big = get_seqs(big_prof)
    sampled_big = random.sample(all_big, min(len(all_big), instance["num_big_profiles"]))
    
    all_small = get_seqs(small_prof)
    sampled_small = random.sample(all_small, min(len(all_small), instance["num_small_profiles"]))
    
    # Form the global pool of profiles
    global_seqs = sampled_big + sampled_small
    
    # Setup Gurobi model and environment parameters
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    model = gp.Model("BerthAllocation", env=env)
    
    if args.time_limit:
        model.setParam("TimeLimit", args.time_limit)
        
    model.setParam("Seed", 0)
    model.setParam("MIPGap", 1e-4)
    model.setParam("NumericFocus", 0)
    model.setParam("Threads", 1)
    
    # Extract terminal and penalty parameters
    S = instance["time_steps_per_shift"]
    T = instance["total_time_steps"]
    steps_per_hour = T / instance["time_horizon_hours"]
    
    c1 = instance["c1"]
    c2 = instance["c2"]
    
    # Dictionaries to aggregate variable indices for constraint building
    x_by_v = {v["id"]: [] for v in instance["vessels"]}
    x_by_k_tau = { (k["id"], tau): [] for k in instance["berths"] for tau in range(T) }
    x_by_tau = { tau: [] for tau in range(T) }
    
    all_vars = []
    all_var_info = []
    
    # Variable creation
    for v in instance["vessels"]:
        v_id = v["id"]
        D_v = int(round(v["duration_hours"] * steps_per_hour))
        a = v["a"]
        b = v["b"]
        k_bar = v["k_bar"]
        t_bar = v["t_bar"]
        
        for berth in instance["berths"]:
            k_id = berth["id"]
            a_k = berth["a_k"]
            b_k = berth["b_k"]
            
            # Start time window bounds matching both vessel and berth feasibility
            min_t = max(a, a_k)
            max_t = min(b, b_k - D_v + 1)
            
            for t in range(min_t, max_t + 1):
                # Expected QC sequence length given the actual start time step
                L = (t + D_v - 1) // S - t // S + 1
                
                for p_idx, seq in enumerate(global_seqs):
                    # Filter profile based on spanned shift length
                    if len(seq) == L:
                        # Deviation penalty mapping
                        cost = c1 * abs(k_id - k_bar) + c2 * abs(t - t_bar)
                        
                        var = model.addVar(vtype=GRB.BINARY, obj=cost)
                        var_info = (v_id, k_id, t, p_idx, D_v)
                        
                        all_vars.append(var)
                        all_var_info.append(var_info)
                        
                        x_by_v[v_id].append(var)
                        
                        # Populate temporal occupancy structures efficiently
                        for tau in range(t, t + D_v):
                            x_by_k_tau[(k_id, tau)].append(var)
                            qc_usage = seq[tau // S - t // S]
                            x_by_tau[tau].append(qc_usage * var)
                            
    # Constraint 1: Exact assignment per vessel
    for v in instance["vessels"]:
        v_id = v["id"]
        # If x_by_v[v_id] is empty, assigning it to 1 renders the model correctly infeasible
        model.addConstr(gp.quicksum(x_by_v[v_id]) == 1, name=f"assign_{v_id}")
            
    # Constraint 2: Non-overlapping berth assignments (Berth capacity)
    for k in instance["berths"]:
        k_id = k["id"]
        for tau in range(T):
            if x_by_k_tau[(k_id, tau)]:
                model.addConstr(gp.quicksum(x_by_k_tau[(k_id, tau)]) <= 1, name=f"berth_cap_{k_id}_{tau}")
                
    # Constraint 3: Terminal-wide Quay Crane available limit
    total_qc = instance["total_qc"]
    for tau in range(T):
        if x_by_tau[tau]:
            model.addConstr(gp.quicksum(x_by_tau[tau]) <= total_qc, name=f"qc_cap_{tau}")
            
    # MIPSOL Callback to log feasible solutions throughout the search process
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                try:
                    sol_vals = model.cbGetSolution(all_vars)
                    vessels_out = []
                    for val, info in zip(sol_vals, all_var_info):
                        if val > 0.5:
                            v_id, k_id, t, p_idx, D_v = info
                            vessels_out.append({
                                "id": v_id,
                                "berth": k_id,
                                "start_time": t,
                                "end_time": t + D_v - 1,
                                "profile": p_idx
                            })
                    sol_dict = {
                        "objective_value": obj,
                        "vessels": vessels_out
                    }
                    logger.log_solution(obj, sol_dict)
                except Exception:
                    logger.log(obj)
                    
    # Optimize
    model.optimize(callback)
    
    # Save best incumbent solution
    if model.SolCount > 0:
        obj = model.ObjVal
        vessels_out = []
        for var, info in zip(all_vars, all_var_info):
            if var.X > 0.5:
                v_id, k_id, t, p_idx, D_v = info
                vessels_out.append({
                    "id": v_id,
                    "berth": k_id,
                    "start_time": t,
                    "end_time": t + D_v - 1,
                    "profile": p_idx
                })
        final_sol = {
            "objective_value": obj,
            "vessels": vessels_out
        }
    else:
        final_sol = {
            "objective_value": None,
            "vessels": []
        }
        
    with open(args.solution_path, "w") as f:
        json.dump(final_sol, f, indent=4)

if __name__ == "__main__":
    solve()