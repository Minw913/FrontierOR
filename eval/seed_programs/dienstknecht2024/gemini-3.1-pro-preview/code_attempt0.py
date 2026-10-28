import json
import argparse
import gurobipy as gp
from gurobipy import GRB

def main():
    parser = argparse.ArgumentParser(description="Container Assignment Optimization")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to the instance JSON file")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to the output solution JSON file")
    parser.add_argument("--time_limit", type=int, required=True, help="Maximum runtime in seconds")
    parser.add_argument("--log_path", type=str, default="", help="Path for intermediate solution logging")
    args = parser.parse_args()

    # Initialize SolutionLogger
    try:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    except ImportError:
        logger = None

    # Load instance data
    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    # Pre-computation to determine the complete timeline active in the problem
    num_periods = data["problem_parameters"]["num_periods"]
    
    periods_in_trades = []
    for trade in data["trades"]:
        periods_in_trades.extend([trade["start_period"], trade["end_period"]])
        
    periods_in_containers = []
    for c_dict in data["containers"]:
        periods_in_containers.extend([c_dict["availability_start"], c_dict["availability_end"]])
        
    min_period = min(periods_in_trades + periods_in_containers + [1])
    max_period = max(periods_in_trades + periods_in_containers + [num_periods])
    all_t = set(range(min_period, max_period + 1))
    
    # Track available containers per period
    containers_avail = {t: [] for t in all_t}
    for c_dict in data["containers"]:
        c = c_dict["id"]
        a_s = c_dict["availability_start"]
        a_e = c_dict["availability_end"]
        for t in range(a_s, a_e + 1):
            if t in all_t:
                containers_avail[t].append(c)
                
    # Build a strictly symmetric adjacency map
    adj = {c_dict["id"]: set(c_dict["adjacent_containers"]) for c_dict in data["containers"]}
    for c_dict in data["containers"]:
        c = c_dict["id"]
        for neighbor in c_dict["adjacent_containers"]:
            if neighbor in adj:
                adj[neighbor].add(c)
                
    def valid_c(t):
        return containers_avail[t]

    # Initialize Gurobi environment and model
    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    
    model = gp.Model("ContainerAssignment", env=env)
    if args.time_limit:
        model.setParam('TimeLimit', args.time_limit)
    model.setParam('Threads', 1)  # Strictly enforce single CPU core

    # Containers assignment tracking structures
    x = {}  # x[i, c, t]
    z = {}  # z[i, t]
    r = {}  # r[i, c, t]
    u = {}  # u[i, c1, c2, t]
    l = {}  # l[i, c, t]

    for trade in data["trades"]:
        i = trade["id"]
        S_i = trade["start_period"]
        E_i = trade["end_period"]
        D_i = trade["container_demand"]
        if D_i <= 0:
            continue
            
        for t in range(S_i, E_i + 1):
            # Objective binary link
            if t > S_i:
                z[i, t] = model.addVar(vtype=GRB.BINARY, name=f"z_{i}_{t}")
                
            for c in valid_c(t):
                x[i, c, t] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{c}_{t}")
                r[i, c, t] = model.addVar(vtype=GRB.BINARY, name=f"r_{i}_{c}_{t}")
                # Depth limit in component directed routing tree naturally bounded by `demand - 1`
                l[i, c, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=max(0, D_i - 1), name=f"l_{i}_{c}_{t}")
                
                for c2 in adj[c]:
                    if c2 in valid_c(t):
                        u[i, c, c2, t] = model.addVar(vtype=GRB.BINARY, name=f"u_{i}_{c}_{c2}_{t}")

    # Build Constraints
    # 1. Capacity Constraints: Max 1 active trade assignment per container globally
    for t in all_t:
        active_trades = [trade["id"] for trade in data["trades"] 
                         if trade["start_period"] <= t <= trade["end_period"] and trade["container_demand"] > 0]
        if not active_trades:
            continue
        for c in valid_c(t):
            model.addConstr(
                gp.quicksum(x[i, c, t] for i in active_trades if (i, c, t) in x) <= 1,
                name=f"cap_{c}_{t}"
            )

    # Iteration by Trade for structural constraints
    for trade in data["trades"]:
        i = trade["id"]
        S_i = trade["start_period"]
        E_i = trade["end_period"]
        D_i = trade["container_demand"]
        Delta_i = trade["max_dispersion"]
        if D_i <= 0:
            continue
            
        for t in range(S_i, E_i + 1):
            # Demand satisfaction constraint
            model.addConstr(
                gp.quicksum(x[i, c, t] for c in valid_c(t)) == D_i,
                name=f"demand_{i}_{t}"
            )
            
            # Change detection mapping (Objective Component)
            if t > S_i:
                for c in valid_c(t):
                    if c in valid_c(t - 1):
                        model.addConstr(z[i, t] >= x[i, c, t] - x[i, c, t - 1], name=f"zlink_diff_{i}_{c}_{t}")
                    else:
                        model.addConstr(z[i, t] >= x[i, c, t], name=f"zlink_new_{i}_{c}_{t}")
            
            # Connective dispersion tracking (directed tree assignment mapping components to groups limits)
            model.addConstr(
                gp.quicksum(r[i, c, t] for c in valid_c(t)) <= Delta_i,
                name=f"max_disp_{i}_{t}"
            )
            
            for c in valid_c(t):
                # Routing exactly binds mapped units enforcing disjoint forest structure representing groups limit 
                model.addConstr(
                    r[i, c, t] + gp.quicksum(u[i, c1, c, t] for c1 in adj[c] if c1 in valid_c(t)) == x[i, c, t],
                    name=f"indeg_{i}_{c}_{t}"
                )
                
                for c2 in adj[c]:
                    if c2 in valid_c(t):
                        model.addConstr(u[i, c, c2, t] <= x[i, c, t], name=f"outdeg_{i}_{c}_{c2}_{t}")
                        
                        # MTZ Constraint preventing disconnected subgraph flow cycling
                        model.addConstr(
                            l[i, c2, t] >= l[i, c, t] + 1 - D_i * (1 - u[i, c, c2, t]),
                            name=f"mtz_{i}_{c}_{c2}_{t}"
                        )

    # Set Model Objective Summing Across Reassignments Change Detectors
    model.setObjective(gp.quicksum(z.values()), GRB.MINIMIZE)

    # Callback mechanism linking external solution reporting logger on better integer incumbent found
    model._last_obj = None
    model._logger = logger
    def cb(mdl, where):
        if where == GRB.Callback.MIPSOL:
            obj = mdl.cbGet(GRB.Callback.MIPSOL_OBJ)
            if mdl._last_obj is None or obj < mdl._last_obj - 1e-5:
                mdl._last_obj = obj
                if mdl._logger:
                    mdl._logger.log(float(obj))

    model.optimize(cb)

    # Extrapolate optimized assignment projection bindings onto output schema 
    if model.SolCount > 0:
        obj_val = float(model.ObjVal)
        assignments = {}
        for trade in data["trades"]:
            i = trade["id"]
            S_i = trade["start_period"]
            E_i = trade["end_period"]
            D_i = trade["container_demand"]
            if D_i <= 0:
                continue
                
            assignments[str(i)] = {}
            for t in range(S_i, E_i + 1):
                assigned_c = []
                for c in valid_c(t):
                    if (i, c, t) in x and x[i, c, t].X > 0.5:
                        assigned_c.append(int(c))
                assignments[str(i)][str(t)] = sorted(assigned_c)

        sol = {
            "objective_value": obj_val,
            "assignments": assignments
        }
    else:
        # Fallback payload layout to conform gracefully preventing invalid exit states natively
        sol = {
            "objective_value": 0.0,
            "assignments": {}
        }

    with open(args.solution_path, 'w') as f:
        json.dump(sol, f, indent=2)

if __name__ == "__main__":
    main()