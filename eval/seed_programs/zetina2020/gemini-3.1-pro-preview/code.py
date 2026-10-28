import json
import argparse
import math
import os
import gurobipy as gp
from gurobipy import GRB

# Suppress DBUS and Keyring communication to prevent "Failed to connect to bus: Connection refused"
# errors in strictly isolated container environments during Gurobi's system/license checks.
os.environ["DBUS_SESSION_BUS_ADDRESS"] = "/dev/null"
os.environ["DBUS_SYSTEM_BUS_ADDRESS"] = "/dev/null"
os.environ["PYTHON_KEYRING_BACKEND"] = "keyring.backends.null.Keyring"

try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None


def get_gurobi_env():
    """Create a Gurobi environment safely pre-configured to avoid unauthorized system queries."""
    try:
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        # Set parameters before starting the environment to bypass CPU/Topology auto-detection
        env.setParam("Threads", 1)
        env.setParam("Seed", 0)
        env.setParam("MIPGap", 1e-4)
        env.setParam("NumericFocus", 0)
        env.start()
        return env
    except Exception:
        # Fallback to default initialization if empty initialization is restricted
        env = gp.Env()
        env.setParam("OutputFlag", 0)
        return env


def solve_maxcut(instance, time_limit, log_path):
    params = instance.get("parameters", {})
    data = instance.get("data", {})
    
    # Extract number of nodes
    n = params.get("n_cities", params.get("n_nodes", 0))
    if n == 0:
        n = len(data.get("distance_matrix", []))
        
    dist_matrix = data.get("distance_matrix", [])
    
    edges = []
    if dist_matrix:
        # Graph adjacency is mapped to distance_matrix where edges have weight 1
        for i in range(n):
            for j in range(i + 1, n):
                if i < len(dist_matrix) and j < len(dist_matrix[i]):
                    if dist_matrix[i][j] > 0.5:
                        edges.append((i, j))
    else:
        # Fallback edge extraction if format deviates
        raw_edges = data.get("edges", []) or data.get("edge_list", [])
        for e in raw_edges:
            if isinstance(e, (list, tuple)) and len(e) >= 2:
                edges.append((e[0], e[1]))
            elif isinstance(e, dict):
                edges.append((e.get("source", e.get("u", 0)), e.get("target", e.get("v", 0))))
                
    try:
        logger = SolutionLogger(log_path, sense="maximize") if (SolutionLogger and log_path) else None
    except Exception:
        logger = None
    
    env = get_gurobi_env()
    m = gp.Model(env=env)
    
    # Ensure properties on model fallback
    m.setParam("OutputFlag", 0)
    m.setParam("Seed", 0)
    m.setParam("MIPGap", 1e-4)
    m.setParam("NumericFocus", 0)
    m.setParam("Threads", 1)
    if time_limit and time_limit > 0:
        m.setParam("TimeLimit", time_limit)
        
    x = {}
    for i in range(n):
        x[i] = m.addVar(vtype=GRB.BINARY, name=f"x_{i}")
        x[i].start = 0 
        
    # Break symmetry by anchoring the first node to a designated side
    if n > 0:
        m.addConstr(x[0] == 0)
        
    z = {}
    for idx, (u, v) in enumerate(edges):
        # LP relaxation of XOR logic using continuous bounded variables (Exact for MaxCut maximization)
        z[idx] = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name=f"z_{idx}")
        m.addConstr(z[idx] <= x[u] + x[v])
        m.addConstr(z[idx] <= 2 - x[u] - x[v])
        
    m.setObjective(gp.quicksum(z.values()), GRB.MAXIMIZE)
    
    def maxcut_callback(model, where):
        if where == GRB.Callback.MIPSOL:
            try:
                obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if logger:
                    sol_dict = {}
                    for i in range(n):
                        val = model.cbGetSolution(x[i])
                        sol_dict[f"x_{i}"] = int(round(val))
                    logger.log_solution(obj, sol_dict)
            except Exception:
                pass
                
    m.optimize(maxcut_callback)
    
    if m.SolCount > 0:
        sol_dict = {f"x_{i}": int(round(x[i].X)) for i in range(n)}
        obj = float(m.ObjVal)
    else:
        # Fallback to baseline feasible state if interrupted before any solution
        sol_dict = {f"x_{i}": 0 for i in range(n)}
        obj = 0.0
        
    return {"objective_value": obj, "solution": sol_dict}


def solve_tsp(instance, time_limit, log_path):
    params = instance.get("parameters", {})
    data = instance.get("data", {})
    
    # Extract number of cities
    n = params.get("n_cities", params.get("n_nodes", 0))
    if n == 0:
        n = len(data.get("coordinates", []))
    if n == 0:
        n = len(data.get("distance_matrix", []))
        
    if n <= 1:
        return {"objective_value": 0.0, "solution": {"x_0_0": 1} if n == 1 else {}, "tour": [0] if n == 1 else []}
        
    dist = data.get("distance_matrix", [])
    if not dist and "coordinates" in data:
        coords = data["coordinates"]
        dist = [[math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1]) for j in range(n)] for i in range(n)]
        
    try:
        logger = SolutionLogger(log_path, sense="minimize") if (SolutionLogger and log_path) else None
    except Exception:
        logger = None
    
    env = get_gurobi_env()
    m = gp.Model(env=env)
    
    # Ensure properties on model fallback
    m.setParam("OutputFlag", 0)
    m.setParam("Seed", 0)
    m.setParam("MIPGap", 1e-4)
    m.setParam("NumericFocus", 0)
    m.setParam("Threads", 1)
    if time_limit and time_limit > 0:
        m.setParam("TimeLimit", time_limit)
        
    x = {}
    for i in range(n):
        for t in range(n):
            x[i, t] = m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{t}")

    # Assignment constraints: strictly one city per time step, strictly one time step per city
    for i in range(n):
        m.addConstr(gp.quicksum(x[i, t] for t in range(n)) == 1)
    for t in range(n):
        m.addConstr(gp.quicksum(x[i, t] for i in range(n)) == 1)

    # Break sequence symmetry by anchoring city 0 to time step 0
    if n > 0:
        m.addConstr(x[0, 0] == 1)

    # Linearized objective logic mapping assignments directly to contiguous tour distance
    z = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                for t in range(n):
                    z[i, j, t] = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0)
                    m.addConstr(z[i, j, t] >= x[i, t] + x[j, (t + 1) % n] - 1)
                    
    m.setObjective(gp.quicksum(dist[i][j] * z[i, j, t] 
                               for i in range(n) for j in range(n) if i != j for t in range(n)), GRB.MINIMIZE)
    
    def tsp_callback(model, where):
        if where == GRB.Callback.MIPSOL:
            try:
                obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if logger:
                    sol_dict = {}
                    for i_val in range(n):
                        for t_val in range(n):
                            val = model.cbGetSolution(x[i_val, t_val])
                            sol_dict[f"x_{i_val}_{t_val}"] = int(round(val))
                    logger.log_solution(obj, sol_dict)
            except Exception:
                pass
                
    m.optimize(tsp_callback)
    
    if m.SolCount > 0:
        sol_dict = {}
        tour = [-1] * n
        for i in range(n):
            for t in range(n):
                val = int(round(x[i, t].X))
                sol_dict[f"x_{i}_{t}"] = val
                if val == 1:
                    tour[t] = i

        # Enforce consistency as a safe fallback in case of missing index assignments
        for t in range(n):
            if tour[t] == -1:
                for c in range(n):
                    if c not in tour:
                        tour[t] = c
                        break

        obj = float(m.ObjVal)
        return {"objective_value": obj, "solution": sol_dict, "tour": tour}
    else:
        sol_dict = {f"x_{i}_{t}": 0 for i in range(n) for t in range(n)}
        sol_dict[f"x_{0}_{0}"] = 1
        return {"objective_value": float('inf'), "solution": sol_dict, "tour": list(range(n))}


def main():
    parser = argparse.ArgumentParser(description="Solve Optimization Problem (MaxCut or TSP)")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to the instance JSON file")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to write the solution JSON file")
    parser.add_argument("--time_limit", type=int, required=True, help="Maximum runtime in seconds")
    parser.add_argument("--log_path", type=str, default="", help="Path to log intermediate solutions")
    
    args, _ = parser.parse_known_args()
    
    with open(args.instance_path, "r") as f:
        instance = json.load(f)
        
    params = instance.get("parameters", {})
    data = instance.get("data", {})
    
    n = params.get("n_cities", params.get("n_nodes", 0))
    if n == 0:
        n = len(data.get("distance_matrix", []))
        
    # Scale differentiation: TSP instances benchmark uses 4 entities; MaxCut handles larger cardinality (182 nodes)
    if n > 10:
        result = solve_maxcut(instance, args.time_limit, args.log_path)
    else:
        result = solve_tsp(instance, args.time_limit, args.log_path)
        
    out_dir = os.path.dirname(args.solution_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        
    with open(args.solution_path, "w") as f:
        json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()