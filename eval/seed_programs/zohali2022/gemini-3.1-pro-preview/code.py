import sys
import argparse
import json
import gurobipy as gp
from gurobipy import GRB

def solve():
    # 1. Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default="")
    args = parser.parse_args()

    # 2. Setup SolutionLogger if path is provided
    logger = None
    if args.log_path:
        try:
            from solution_logger import SolutionLogger
            logger = SolutionLogger(args.log_path, sense="minimize")
        except ImportError:
            pass

    # 3. Read instance data
    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    N = data["num_tasks"]
    S = data["num_stations"]
    P = data["processing_times"]
    F_mat = data["forward_setup_times"]
    B_mat = data["backward_setup_times"]
    C_bound = data.get("cycle_time", None)

    # Autodetect if precedence indices are 1-based or 0-based
    is_1_based_prec = False
    prec_list = data.get("precedence_relations", [])
    if prec_list:
        max_idx = max(max(pair) for pair in prec_list)
        if max_idx == N:
            is_1_based_prec = True

    precedences = []
    for u_node, v_node in prec_list:
        if is_1_based_prec:
            precedences.append((u_node - 1, v_node - 1))
        else:
            precedences.append((u_node, v_node))

    # 4. Initialize Gurobi environment and model
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()

    m = gp.Model("UALBP", env=env)
    m.setParam("TimeLimit", args.time_limit)
    m.setParam("Threads", 1)  # Single core constraint

    # 5. Define variables
    # z[i,k] = 1 if task i is assigned to station k
    z = m.addVars(N, S, vtype=GRB.BINARY, name="z")
    # F[i,j] = 1 if task i immediately precedes task j in the SAME station (forward setup)
    F = m.addVars(N, N, vtype=GRB.BINARY, name="F")
    # B[i,j] = 1 if task i is last and task j is first in the SAME station (backward setup)
    B = m.addVars(N, N, vtype=GRB.BINARY, name="B")
    
    # Auxiliary variables logic
    y = m.addVars(N, S, vtype=GRB.CONTINUOUS, lb=0, name="y")
    u = m.addVars(N, vtype=GRB.CONTINUOUS, lb=1, ub=N, name="u")
    w = m.addVars(N, S, vtype=GRB.CONTINUOUS, lb=0, name="w")
    C = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="C")

    # 6. Add Constraints
    # (a) Each task to exactly one station
    for i in range(N):
        m.addConstr(gp.quicksum(z[i, k] for k in range(S)) == 1)

    # (b) Every station must be assigned at least one task
    for k in range(S):
        m.addConstr(gp.quicksum(z[i, k] for i in range(N)) >= 1)

    # (c) Basic graph definition blocks
    for i in range(N):
        m.addConstr(F[i, i] == 0)

    # (d) Degree constraints to form cyclic paths (exactly 1 successor and predecessor)
    for i in range(N):
        m.addConstr(gp.quicksum(F[i, j] for j in range(N) if j != i) + gp.quicksum(B[i, j] for j in range(N)) == 1)
        m.addConstr(gp.quicksum(F[j, i] for j in range(N) if j != i) + gp.quicksum(B[j, i] for j in range(N)) == 1)

    # (e) Connect F/B edges strictly inside stations assignment
    for i in range(N):
        for j in range(N):
            if i != j:
                for k in range(S):
                    m.addConstr(z[i, k] - z[j, k] <= 1 - F[i, j] - B[i, j])
                    m.addConstr(z[j, k] - z[i, k] <= 1 - F[i, j] - B[i, j])

    # (f) Subtour elimination for F edges
    for i in range(N):
        for j in range(N):
            if i != j:
                m.addConstr(u[j] >= u[i] + 1 - N * (1 - F[i, j]))

    # (g) Linearized precedences handling both station ordering and cyclic execution sequences
    for (i, j) in precedences:
        st_i = gp.quicksum(k * z[i, k] for k in range(S))
        st_j = gp.quicksum(k * z[j, k] for k in range(S))
        m.addConstr(N * st_i + u[i] <= N * st_j + u[j] - 1)

    # (h) Exactly 1 backward edge (B) strictly permitted per station
    for i in range(N):
        B_out_i = gp.quicksum(B[i, j] for j in range(N))
        for k in range(S):
            m.addConstr(y[i, k] <= B_out_i)
            m.addConstr(y[i, k] <= z[i, k])
            m.addConstr(y[i, k] >= B_out_i + z[i, k] - 1)

    for k in range(S):
        m.addConstr(gp.quicksum(y[i, k] for i in range(N)) == 1)

    # (i) Exact cycle time bounding and evaluation per station
    M = [0] * N
    for i in range(N):
        max_f = max([F_mat[i][j] for j in range(N) if j != i]) if N > 1 else 0
        max_b = max([B_mat[i][j] for j in range(N)])
        M[i] = P[i] + max(max_f, max_b)

    for i in range(N):
        c_i = P[i] + gp.quicksum(F_mat[i][j] * F[i, j] for j in range(N)) + gp.quicksum(B_mat[i][j] * B[i, j] for j in range(N))
        for k in range(S):
            m.addConstr(w[i, k] >= 0)
            m.addConstr(w[i, k] <= M[i] * z[i, k])
            m.addConstr(w[i, k] <= c_i)
            m.addConstr(w[i, k] >= c_i - M[i] * (1 - z[i, k]))

    # Max bottleneck
    for k in range(S):
        m.addConstr(C >= gp.quicksum(w[i, k] for i in range(N)))
        
    m.addConstr(C >= sum(P) / S)
    m.addConstr(C >= max(P))

    # Upper bound adherence
    if C_bound is not None:
        m.addConstr(C <= C_bound)

    # Objective
    m.setObjective(C, GRB.MINIMIZE)

    # 7. Optimize via a callback logging solutions
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            try:
                obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if logger:
                    logger.log(obj)
            except Exception:
                pass

    m.optimize(callback)

    # 8. Unpack best-discovered solution mapping bounds against schema standards
    if m.SolCount > 0:
        obj_val = m.ObjVal
        assignment = {}
        station_sequences = {}

        z_val = m.getAttr('X', z)
        F_val = m.getAttr('X', F)
        B_val = m.getAttr('X', B)

        for i in range(N):
            for k in range(S):
                if z_val[i, k] > 0.5:
                    assignment[str(i + 1)] = k + 1

        for k in range(S):
            nodes_k = [i for i in range(N) if z_val[i, k] > 0.5]
            if not nodes_k:
                station_sequences[str(k + 1)] = []
                continue
            
            start_node = None
            for i in nodes_k:
                for j in nodes_k:
                    if B_val[i, j] > 0.5:
                        start_node = j
                        break
                if start_node is not None:
                    break
                    
            if start_node is None:
                start_node = nodes_k[0]
                
            seq = [start_node]
            curr = start_node
            while len(seq) < len(nodes_k):
                next_n = None
                for j in nodes_k:
                    if curr != j and F_val[curr, j] > 0.5:
                        next_n = j
                        break
                if next_n is None:
                    break
                seq.append(next_n)
                curr = next_n
                
            station_sequences[str(k + 1)] = [node + 1 for node in seq]
    else:
        # Fallback trivial structure protecting strictly required JSON execution paths dynamically under cutoff
        obj_val = float('inf')
        assignment = {str(i + 1): (i % S) + 1 for i in range(N)}
        station_sequences = {str(k + 1): [] for k in range(S)}
        for i in range(N):
            station_sequences[str((i % S) + 1)].append(i + 1)

    # 9. JSON Output Formatting
    out_dict = {
        "objective_value": float(obj_val),
        "assignment": assignment,
        "station_sequences": station_sequences,
        "cycle_time": float(obj_val)
    }

    with open(args.solution_path, 'w') as f:
        json.dump(out_dict, f, indent=2)

if __name__ == "__main__":
    solve()