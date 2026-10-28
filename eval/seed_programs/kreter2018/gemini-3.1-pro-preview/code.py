import sys
import argparse
import json
import gurobipy as gp

def main():
    parser = argparse.ArgumentParser(description="Resource Availability Cost Problem (RACP) Solver")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to the JSON problem instance.")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to the JSON solution output.")
    parser.add_argument("--time_limit", type=int, required=True, help="Maximum runtime in seconds.")
    parser.add_argument("--log_path", type=str, default=None, help="Path for JSONL logging.")
    args = parser.parse_args()

    # Initialize solution logger
    try:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    except ImportError:
        logger = None

    # Load instance
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    D = instance["max_project_duration"]
    num_res_types = instance["num_resource_types"]
    res_costs = instance["resource_costs"]
    act_list = instance["activities"]
    prec_list = instance["precedence_relations"]

    # Map activities
    act_ids = [act["id"] for act in act_list]
    duration = {act["id"]: act["duration"] for act in act_list}
    resources = {act["id"]: act["resource_requirements"] for act in act_list}

    # Graph construction
    adj = {i: [] for i in act_ids}
    rev_adj = {i: [] for i in act_ids}
    indegree = {i: 0 for i in act_ids}
    outdegree = {i: 0 for i in act_ids}

    for u, v in prec_list:
        adj[u].append(v)
        rev_adj[v].append(u)
        indegree[v] += 1
        outdegree[u] += 1

    # Find dummy start and end nodes
    start_nodes = [i for i in act_ids if indegree[i] == 0]
    start_node = None
    for i in start_nodes:
        if duration[i] == 0 and sum(resources[i]) == 0:
            start_node = i
            break
    if start_node is None:
        start_node = start_nodes[0] if start_nodes else act_ids[0]

    end_nodes = [i for i in act_ids if outdegree[i] == 0]
    end_node = None
    for i in end_nodes:
        if duration[i] == 0 and sum(resources[i]) == 0:
            end_node = i
            break
    if end_node is None:
        end_node = end_nodes[0] if end_nodes else act_ids[-1]

    # Compute Early Start (ES)
    ES = {i: 0 for i in act_ids}
    topo_order = []
    
    queue = list(start_nodes)
    in_deg_tmp = indegree.copy()
    while queue:
        u = queue.pop(0)
        topo_order.append(u)
        for v in adj[u]:
            if ES[v] < ES[u] + duration[u]:
                ES[v] = ES[u] + duration[u]
            in_deg_tmp[v] -= 1
            if in_deg_tmp[v] == 0:
                queue.append(v)
    
    if len(topo_order) < len(act_ids):
        raise ValueError("Cycle detected in precedence relations. Instance is not a DAG.")
    
    # Compute Late Start (LS)
    LS = {i: D for i in act_ids}
    for u in reversed(topo_order):
        for v in adj[u]:
            if LS[u] > LS[v] - duration[u]:
                LS[u] = LS[v] - duration[u]
                
    # Pin down project start strictly at time 0
    LS[start_node] = 0

    # Build Gurobi model using Step Formulation for RACP
    env = gp.Env(empty=True)
    env.setParam("LogToConsole", 1)
    env.start()

    model = gp.Model("RACP", env=env)
    model.setParam("TimeLimit", args.time_limit)
    model.setParam("Threads", 1)  # Constrained to a single CPU core

    # y[i, t] = 1 if activity i starts at or before time t
    y = {}
    for i in act_ids:
        for t in range(ES[i], LS[i]):
            y[i, t] = model.addVar(vtype=gp.GRB.BINARY, name=f"y_{i}_{t}")

    def get_y(i, t):
        if t < ES[i]:
            return 0
        elif t >= LS[i]:
            return 1
        else:
            return y[i, t]

    # Monotonicity constraints: y[i,t] <= y[i,t+1]
    for i in act_ids:
        for t in range(ES[i], LS[i] - 1):
            model.addConstr(y[i, t] <= y[i, t+1], name=f"mono_{i}_{t}")

    # Precedence constraints
    for u, v in prec_list:
        for t in range(ES[v], LS[v]):
            model.addConstr(get_y(v, t) <= get_y(u, t - duration[u]), name=f"prec_{u}_{v}_{t}")

    # Resource setup and peak provision bounding
    R = {}
    for k in range(num_res_types):
        R[k] = model.addVar(vtype=gp.GRB.INTEGER, lb=0, name=f"R_{k}")

    # For each active period t, constraint resource bounds
    for t in range(D):
        for k in range(num_res_types):
            expr_terms = []
            const_val = 0
            for i in act_ids:
                req = resources[i][k]
                # Activity i overlaps with period t if ES_i <= t AND t - duration < LS_i
                if req > 0 and t >= ES[i] and t - duration[i] < LS[i]:
                    y1 = get_y(i, t)
                    if isinstance(y1, gp.Var):
                        expr_terms.append(req * y1)
                    else:
                        const_val += req * y1
                    
                    y2 = get_y(i, t - duration[i])
                    if isinstance(y2, gp.Var):
                        expr_terms.append(-req * y2)
                    else:
                        const_val -= req * y2
            
            if expr_terms or const_val > 0:
                model.addConstr(gp.quicksum(expr_terms) + const_val <= R[k], name=f"res_{t}_{k}")

    # Objective logic
    obj = gp.LinExpr()
    for k in range(num_res_types):
        obj.addTerms(res_costs[k], R[k])
    model.setObjective(obj, gp.GRB.MINIMIZE)

    # Provide initial valid upper bounds to solve faster (Early-Start and Late-Start Schedules)
    model.NumStart = 2

    # Start 0: ES Schedule (S_i = ES[i])
    model.params.StartNumber = 0
    for i in act_ids:
        for t in range(ES[i], LS[i]):
            y[i, t].Start = 1.0 if t >= ES[i] else 0.0

    # Start 1: LS Schedule (S_i = LS[i])
    model.params.StartNumber = 1
    for i in act_ids:
        for t in range(ES[i], LS[i]):
            y[i, t].Start = 1.0 if t >= LS[i] else 0.0

    # Optimization callback for logging incumbent solutions
    class SolutionCallback:
        def __init__(self, logger_handle):
            self.logger = logger_handle
            self.best_obj = float('inf')

        def __call__(self, m, where):
            if where == gp.GRB.Callback.MIPSOL:
                current_obj = m.cbGet(gp.GRB.Callback.MIPSOL_OBJ)
                if current_obj < self.best_obj - 1e-5:
                    self.best_obj = current_obj
                    if self.logger:
                        self.logger.log(current_obj)

    cb = SolutionCallback(logger)
    model.optimize(cb)

    # Wrap the best solution if discovered
    if model.SolCount > 0:
        start_times = {}
        for i in act_ids:
            s_i = LS[i]
            for t in range(ES[i], LS[i]):
                if y[i, t].X > 0.5:
                    s_i = t
                    break
            start_times[str(i)] = s_i
            
        resource_amounts = {}
        for k in range(num_res_types):
            resource_amounts[str(k)] = int(round(R[k].X))
            
        final_solution = {
            "objective_value": float(model.ObjVal),
            "start_times": start_times,
            "resource_amounts": resource_amounts
        }
        
        with open(args.solution_path, 'w') as f:
            json.dump(final_solution, f, indent=2)
    else:
        print("No feasible solution was found within bounds or time limit.")

if __name__ == "__main__":
    main()