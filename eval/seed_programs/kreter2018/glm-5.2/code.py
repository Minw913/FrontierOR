import argparse
import json
import sys
from collections import defaultdict, deque
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def solve_instance(instance, time_limit, log_path):
    # parse instance
    num_real = instance["num_real_activities"]
    num_total = instance["num_total_activities"]
    num_res = instance["num_resource_types"]
    D = instance["max_project_duration"]
    activities = instance["activities"]
    precedences = instance["precedence_relations"]
    costs = instance["resource_costs"]
    
    # map id to activity
    act_by_id = {a["id"]: a for a in activities}
    ids = [a["id"] for a in activities]
    durations = {a["id"]: a["duration"] for a in activities}
    reqs = {a["id"]: a["resource_requirements"] for a in activities}
    
    # build adjacency
    successors = defaultdict(list)
    predecessors = defaultdict(list)
    for i, j in precedences:
        successors[i].append(j)
        predecessors[j].append(i)
    
    # topological sort
    # compute ES
    ES = {id: 0 for id in ids}
    # Kahn's algorithm
    in_degree = {id: 0 for id in ids}
    for id in ids:
        in_degree[id] = len(predecessors[id])
    queue = deque([id for id in ids if in_degree[id] == 0])
    topo = []
    while queue:
        u = queue.popleft()
        topo.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    # forward pass
    for u in topo:
        for v in successors[u]:
            if ES[v] < ES[u] + durations[u]:
                ES[v] = ES[u] + durations[u]
    
    # compute LS
    LS = {id: D for id in ids}
    # backward pass in reverse topological order
    for u in reversed(topo):
        for v in predecessors[u]:
            if LS[v] > LS[u] - durations[u]:
                LS[v] = LS[u] - durations[u]
    
    # check feasibility
    for id in ids:
        if ES[id] > LS[id]:
            print(f"Infeasible: activity {id} ES={ES[id]} > LS={LS[id]}", file=sys.stderr)
            return None
    
    # create Gurobi model
    env = gp.Env(empty=True)
    env.setParam('Seed', 0)
    env.setParam('Threads', 1)
    env.setParam('NumericFocus', 0)
    env.setParam('OutputFlag', 0)  # quiet
    env.start()
    model = gp.Model(env=env)
    
    # variables
    x = {}
    for id in ids:
        for t in range(ES[id], LS[id] + 1):
            x[id, t] = model.addVar(vtype=GRB.BINARY, name=f"x_{id}_{t}")
    
    amount = {}
    for k in range(num_res):
        amount[k] = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"amount_{k}")
    
    model.update()
    
    # constraints
    # each activity starts once
    for id in ids:
        model.addConstr(gp.quicksum(x[id, t] for t in range(ES[id], LS[id] + 1)) == 1)
    
    # precedence
    for i, j in precedences:
        start_i = gp.quicksum(t * x[i, t] for t in range(ES[i], LS[i] + 1))
        start_j = gp.quicksum(t * x[j, t] for t in range(ES[j], LS[j] + 1))
        model.addConstr(start_j >= start_i + durations[i])
    
    # resource constraints
    # precompute coverage
    # for each activity i and period p, the set of t that cover p
    # t in [max(ES_i, p - d_i + 1), min(LS_i, p)]
    for k in range(num_res):
        for p in range(D):
            terms = []
            for id in ids:
                d = durations[id]
                r = reqs[id][k]
                if r == 0:
                    continue
                t_min = max(ES[id], p - d + 1)
                t_max = min(LS[id], p)
                if t_min > t_max:
                    continue
                # sum of x[id,t] for t in [t_min, t_max]
                # we can use quicksum
                terms.append(r * gp.quicksum(x[id, t] for t in range(t_min, t_max + 1)))
            if terms:
                model.addConstr(gp.quicksum(terms) <= amount[k])
    
    # objective
    model.setObjective(gp.quicksum(costs[k] * amount[k] for k in range(num_res)), GRB.MINIMIZE)
    
    # set params
    model.setParam('MIPGap', 1e-4)
    model.setParam('TimeLimit', time_limit)
    # Seed, Threads, NumericFocus already set in env
    
    # callback or just optimize
    # we need to log solutions. Gurobi doesn't easily give a callback that can call our logger with full solution? We can use a callback.
    # But we can also just run optimize and then log the final solution. The requirement says "Whenever a better feasible solution is found". So we should use a callback to log incumbents.
    
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    def callback(model, where):
        if where == GRB.Callback.MIP:
            objbst = model.cbGet(GRB.Callback.MIP_OBJBST)
            objbnd = model.cbGet(GRB.Callback.MIP_OBJBND)
            # we only log if we have a new incumbent? The callback doesn't directly give the solution unless we use MIPSOL.
            pass
        elif where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            # extract solution
            x_val = model.cbGetSolution(x)
            amount_val = model.cbGetSolution(amount)
            start_times = {}
            for id in ids:
                for t in range(ES[id], LS[id] + 1):
                    if x_val[x[id, t]] > 0.5:
                        start_times[id] = t
                        break
            resource_amounts = {k: amount_val[amount[k]] for k in range(num_res)}
            sol_dict = {
                "objective_value": obj,
                "start_times": {str(id): start_times[id] for id in ids},
                "resource_amounts": {str(k): resource_amounts[k] for k in range(num_res)}
            }
            if logger:
                logger.log_solution(obj, sol_dict)
    
    model.optimize(callback)
    
    if model.SolCount == 0:
        print("No feasible solution found", file=sys.stderr)
        return None
    
    # extract final solution
    obj = model.ObjVal
    start_times = {}
    for id in ids:
        for t in range(ES[id], LS[id] + 1):
            if x[id, t].X > 0.5:
                start_times[id] = t
                break
    resource_amounts = {k: amount[k].X for k in range(num_res)}
    
    sol = {
        "objective_value": obj,
        "start_times": {str(id): start_times[id] for id in ids},
        "resource_amounts": {str(k): resource_amounts[k] for k in range(num_res)}
    }
    return sol

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()
    
    with open(args.instance_path, "r") as f:
        instance = json.load(f)
    
    sol = solve_instance(instance, args.time_limit, args.log_path)
    if sol is None:
        sys.exit(1)
    
    with open(args.solution_path, "w") as f:
        json.dump(sol, f, indent=2)

if __name__ == "__main__":
    main()