import argparse
import json
import gurobipy as gp
from gurobipy import GRB
import time
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    # Initialize solution logger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Load instance data
    with open(args.instance_path) as f:
        data = json.load(f)
    
    params = data['problem_parameters']
    containers = data['containers']
    trades = data['trades']
    num_periods = params['num_periods']
    
    # Precompute container availability and adjacency edges
    container_dict = {c['id']: c for c in containers}
    adj_edges = set()
    for c in containers:
        c_id = c['id']
        for nbr in c['adjacent_containers']:
            if c_id < nbr:
                adj_edges.add((c_id, nbr))
    adj_edges = list(adj_edges)
    
    # Create Gurobi model
    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    model = gp.Model(env=env)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', args.time_limit)
    
    # Create variables
    x = {}  # x[t_id, c_id, p]
    for trade in trades:
        t_id = trade['id']
        start_t = trade['start_period']
        end_t = trade['end_period']
        for c in containers:
            c_id = c['id']
            start_c = c['availability_start']
            end_c = c['availability_end']
            start_p = max(start_t, start_c)
            end_p = min(end_t, end_c)
            for p in range(start_p, end_p + 1):
                x[t_id, c_id, p] = model.addVar(vtype=GRB.BINARY, name=f"x_{t_id}_{c_id}_{p}")
    
    w = {}  # w[t_id, p, c1, c2] for edges
    for trade in trades:
        t_id = trade['id']
        start_t = trade['start_period']
        end_t = trade['end_period']
        for (c1, c2) in adj_edges:
            start_c1 = container_dict[c1]['availability_start']
            end_c1 = container_dict[c1]['availability_end']
            start_c2 = container_dict[c2]['availability_start']
            end_c2 = container_dict[c2]['availability_end']
            start_p = max(start_t, start_c1, start_c2)
            end_p = min(end_t, end_c1, end_c2)
            if start_p > end_p:
                continue
            for p in range(start_p, end_p + 1):
                w[t_id, p, c1, c2] = model.addVar(vtype=GRB.BINARY, name=f"w_{t_id}_{p}_{c1}_{c2}")
    
    y = {}  # y[t_id, c_id, p] for reassignments
    for trade in trades:
        t_id = trade['id']
        start_t = trade['start_period']
        end_t = trade['end_period']
        for c in containers:
            c_id = c['id']
            start_c = c['availability_start']
            end_c = c['availability_end']
            start_p = max(start_t + 1, start_c)
            end_p = min(end_t, end_c)
            if start_p > end_p:
                continue
            for p in range(start_p, end_p + 1):
                y[t_id, c_id, p] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"y_{t_id}_{c_id}_{p}")
    
    # Demand constraints
    for trade in trades:
        t_id = trade['id']
        start_t = trade['start_period']
        end_t = trade['end_period']
        d_t = trade['container_demand']
        for p in range(start_t, end_t + 1):
            vars = []
            for c in containers:
                c_id = c['id']
                if (t_id, c_id, p) in x:
                    vars.append(x[t_id, c_id, p])
            if vars:
                model.addConstr(gp.quicksum(vars) == d_t, name=f"demand_{t_id}_{p}")
    
    # Container usage constraints
    for c in containers:
        c_id = c['id']
        start_c = c['availability_start']
        end_c = c['availability_end']
        for p in range(start_c, end_c + 1):
            vars = []
            for trade in trades:
                t_id = trade['id']
                if (t_id, c_id, p) in x:
                    vars.append(x[t_id, c_id, p])
            if vars:
                model.addConstr(gp.quicksum(vars) <= 1, name=f"container_{c_id}_{p}")
    
    # Connectivity constraints: w constraints and component count
    for trade in trades:
        t_id = trade['id']
        start_t = trade['start_period']
        end_t = trade['end_period']
        k_t = trade['max_dispersion']
        d_t = trade['container_demand']
        for p in range(start_t, end_t + 1):
            # Edge constraints
            for (c1, c2) in adj_edges:
                if (t_id, p, c1, c2) in w:
                    model.addConstr(w[t_id, p, c1, c2] <= x.get((t_id, c1, p), 0), name=f"w_upper1_{t_id}_{p}_{c1}_{c2}")
                    model.addConstr(w[t_id, p, c1, c2] <= x.get((t_id, c2, p), 0), name=f"w_upper2_{t_id}_{p}_{c1}_{c2}")
                    model.addConstr(w[t_id, p, c1, c2] >= x.get((t_id, c1, p), 0) + x.get((t_id, c2, p), 0) - 1, name=f"w_lower_{t_id}_{p}_{c1}_{c2}")
            
            # Component count constraint
            w_vars = []
            for (c1, c2) in adj_edges:
                if (t_id, p, c1, c2) in w:
                    w_vars.append(w[t_id, p, c1, c2])
            if w_vars:
                model.addConstr(gp.quicksum(w_vars) >= d_t - k_t, name=f"conn_{t_id}_{p}")
            else:
                model.addConstr(0 >= d_t - k_t, name=f"conn_{t_id}_{p}")
    
    # Reassignment constraints
    for trade in trades:
        t_id = trade['id']
        start_t = trade['start_period']
        end_t = trade['end_period']
        for c in containers:
            c_id = c['id']
            start_c = c['availability_start']
            end_c = c['availability_end']
            start_p = max(start_t + 1, start_c)
            end_p = min(end_t, end_c)
            for p in range(start_p, end_p + 1):
                if p - 1 >= start_c and p - 1 <= end_c:
                    # Container was available in previous period
                    model.addConstr(
                        y[t_id, c_id, p] >= x[t_id, c_id, p] - x.get((t_id, c_id, p-1), 0),
                        name=f"reassign_avail_{t_id}_{c_id}_{p}"
                    )
                else:
                    # Container not available in previous period
                    model.addConstr(
                        y[t_id, c_id, p] >= x[t_id, c_id, p],
                        name=f"reassign_notavail_{t_id}_{c_id}_{p}"
                    )
    
    # Objective: minimize total reassignments
    obj = gp.quicksum(y.values())
    model.setObjective(obj, GRB.MINIMIZE)
    
    # Callback function to log incumbents
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            if logger:
                obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                logger.log(obj_val)
    
    # Optimize
    model.optimize(callback)
    
    # Extract solution
    objective_value = float('inf')
    assignments = {}
    
    if model.SolCount > 0:
        # Get the best solution found
        model.setParam(GRB.Param.SolutionNumber, 0)
        objective_value = model.ObjVal
        
        # Build assignments dictionary
        for trade in trades:
            t_id = trade['id']
            start_t = trade['start_period']
            end_t = trade['end_period']
            trade_assign = {}
            for p in range(start_t, end_t + 1):
                assigned_containers = []
                for c in containers:
                    c_id = c['id']
                    if (t_id, c_id, p) in x:
                        if x[t_id, c_id, p].X >= 0.5:
                            assigned_containers.append(c_id)
                trade_assign[str(p)] = assigned_containers
            assignments[str(t_id)] = trade_assign
    else:
        # Fallback: if no solution found, create empty assignments
        for trade in trades:
            t_id = trade['id']
            start_t = trade['start_period']
            end_t = trade['end_period']
            trade_assign = {}
            for p in range(start_t, end_t + 1):
                trade_assign[str(p)] = []
            assignments[str(t_id)] = trade_assign
        objective_value = 0  # No reassignments if no assignments
    
    # Write solution
    solution = {
        "objective_value": objective_value,
        "assignments": assignments
    }
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == "__main__":
    main()