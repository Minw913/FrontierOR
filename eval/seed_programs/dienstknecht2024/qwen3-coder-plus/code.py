import argparse
import json
import numpy as np
from collections import defaultdict, deque
import gurobipy as gp
from gurobipy import GRB
import time
from itertools import combinations

def parse_args():
    parser = argparse.ArgumentParser(description="Solve the construction container assignment problem.")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to the input instance file")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to write the output solution")
    parser.add_argument("--time_limit", type=int, required=True, help="Time limit in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Path for logging solutions (optional)")
    return parser.parse_args()

def log_solution(logger, objective_value):
    if logger:
        logger.log(objective_value)

def get_connected_components(assigned_containers, adjacency_dict):
    assigned_set = set(assigned_containers)
    visited = set()
    components = []
    
    for container in assigned_containers:
        if container not in visited:
            # BFS/DFS for component
            component = []
            queue = [container]
            visited.add(container)
            
            while queue:
                curr = queue.pop()
                component.append(curr)
                
                for neighbor in adjacency_dict.get(curr, []):
                    if neighbor not in visited and neighbor in assigned_set:
                        visited.add(neighbor)
                        queue.append(neighbor)
                        
            components.append(component)
    
    return components

def solve_with_greedy_heuristic(instance, time_limit, logger):
    start_time = time.time()
    
    num_periods = instance["problem_parameters"]["num_periods"]
    trades = instance["trades"]
    containers = instance["containers"]
    
    # Preprocessing
    container_adjacency = {c['id']: c['adjacent_containers'] for c in containers}
    container_availability = {c['id']: (c['availability_start'], c['availability_end']) for c in containers}
    
    total_reassignments = 0
    assignments = {}
    
    # Process trades in order of start period
    sorted_trades = sorted(trades, key=lambda t: t['start_period'])
    
    current_assignments = {}  # For tracking from previous periods
    
    for trade in sorted_trades:
        trade_id = trade['id']
        start_period = trade['start_period']
        end_period = trade['end_period']
        demand = trade['container_demand']
        max_dispersion = trade['max_dispersion']
        
        assignments[trade_id] = {}
        
        for period in range(start_period, end_period + 1):
            if time.time() - start_time > time_limit * 0.95:  # Stop if we're almost out of time
                break
                
            # Determine available containers for this period
            available_containers = []
            for container in containers:
                av_start, av_end = container_availability[container['id']]
                if av_start <= period <= av_end:
                    available_containers.append(container['id'])
                    
            # Check if this trade is already assigned in the previous period (for reassignment counting)
            need_reassignment = False
            prev_assignment = current_assignments.get((trade_id, period - 1), [])
            
            # Find containers to assign
            selected_containers = []
            
            # Try to keep as much of previous assignment as possible (reduces reassignments)
            if period > start_period:  # Not the first period for this trade
                valid_prev = [c for c in prev_assignment if c in available_containers]
                selected_containers.extend(valid_prev)
                
            # If we need more containers, add them
            while len(selected_containers) < demand:
                # Get remaining available containers not yet taken
                remaining_available = [c for c in available_containers if c not in selected_containers]
                
                # Try to add containers that will keep cluster count within limit
                best_option = None
                best_clusters = float('inf')
                
                for c in remaining_available:
                    test_assignment = selected_containers + [c]
                    clusters = get_connected_components(test_assignment, container_adjacency)
                    
                    if len(clusters) <= max_dispersion and len(test_assignment) <= demand:
                        if len(clusters) < best_clusters:
                            best_clusters = len(clusters)
                            best_option = c
                
                if best_option is not None:
                    selected_containers.append(best_option)
                else:
                    # No valid option found with current clustering, try any available container
                    if not remaining_available:
                        # This is a problematic case - break and continue with what we have
                        break
                    selected_containers.append(remaining_available[0])
                    # Recalculate clusters after adding
                    clusters = get_connected_components(selected_containers, container_adjacency)
                    if len(clusters) > max_dispersion:
                        # Still exceeds, remove the latest addition and try next option if needed
                        selected_containers.pop()
                        # For now, just accept the clustering as the best we can do for this period
                        selected_containers.extend(
                            [c for c in remaining_available if c not in selected_containers][:demand-len(selected_containers)]
                        )
                        break
            
            # Ensure we have exactly the required demand
            if len(selected_containers) < demand:
                remaining_needed = demand - len(selected_containers)
                remaining_available = [c for c in available_containers if c not in selected_containers]
                selected_containers.extend(remaining_available[:remaining_needed])
                
            selected_containers = selected_containers[:demand]
            
            # Check if this creates a reassignment compared to previous period
            if period > start_period:
                prev_selected = current_assignments.get((trade_id, period - 1), [])
                if set(prev_selected) != set(selected_containers):
                    total_reassignments += 1
                    need_reassignment = True
                    
            assignments[trade_id][period] = selected_containers[:]
            current_assignments[(trade_id, period)] = selected_containers[:]
    
    return total_reassignments, assignments

def solve_with_mip(instance, time_limit, logger):
    start_time = time.time()
    
    num_periods = instance["problem_parameters"]["num_periods"]
    num_trades = instance["problem_parameters"]["num_trades"]
    num_containers = instance["problem_parameters"]["num_containers"]
    trades = instance["trades"]
    containers = instance["containers"]
    
    # Create mapping from container IDs to indices starting from 0
    container_id_to_idx = {c['id']: idx for idx, c in enumerate(containers)}
    idx_to_container_id = {idx: c['id'] for idx, c in enumerate(containers)}
    
    # Prepare data structures
    container_adjacency_list = [[] for _ in range(num_containers)]
    for c in containers:
        c_idx = container_id_to_idx[c['id']]
        adj_indices = [container_id_to_idx[adj_id] for adj_id in c['adjacent_containers']]
        container_adjacency_list[c_idx] = adj_indices
    
    container_start = [0] * num_containers
    container_end = [num_periods - 1] * num_containers
    for c in containers:
        c_idx = container_id_to_idx[c['id']]
        container_start[c_idx] = c['availability_start']
        container_end[c_idx] = c['availability_end']
    
    # Trade data with adjusted indices
    trade_data = {}
    for t in trades:
        trade_data[t['id']] = t
    
    trade_ids = [t['id'] for t in trades]
    trade_idx_map = {tid: i for i, tid in enumerate(trade_ids)}
    
    try:
        model = gp.Model("ContainerAssignment")
        model.setParam("Threads", 1)
        model.setParam(GRB.Param.TimeLimit, time_limit)
        model.setParam(GRB.Param.Heuristics, 0.01)  # Low heuristics usage to spend more time on branch & cut
        
        # Decision variables: x[trade_id][container_idx][period] = 1 if trade_id uses container_idx at period
        x = {}
        for t in trades:
            t_id = t['id']
            start_p = t['start_period']
            end_p = t['end_period']
            demand = t['container_demand']
            
            x[t_id] = {}
            for c_idx in range(num_containers):
                cont_id = idx_to_container_id[c_idx]
                av_start = container_start[c_idx]
                av_end = container_end[c_idx]
                
                x[t_id][c_idx] = {}
                for p in range(start_p, end_p + 1):
                    if av_start <= p <= av_end:
                        x[t_id][c_idx][p] = model.addVar(vtype=GRB.BINARY, name=f"x_{t_id}_{cont_id}_{p}")
                    else:
                        x[t_id][c_idx][p] = 0  # This container is not available in this period
    
        # Assignment change variable: z[trade_id][period] = 1 if assignment changes in this period
        z = {}
        for t in trades:
            t_id = t['id']
            start_p = t['start_period']
            end_p = t['end_period']
            
            z[t_id] = {}
            for p in range(start_p + 1, end_p + 1):  # From 2nd period onward
                z[t_id][p] = model.addVar(vtype=GRB.BINARY, name=f"z_{t_id}_{p}")
        
        model.update()
        
        # Constraint 1: Each trade gets exactly their demand of containers in each required period
        for t in trades:
            t_id = t['id']
            start_p = t['start_period']
            end_p = t['end_period']
            demand = t['container_demand']
            
            for p in range(start_p, end_p + 1):
                model.addConstr(
                    gp.quicksum(x[t_id][c_idx][p] for c_idx in range(num_containers) 
                               if isinstance(x[t_id][c_idx][p], gp.Var)) == demand,
                    name=f"demand_{t_id}_{p}"
                )
        
        # Constraint 2: Each container can be assigned to at most one trade per period
        for c_idx in range(num_containers):
            av_start = container_start[c_idx]
            av_end = container_end[c_idx]
            
            for p in range(av_start, av_end + 1):
                model.addConstr(
                    gp.quicksum(x[t_id][c_idx][p] for t_id in trade_idx_map.keys()
                               if p in x[t_id][c_idx] and isinstance(x[t_id][c_idx][p], gp.Var)) <= 1,
                    name=f"capacity_{idx_to_container_id[c_idx]}_{p}"
                )
        
        # Constraint 3: Link reassignment variable to actual changes in assignment
        for t in trades:
            t_id = t['id']
            start_p = t['start_period']
            end_p = t['end_period']
            
            for p in range(start_p + 1, end_p + 1):
                # We need to use big-M constraints to link z to changes in x
                # z[t_id][p] >= sum of differences from previous period
                changed_from_prev = []
                removed_in_current = []
                added_in_current = []
                
                # Get all containers that were assigned in previous period
                for c_idx in range(num_containers):
                    if p-1 in x[t_id][c_idx] and isinstance(x[t_id][c_idx][p-1], gp.Var):
                        # Container assigned in previous period but not in current
                        model.addConstr(z[t_id][p] >= x[t_id][c_idx][p-1] - x[t_id][c_idx][p],
                                       name=f"change_down_{t_id}_{idx_to_container_id[c_idx]}_{p}")
                
                for c_idx in range(num_containers):
                    if p in x[t_id][c_idx] and isinstance(x[t_id][c_idx][p], gp.Var):
                        # Container assigned in current period but not in previous  
                        if p-1 in x[t_id][c_idx] and isinstance(x[t_id][c_idx][p-1], gp.Var):
                            model.addConstr(z[t_id][p] >= x[t_id][c_idx][p] - x[t_id][c_idx][p-1],
                                           name=f"change_up_{t_id}_{idx_to_container_id[c_idx]}_{p}")
                        elif p-1 in x[t_id][c_idx]:  # Fixed value (not variable) 
                            model.addConstr(z[t_id][p] >= x[t_id][c_idx][p] - x[t_id][c_idx][p-1],
                                           name=f"change_up_fix_{t_id}_{idx_to_container_id[c_idx]}_{p}")
        
        # Constraint 6: Ensure maximum dispersion per trade per period
        # This is tricky - linearizing the connected components constraint
        # Use flow-based formulation for connectivity
        y = {}  # y[t,c,c',p] = flow of virtual commodity from container c
        r = {}  # r[t,c,p] = indicator that container c is root of a cluster for trade t at period p
        
        for t in trades:
            t_id = t['id']
            start_p = t['start_period']
            end_p = t['end_period']
            max_disp = t['max_dispersion']
            
            for p in range(start_p, end_p + 1):
                # Create binary vars
                r[t_id, p] = {}
                assigned_containers = [c_idx for c_idx in range(num_containers) 
                                      if p in x[t_id][c_idx] and isinstance(x[t_id][c_idx][p], gp.Var)]
                
                for c_idx in assigned_containers:
                    r[t_id, p][c_idx] = model.addVar(vtype=GRB.BINARY, name=f"r_{t_id}_{idx_to_container_id[c_idx]}_{p}")
                
                # Now add flows
                y[t_id, p] = {}
                for i in assigned_containers:
                    y[t_id, p][i] = {}
                    for j in container_adjacency_list[i]:
                        if j in assigned_containers:
                            y[t_id, p][i][j] = model.addVar(lb=0, ub=1, name=f"flow_{t_id}_{p}_{i}_{j}")
        
        # Update model to add vars
        model.update()
        
        # Add connectivity constraints
        for t in trades:
            t_id = t['id']
            start_p = t['start_period']
            end_p = t['end_period']
            max_disp = t['max_dispersion']
            
            for p in range(start_p, end_p + 1):
                assigned_containers = [c_idx for c_idx in range(num_containers) 
                                      if p in x[t_id][c_idx] and isinstance(x[t_id][c_idx][p], gp.Var)]
                
                # Root selection: each assigned container must either be root or receive flow
                for i in assigned_containers:
                    flow_out = gp.quicksum(y[t_id, p].get(i, {}).get(j, 0) for j in container_adjacency_list[i] if j in assigned_containers)
                    inflow = gp.quicksum(y[t_id, p].get(j, {}).get(i, 0) for j in container_adjacency_list[i] if j in assigned_containers)
                    
                    # If container is assigned, it must either be a root or have positive net inflow
                    model.addConstr(
                        x[t_id][i][p] <= (r[t_id, p][i] + inflow - flow_out),
                        name=f"netflow_{t_id}_{idx_to_container_id[i]}_{p}"
                    )
    
                # At most max_disp roots per trade per period
                model.addConstr(
                    gp.quicksum(r[t_id, p][i] for i in assigned_containers) <= max_disp,
                    name=f"dispersion_{t_id}_{p}"
                )
                
        # Set objective function: minimize total re-assignments
        total_changes = gp.quicksum(z[t['id']][p] for t in trades 
                                  for p in range(t['start_period'] + 1, t['end_period'] + 1))
        model.setObjective(total_changes, GRB.MINIMIZE)
        
        # Callback for logging improvement
        def callback(model, where):
            if where == GRB.Callback.MIPSOL:
                obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
                log_solution(logger, obj_val)
        
        model.optimize(callback)
        
        # Extract solution if found
        if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
            assignments = {}
            
            for t in trades:
                t_id = t['id']
                start_p = t['start_period']
                end_p = t['end_period']
                
                assignments[t_id] = {}
                for p in range(start_p, end_p + 1):
                    assigned_containers = []
                    for c_idx in range(num_containers):
                        if isinstance(x[t_id][c_idx][p], gp.Var) and x[t_id][c_idx][p].X > 0.5:
                            assigned_containers.append(idx_to_container_id[c_idx])
                    assignments[t_id][str(p)] = assigned_containers
            
            objective_value = int(model.objVal) if model.SolCount > 0 else float('inf')
            
            return objective_value, assignments
        else:
            raise Exception(f"MIP solver did not find a suitable solution. Status: {model.Status}")
            
    except Exception as e:
        print(f"MIP approach failed: {e}. Trying greedy heuristic instead.")
        return solve_with_greedy_heuristic(instance, time_limit, logger)


def main():
    args = parse_args()
    
    # Import solution logger
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize")
    else:
        logger = None
    
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    # Start with greedy solution as fallback
    obj_val, assignments = solve_with_greedy_heuristic(instance, args.time_limit, logger)
    
    # Try MIP if time permits (MIP can take longer, but might give better results)
    try:
        mip_obj_val, mip_assignments = solve_with_mip(instance, args.time_limit, logger)
        
        # Pick the better solution
        if mip_obj_val < obj_val:
            obj_val = mip_obj_val
            assignments = mip_assignments
            
    except Exception as e:
        print(f"MIP solving failed gracefully: {e}")
        # Continue with the greedy solution
    
    solution = {
        "objective_value": obj_val,
        "assignments": assignments
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)
    
    print(f"Solution written to {args.solution_path} with objective value: {obj_val}")

if __name__ == "__main__":
    main()