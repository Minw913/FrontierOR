import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        return json.load(f)


def build_network_graph(data):
    """
    Builds a directed graph representation of the problem.
    The graph includes depot nodes and task nodes.
    """
    tasks = {t['id']: t for t in data['tasks']}
    depots = {d['id']: d for d in data['depots']}

    # Create arc-based model structure
    # Each depot will have its own subgraph
    graph = {}
    
    # Add depot-task arcs (outgoing from depots)
    for arc in data['depot_to_task_costs']:
        depot_id, task_id = arc['depot_id'], arc['task_id']
        if depot_id not in graph:
            graph[depot_id] = {}
        if 'depot_nodes' not in graph[depot_id]:
            graph[depot_id]['depot_nodes'] = []
        if 'task_nodes' not in graph[depot_id]:
            graph[depot_id]['task_nodes'] = []
        if 'arc_data' not in graph[depot_id]:
            graph[depot_id]['arc_data'] = {}

        
        task_key = ('task', task_id)
        graph[depot_id]['task_nodes'].append(task_key)
        graph[depot_id]['arc_data'][(depot_id, task_key)] = {
            'cost': arc['cost'],
            'travel_time': arc['travel_time']
        }

    # Add task-task arcs (compatible pairs)
    for pair in data['compatible_pairs']:
        task_i, task_j = pair['task_i'], pair['task_j']
        task_i_key = ('task', task_i)
        task_j_key = ('task', task_j)
        
        # Connect pairs if they belong to the same subgraph
        for depot_id in data['depots']:
            if depot_id in graph and task_i_key in graph[depot_id]['task_nodes']:
                if task_j_key not in graph[depot_id]['task_nodes']:
                    graph[depot_id]['task_nodes'].append(task_j_key)
                
                if 'depot_nodes' not in graph[depot_id]:
                    graph[depot_id]['depot_nodes'] = [depot_id]
                
                graph[depot_id]['arc_data'][(task_i_key, task_j_key)] = {
                    'cost': pair['cost'],
                    'travel_time': pair['travel_time']
                }

    # Add task-depot arcs (returning to depots)
    for arc in data['task_to_depot_costs']:
        task_id, depot_id = arc['task_id'], arc['depot_id']
        task_key = ('task', task_id)
        
        if depot_id not in graph:
            graph[depot_id] = {'depot_nodes': [depot_id], 'task_nodes': [], 'arc_data': {}}
        if task_key not in graph[depot_id]['task_nodes']:
            graph[depot_id]['task_nodes'].append(task_key)
            
        graph[depot_id]['arc_data'][(task_key, depot_id)] = {
            'cost': arc['cost'],
            'travel_time': arc['travel_time']
        }

    return graph, tasks, depots


def solve_transit_assignment(instance_path, time_limit, log_path=None):
    instance_data = read_instance(instance_path)
    graph, tasks, depots = build_network_graph(instance_data)

    # Create model
    m = gp.Model("transit_assignment")
    m.setParam("Threads", 1)
    m.setParam(GRB.Param.TimeLimit, time_limit)
    m.setParam(GRB.Param.OutputFlag, 0)
    
    if log_path:
        # Set callback function for logging intermediate solutions
        def sol_log(model, where):
            if where == GRB.Callback.MIPSOL:
                obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if logger:
                    logger.log(obj_val)
        logger = SolutionLogger(log_path, sense="minimize")
        m.optimize(sol_log)
    else:
        logger = None
        m.optimize()

    # Check status
    if m.status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        # Extract solution
        selected_arcs = []
        for depot_id in depots:
            if depot_id not in graph:
                continue
            
            # Get all arc variables for this depot
            depot_arcs = {}
            for (i, j), data in graph[depot_id]['arc_data'].items():
                var_name = f"flow_{depot_id}_{i}_{j}"
                if var_name in m.getAttr('VarName'):
                    var = m.getVarByName(var_name)
                    if var.x > 1e-6:  # Only consider positive flows
                        depot_arcs[(i, j)] = data
            
            # Extract routes by traversing flow paths
            depot_routes = extract_solution_from_flow(m, graph[depot_id], depot_id)
            selected_arcs.extend(depot_routes)
        
        # Calculate objective value
        obj_value = m.ObjVal if m.SolCount > 0 else float('inf')
        
        # Build final routes from selected arcs
        routes = []
        for route_info in selected_arcs:
            depot_id = route_info['depot']
            task_sequence = route_info['tasks']
            route_cost = route_info['cost']
            routes.append({
                'depot': depot_id,
                'tasks': task_sequence,
                'cost': route_cost
            })
        
        solution = {
            "objective_value": obj_value,
            "routes": routes
        }
        
        return solution, logger

    return {"objective_value": float('inf'), "routes": []}, logger


def solve_with_gurobi(instance_data):
    """Solve the instance using a multi-commodity flow network model."""
    num_tasks = instance_data['num_tasks']
    num_depots = instance_data['num_depots']

    tasks = {t['id']: t for t in instance_data['tasks']}
    depots = {d['id']: d for d in instance_data['depots']}

    # Create a directed graph with nodes for tasks and each depot (start & end)
    # Node IDs: 0..num_tasks-1 are task ids, num_tasks and beyond are depot start/end nodes
    
    # Maps depot and task ids to node indices in the flow network
    task_map_to_idx = {t['id']: i for i, t in enumerate(instance_data['tasks'])}
    task_idx_to_task_id = {i: t['id'] for i, t in enumerate(instance_data['tasks'])}
    
    # Additional nodes for depot entry/exit
    depot_entry_node_start = num_tasks
    depot_exit_node_end = num_tasks + num_depots - 1

    try:
        m = gp.Model()
        m.setParam("Threads", 1)
        m.setAttr(GRB.Param.OutputFlag, 0)
        m.setAttr(GRB.Param.PoolSolutions, 1)
        
        # Decision variables: flow on each arc
        x = {}  # x[(start_node, end_node, depot_id)] = flow

        # Arcs definition
        all_arcs = []

        # Depot to task arcs
        for i, dtc in enumerate(instance_data['depot_to_task_costs']):
            depot_id, task_id = dtc['depot_id'], dtc['task_id']
            task_node = task_map_to_idx[task_id]
            depot_out_node = num_tasks + depots[depot_id]['id'] - min(depots.keys())  # Map depot id to a range starting from num_tasks
            all_arcs.append((depot_out_node, task_node, depot_id))

        # Task to task compatible arcs
        for cp in instance_data['compatible_pairs']:
            task_i, task_j = cp['task_i'], cp['task_j']
            node_i, node_j = task_map_to_idx[task_i], task_map_to_idx[task_j]
            # This arc can potentially be used by any depot serving both tasks eventually connected
            for depot_id in depots.keys():
                all_arcs.append((node_i, node_j, depot_id))

        # Task to depot arcs
        for i, tdc in enumerate(instance_data['task_to_depot_costs']):
            task_id, depot_id = tdc['task_id'], tdc['depot_id']
            task_node = task_map_to_idx[task_id]
            depot_in_node = num_tasks + depots[depot_id]['id'] - min(depots.keys())
            all_arcs.append((task_node, depot_in_node, depot_id))

        # Create binary flow variables
        for (i, j, depot) in all_arcs:
            x[i, j, depot] = m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}_{depot}")

        # Constraints
        # 1. Each task is covered by exactly one vehicle
        for task_idx in range(num_tasks):
            m.addConstr(
                gp.quicksum(x[i, j, depot] for (i, j, depot) in all_arcs 
                           if j == task_idx or (i == task_idx and (i, j, depot)[2] >= 0)) >= 1,
                f"visit_task_{task_idx}"
            )

        # Reconstruct constraints properly
        m.remove(m.getConstrs())

        # Variable dictionary: (start_loc, end_loc, depot_id) -> var
        y = {}
        
        # Process depot-to-task connections
        depot_task_pairs = {}
        for dtc in instance_data['depot_to_task_costs']:
            depot_id = dtc['depot_id']
            task_id = dtc['task_id']
            if depot_id not in depot_task_pairs:
                depot_task_pairs[depot_id] = []
            depot_task_pairs[depot_id].append(task_id)
            var_name = f"dtc_{depot_id}_{task_id}"
            y[(depot_id, task_id, 'out')] = m.addVar(vtype=GRB.BINARY, obj=dtc['cost'], name=var_name)

        # Process task-to-task connections
        tt_pairs = {}
        for cp in instance_data['compatible_pairs']:
            task_i = cp['task_i']
            task_j = cp['task_j']
            var_name = f"tt_{task_i}_{task_j}"
            y[(task_i, task_j, 'tt')] = m.addVar(vtype=GRB.BINARY, obj=cp['cost'], name=var_name)
            if task_i not in tt_pairs:
                tt_pairs[task_i] = []
            tt_pairs[task_i].append(task_j)

        # Process task-to-depot connections
        task_depot_pairs = {}
        for tdc in instance_data['task_to_depot_costs']:
            task_id = tdc['task_id']
            depot_id = tdc['depot_id']
            var_name = f"tdc_{task_id}_{depot_id}"
            y[(task_id, depot_id, 'in')] = m.addVar(vtype=GRB.BINARY, obj=tdc['cost'], name=var_name)
            if task_id not in task_depot_pairs:
                task_depot_pairs[task_id] = []
            task_depot_pairs[task_id].append(depot_id)

        # Flow conservation per depot
        # Source (depot) constraints: out_flow - in_flow = 0 except for net outflow
        for depot_id, info in depots.items():
            # Each vehicle leaves from depot once
            total_vehicles_from_depot = m.addVar(vtype=GRB.INTEGER, ub=info['capacity'], name=f"total_vehicles_d{depot_id}")
            
            # Flow balance for depot
            outflow = gp.quicksum(y[(depot_id, task_id, 'out')] for task_id in depot_task_pairs.get(depot_id, []))
            inflow = gp.quicksum(y[(task_id, depot_id, 'in')] for task_id in task_depot_pairs.get(depot_id, []))
            m.addConstr(outflow == total_vehicles_from_depot, f"depot_out_{depot_id}")  # Vehicles leaving depot
            m.addConstr(inflow == total_vehicles_from_depot, f"depot_in_{depot_id}")     # Vehicles returning to depot

        # Flow conservation per task
        for task_id, _ in tasks.items():
            inflow = gp.quicksum(y[(prev_task, task_id, 'tt')] for prev_task, next_tasks in tt_pairs.items() if task_id in next_tasks)
            inflow += gp.quicksum(y[(depot_id, task_id, 'out')] for depot_id, task_ids in depot_task_pairs.items() if task_id in task_ids)
            
            outflow = gp.quicksum(y[(task_id, next_task, 'tt')] for next_task in tt_pairs.get(task_id, []))
            outflow += gp.quicksum(y[(task_id, depot_id, 'in')] for depot_id in task_depot_pairs.get(task_id, []))
            
            m.addConstr(inflow == 1, f"task_inflow_{task_id}")   # Each task has exactly one precursor
            m.addConstr(outflow == 1, f"task_outflow_{task_id}") # Each task has exactly one successor

        # Optimize
        m.optimize()
        
        if m.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT] and m.SolCount > 0:
            # Extract routes
            routes = []
            for depot_id in depots.keys():
                # Find starting arcs for this depot
                for task_id in depot_task_pairs.get(depot_id, []):
                    if y[(depot_id, task_id, 'out')].X > 0.9:
                        # Start building route from this task
                        current_task = task_id
                        route = [current_task]
                        
                        while True:
                            # Find next task in the sequence
                            next_found = False
                            for next_task_id in tt_pairs.get(current_task, []):
                                if y[(current_task, next_task_id, 'tt')].X > 0.9:
                                    route.append(next_task_id)
                                    current_task = next_task_id
                                    next_found = True
                                    break
                            
                            if not next_found:
                                # Find return to depot
                                for return_depot_id in task_depot_pairs.get(current_task, []):
                                    if y[(current_task, return_depot_id, 'in')].X > 0.9:
                                        # We've completed the route, calculate cost
                                        route_obj = 0
                                        # Add depot out cost
                                        for dtc in instance_data['depot_to_task_costs']:
                                            if dtc['depot_id'] == depot_id and dtc['task_id'] == route[0]:
                                                route_obj += dtc['cost']
                                                break
                                        # Add task to task costs
                                        for i in range(len(route)-1):
                                            for cp in instance_data['compatible_pairs']:
                                                if cp['task_i'] == route[i] and cp['task_j'] == route[i+1]:
                                                    route_obj += cp['cost']
                                                    break
                                        # Add task to depot cost
                                        for tdc in instance_data['task_to_depot_costs']:
                                            if tdc['task_id'] == route[-1] and tdc['depot_id'] == return_depot_id:
                                                route_obj += tdc['cost']
                                                break
                                        
                                        routes.append({
                                            'depot': depot_id,
                                            'tasks': route[:],
                                            'cost': route_obj
                                        })
                                        break
                                break

            return {"objective_value": m.ObjVal, "routes": routes}
        else:
            return {"objective_value": float('inf'), "routes": []}
            
    except Exception as e:
        print(f"Error in optimization: {e}")
        return {"objective_value": float('inf'), "routes": []}


def extract_solution_from_flow(model, depot_graph, depot_id):
    """
    Extract actual vehicle routes from flow values
    """
    routes = []
    
    arc_vars = []
    # Collect all variables in the model related to this depot
    vars_dict = {v.VarName: v.X for v in model.getVars()}
    
    # Find arcs that are used (x > 0.5 typically means used in integer solution)
    for (i, j), data in depot_graph['arc_data'].items():
        var_name = f"flow_{depot_id}_{i}_{j}"
        if var_name in vars_dict and vars_dict[var_name] > 0.5:
            arc_vars.append((i, j, data))
    
    # Group into routes
    # First find starting points (arcs from depot to task)
    for src, dest, arc_data in arc_vars:
        if isinstance(src, int) and str(src) == str(depot_id):  # Arc from depot
            route = {'depot': depot_id, 'tasks': [], 'cost': 0.0}
            current_node = dest
            route['tasks'].append(current_node[1])  # Extract task id from tuple
            route['cost'] += arc_data['cost']
            
            # Follow the trail until return to depot
            while True:
                found_next = False
                for src_iter, dest_iter, arc_data_iter in arc_vars:
                    if src_iter == current_node and src_iter != depot_id:  # From current task
                        if isinstance(dest_iter, int) and str(dest_iter) == str(depot_id):  # To depot
                            route['cost'] += arc_data_iter['cost']
                            # Completed route - add it
                            routes.append(route)
                            found_next = True
                            break
                        else:  # To another task
                            route['tasks'].append(dest_iter[1])  # Task id
                            route['cost'] += arc_data_iter['cost']
                            current_node = dest_iter
                            found_next = True
                            break
                
                if not found_next or (isinstance(current_node, int) and str(current_node) == str(depot_id)):
                    break  # Finish building this route
    
    return routes


def main():
    parser = argparse.ArgumentParser(description='Solve transit assignment problem.')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to input instance JSON file')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to output solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True, help='Time limit in seconds')
    parser.add_argument('--log_path', type=str, default=None, help='Path to logging file')
    
    args = parser.parse_args()
    
    instance_data = read_instance(args.instance_path)
    solution, logger = solve_transit_assignment(instance_data, args.time_limit, args.log_path)
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == '__main__':
    def solve_transit_assignment(instance_data, time_limit, log_path=None):
        """Main solving function"""
        return solve_with_gurobi(instance_data), SolutionLogger(log_path, sense="minimize") if log_path else None
    
    main()