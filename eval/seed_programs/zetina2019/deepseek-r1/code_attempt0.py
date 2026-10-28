import argparse
import json
import gurobipy as grb
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser(description='Solve Network Design with Multi-Commodity Flows')
    parser.add_argument('--instance_path', required=True, help='Path to the instance JSON file.')
    parser.add_argument('--solution_path', required=True, help='Path to write the solution JSON file.')
    parser.add_argument('--time_limit', type=int, required=True, help='Maximum runtime in seconds.')
    parser.add_argument('--log_path', default=None, help='Path for logging intermediate solutions (JSONL).')
    args = parser.parse_args()

    # Read the instance data
    try:
        with open(args.instance_path, 'r') as f:
            instance = json.load(f)
    except FileNotFoundError:
        print(f"Error: Instance file not found at {args.instance_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in the instance file at {args.instance_path}")
        return

    # Initialize the solution logger if log_path is provided
    logger = None
    if args.log_path:
        try:
            logger = SolutionLogger(args.log_path, sense="minimize")
        except Exception as e:
            print(f"Warning: Failed to initialize SolutionLogger: {e}")
            logger = None

    try:
        solve_instance(instance, args, logger)
    except grb.GurobiError as e:
        print(f"Gurobi error encountered: {e}")
        output_empty_solution(instance, args.solution_path)
    except Exception as e:
        print(f"Error: {e}")
        output_empty_solution(instance, args.solution_path)

def solve_instance(instance, args, logger):
    num_nodes = instance['num_nodes']
    arcs_list = instance['arcs']
    num_commodities = instance['num_commodities']
    commodities = instance['commodities']
    
    # Precompute the graph structure: incoming and outgoing arcs per node
    incoming_arcs = [[] for _ in range(num_nodes)]
    outgoing_arcs = [[] for _ in range(num_nodes)]
    all_arc_keys = []
    for arc in arcs_list:
        u, v = arc
        arc_key = f"{u}_{v}"
        all_arc_keys.append(arc_key)
        outgoing_arcs[u].append(arc_key)
        incoming_arcs[v].append(arc_key)
    
    # Initialize Gurobi model and set parameters
    model = grb.Model()
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('OutputFlag', 0)  # Suppress standard output
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', args.time_limit)

    # Add fixed-cost (arc installation) variables
    y_vars = {}
    for arc_key in all_arc_keys:
        y_vars[arc_key] = model.addVar(vtype=grb.GRB.BINARY, name=f'y_{arc_key}')

    # Add flow variables for each commodity and arc
    x_vars = {}
    for k_idx in range(num_commodities):
        for arc_key in all_arc_keys:
            var_key = (k_idx, arc_key)
            x_vars[var_key] = model.addVar(lb=0.0, ub=1.0, vtype=grb.GRB.CONTINUOUS, name=f'x_{k_idx}_{arc_key}')

    # Add flow conservation constraints for each commodity and node
    for k_idx, comm in enumerate(commodities):
        origin = comm['origin']
        dest = comm['destination']
        demand = comm['demand']  # Not needed here, used in objective
        for node in range(num_nodes):
            if node == origin:
                rhs = -1.0
            elif node == dest:
                rhs = 1.0
            else:
                rhs = 0.0
            # Sum flows into the node minus flows out of the node
            inflows = []
            for arc_key in incoming_arcs[node]:
                var_key = (k_idx, arc_key)
                inflows.append(x_vars[var_key])
            outflows = []
            for arc_key in outgoing_arcs[node]:
                var_key = (k_idx, arc_key)
                outflows.append(x_vars[var_key])
            expr = grb.quicksum(inflows) - grb.quicksum(outflows)
            model.addConstr(expr == rhs, name=f'flow_cons_k{k_idx}_n{node}')
    
    # Add arc usage constraints: flow must be 0 if arc not installed
    for k_idx in range(num_commodities):
        for arc_key in all_arc_keys:
            var_key = (k_idx, arc_key)
            model.addConstr(x_vars[var_key] <= y_vars[arc_key], name=f'arc_usage_k{k_idx}_{arc_key}')

    # Set objective: fixed costs plus variable (flow) costs
    fixed_cost_expr = grb.quicksum(instance['fixed_costs'][arc_key] * y_vars[arc_key] for arc_key in all_arc_keys)
    variable_cost_expr = 0.0
    for k_idx, comm in enumerate(commodities):
        demand = comm['demand']
        for arc_key in all_arc_keys:
            unit_cost = instance['variable_costs'][arc_key][k_idx]
            var_key = (k_idx, arc_key)
            variable_cost_expr += demand * unit_cost * x_vars[var_key]
    total_cost = fixed_cost_expr + variable_cost_expr
    model.setObjective(total_cost, grb.GRB.MINIMIZE)

    # Optimize with callback for logging intermediate solutions
    if logger:
        def gurobi_callback(model, where):
            if where == grb.GRB.Callback.MIPSOL:
                obj = model.cbGet(grb.GRB.Callback.MIPSOL_OBJ)
                logger.log(obj)
        model.optimize(gurobi_callback)
    else:
        model.optimize()

    # Extract and save solution
    if model.SolCount > 0:
        sol_open_arcs = {}
        for arc_key in all_arc_keys:
            y_val = y_vars[arc_key].X
            if y_val > 0.5:
                sol_open_arcs[arc_key] = 1
            else:
                sol_open_arcs[arc_key] = 0

        sol_routings = {}
        for k_idx in range(num_commodities):
            comm_dict = {}
            for arc_key in all_arc_keys:
                var_key = (k_idx, arc_key)
                x_val = x_vars[var_key].X
                comm_dict[arc_key] = x_val
            sol_routings[str(k_idx)] = comm_dict
        
        sol_obj = model.ObjVal
    else:
        sol_obj, sol_open_arcs, sol_routings = build_empty_solution(instance)

    # Construct and save the solution dictionary
    solution_dict = {
        "objective_value": sol_obj,
        "open_arcs": sol_open_arcs,
        "routings": sol_routings
    }
    with open(args.solution_path, 'w') as f:
        json.dump(solution_dict, f)

def build_empty_solution(instance):
    arcs_list = instance['arcs']
    num_commodities = instance['num_commodities']
    open_arcs = {}
    routings = {}
    for arc in arcs_list:
        u, v = arc
        arc_key = f"{u}_{v}"
        open_arcs[arc_key] = 0
    for k_idx in range(num_commodities):
        comm_dict = {}
        for arc in arcs_list:
            u, v = arc
            arc_key = f"{u}_{v}"
            comm_dict[arc_key] = 0.0
        routings[str(k_idx)] = comm_dict
    objective_value = 0.0
    return objective_value, open_arcs, routings

def output_empty_solution(instance, solution_path):
    objective_value, open_arcs, routings = build_empty_solution(instance)
    solution_dict = {
        "objective_value": objective_value,
        "open_arcs": open_arcs,
        "routings": routings
    }
    with open(solution_path, 'w') as f:
        json.dump(solution_dict, f)

if __name__ == '__main__':
    main()