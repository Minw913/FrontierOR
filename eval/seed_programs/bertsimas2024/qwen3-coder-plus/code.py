import argparse
import json
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import signal
import sys
import time


class TimeoutHandler:
    def __init__(self, timeout_duration):
        self.timeout_duration = timeout_duration
        self.timed_out = False
        
    def timeout_handler(self, signum, frame):
        self.timed_out = True
        raise TimeoutError("Time limit exceeded")

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.timeout_handler)
        signal.alarm(self.timeout_duration)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.alarm(0)
        if isinstance(exc_val, TimeoutError):
            return True  # Suppress the exception


def solve_network_design(instance_data, time_limit, log_path=None):
    # Extract instance data
    num_nodes = instance_data["num_nodes"]
    num_commodities = instance_data["num_commodities"]
    num_scenarios = instance_data["num_scenarios"]
    num_edges = instance_data["num_edges"]
    gamma = instance_data["gamma"]
    c_0 = instance_data["c_0"]
    edges = instance_data["edges"]
    construction_costs = instance_data["construction_costs"]
    flow_costs = instance_data["flow_costs"]
    capacities = instance_data["capacities"]
    demands = instance_data["demands"]

    # Create model
    model = gp.Model("NetworkDesign")
    
    # Set parameters
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', time_limit)

    # Decision variables
    # z[e]: binary variable indicating if edge e is activated
    z = model.addVars(num_edges, vtype=GRB.BINARY, name="z")
    
    # x[k,r,e]: flow of commodity k in scenario r on edge e
    x = model.addVars(num_commodities, num_scenarios, num_edges, 
                      lb=0, name="x")
    
    # tf[r,e]: total flow on edge e in scenario r
    tf = model.addVars(num_scenarios, num_edges, lb=0, name="tf")

    # Constraints
    # Flow conservation for each commodity k, scenario r, and node i
    for k in range(num_commodities):
        for r in range(num_scenarios):
            for i in range(num_nodes):
                # Calculate net supply/demand for commodity k in scenario r at node i
                net_supply = demands[r][k][i]
                
                # Calculate outgoing and incoming flows for node i
                outgoing_flow = gp.quicksum(x[k, r, e] for e in range(num_edges) 
                                            if edges[e][0] == i)
                incoming_flow = gp.quicksum(x[k, r, e] for e in range(num_edges) 
                                            if edges[e][1] == i)
                
                model.addConstr(outgoing_flow - incoming_flow == net_supply,
                                name=f"flow_conservation_k{k}_r{r}_i{i}")

    # Capacity constraints: total flow on edge e in scenario r <= capacity if edge is active
    for r in range(num_scenarios):
        for e in range(num_edges):
            model.addConstr(tf[r, e] == gp.quicksum(x[k, r, e] for k in range(num_commodities)),
                            name=f"total_flow_def_r{r}_e{e}")
            model.addConstr(tf[r, e] <= capacities[e] * z[e],
                            name=f"capacity_r{r}_e{e}")

    # Edge activation constraint: total number of activated edges <= c_0
    model.addConstr(gp.quicksum(z[e] for e in range(num_edges)) <= c_0,
                    name="max_edges")

    # Objective function
    # Construction cost
    construction_cost = gp.quicksum(construction_costs[e] * z[e] for e in range(num_edges))
    
    # Expected transportation cost
    transport_cost = (1.0 / num_scenarios) * gp.quicksum(
        flow_costs[e] * x[k, r, e] 
        for k in range(num_commodities) 
        for r in range(num_scenarios) 
        for e in range(num_edges)
    )
    
    # Expected quadratic congestion penalty
    congestion_penalty = (1.0 / num_scenarios) * gp.quicksum(
        (1.0 / (2 * gamma)) * tf[r, e] * tf[r, e] 
        for r in range(num_scenarios) 
        for e in range(num_edges)
    )
    
    # Total objective
    model.setObjective(construction_cost + transport_cost + congestion_penalty, GRB.MINIMIZE)

    # Solve
    start_time = time.time()
    model.optimize()
    end_time = time.time()

    # Prepare solution
    solution = {}
    
    # Extract variable values
    z_sol = {e: int(z[e].X) for e in range(num_edges) if z[e].X > 0.5}
    x_sol = {(k, r, e): x[k, r, e].X 
             for k in range(num_commodities) 
             for r in range(num_scenarios) 
             for e in range(num_edges) 
             if x[k, r, e].X > 1e-6}
    
    # Build model_variables dict according to schema
    model_variables = {}
    for e in z_sol:
        model_variables[f"z[{e}]"] = z_sol[e]
    for (k, r, e) in x_sol:
        model_variables[f"x[{k},{r},{e}]"] = x_sol[(k, r, e)]
    for r in range(num_scenarios):
        for e in range(num_edges):
            if tf[r, e].X > 1e-6:
                model_variables[f"tf[{r},{e}]"] = tf[r, e].X
    
    # Also build z dict separately as required by schema
    z_dict = {f"{e}": int(z[e].X) for e in range(num_edges)}
    
    solution = {
        "objective_value": model.ObjVal,
        "model_variables": model_variables,
        "z": z_dict
    }
    
    return solution


def main():
    parser = argparse.ArgumentParser(description='Solve Network Design Problem')
    parser.add_argument('--instance_path', type=str, required=True,
                        help='Path to the JSON instance file')
    parser.add_argument('--solution_path', type=str, required=True,
                        help='Path to write the solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True,
                        help='Maximum runtime in seconds')
    parser.add_argument('--log_path', type=str, default=None,
                        help='Path to log file for intermediate solutions')

    args = parser.parse_args()

    # Load instance data
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)

    # Initialize logger if provided
    logger = None
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize")

    try:
        # Solve the problem
        solution = solve_network_design(instance_data, args.time_limit, logger)
        
        # Log final solution if logger exists
        if logger:
            logger.log_solution(solution["objective_value"], solution)
        
        # Write solution to file
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f)
        
    except Exception as e:
        print(f"Error occurred: {e}", file=sys.stderr)
        # In case of error, still try to output the best solution found so far
        try:
            model = gp.Model()  # Placeholder to check if we have a model object
            if model.SolCount > 0:
                solution = {
                    "objective_value": model.ObjVal,
                    "model_variables": {},
                    "z": {}
                }
                with open(args.solution_path, 'w') as f:
                    json.dump(solution, f)
        except:
            pass  # If there's no model or solution, just exit


if __name__ == "__main__":
    main()