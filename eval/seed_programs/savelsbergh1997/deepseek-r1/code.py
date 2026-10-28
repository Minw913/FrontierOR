import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def greedy_assignment(num_agents, num_jobs, profits, weights, capacities):
    remaining_cap = capacities[:]  # Make a copy
    assignment = {}
    total_profit = 0

    job_list = []
    for j in range(num_jobs):
        max_profit = max(profits[i][j] for i in range(num_agents))
        job_list.append((j, max_profit))
    job_list.sort(key=lambda x: x[1], reverse=True)

    for job_index, _ in job_list:
        best_agent = None
        best_profit = -1
        for i in range(num_agents):
            if weights[i][job_index] <= remaining_cap[i]:
                if profits[i][job_index] > best_profit:
                    best_profit = profits[i][job_index]
                    best_agent = i
        
        if best_agent is None:
            min_violation = float('inf')
            candidate = None
            for i in range(num_agents):
                violation = weights[i][job_index] - remaining_cap[i]
                if violation < min_violation:
                    min_violation = violation
                    candidate = i
            best_agent = candidate
        
        assignment[job_index] = best_agent
        total_profit += profits[best_agent][job_index]
        remaining_cap[best_agent] -= weights[best_agent][job_index]
    
    return assignment, total_profit

def main():
    parser = argparse.ArgumentParser(description='Solve job assignment problem.')
    parser.add_argument('--instance_path', type=str, required=True, help='Path to the instance JSON file.')
    parser.add_argument('--solution_path', type=str, required=True, help='Path to write the solution JSON file.')
    parser.add_argument('--time_limit', type=int, required=True, help='Maximum runtime in seconds.')
    parser.add_argument('--log_path', type=str, default=None, help='Path to log intermediate solutions (optional).')
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        data = json.load(f)
    
    num_agents = data['num_agents']
    num_jobs = data['num_jobs']
    profits = data['profits']
    weights = data['weights']
    capacities = data['capacities']
    
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    try:
        model = gp.Model()
        model.setParam("Threads", 1)
        x = {}
        for i in range(num_agents):
            for j in range(num_jobs):
                x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

        for j in range(num_jobs):
            model.addConstr(gp.quicksum(x[i, j] for i in range(num_agents)) == 1, name=f"job_{j}_assignment")
        
        for i in range(num_agents):
            model.addConstr(gp.quicksum(weights[i][j] * x[i, j] for j in range(num_jobs)) <= capacities[i], name=f"cap_{i}")
        
        model.setObjective(gp.quicksum(profits[i][j] * x[i, j] for i in range(num_agents) for j in range(num_jobs)), GRB.MAXIMIZE)
        
        model.Params.TimeLimit = args.time_limit
        model.Params.LogToConsole = 0
        
        if logger:
            def callback(model, where):
                if where == GRB.Callback.MIPSOL:
                    obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    logger.log(obj)
            model.optimize(callback)
        else:
            model.optimize()
        
        if model.SolCount > 0:
            total_profit = model.objVal
            assignment = {}
            for j in range(num_jobs):
                for i in range(num_agents):
                    if x[i, j].X > 0.5:
                        assignment[j] = i
                        break
        else:
            assignment, total_profit = greedy_assignment(num_agents, num_jobs, profits, weights, capacities)
    
    except gp.GurobiError as e:
        print(f"Gurobi encountered an error: {e}")
        assignment, total_profit = greedy_assignment(num_agents, num_jobs, profits, weights, capacities)
    except Exception as e:
        print(f"Unexpected error: {e}")
        assignment, total_profit = greedy_assignment(num_agents, num_jobs, profits, weights, capacities)
    
    assignment_output = {str(j): int(agent) for j, agent in assignment.items()}
    solution = {
        "objective_value": float(total_profit),
        "assignment": assignment_output
    }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == "__main__":
    main()