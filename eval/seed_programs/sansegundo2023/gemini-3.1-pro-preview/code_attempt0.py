import json
import argparse
import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None

def solve_max_clique(instance_path, solution_path, time_limit, log_path):
    # 1. Load data
    with open(instance_path, 'r') as f:
        data = json.load(f)
        
    num_vertices = data.get("num_vertices", 0)
    vertices = data.get("vertices", [])
    adjacency_list = data.get("adjacency_list", {})
    
    # In JSON, dict keys are strings; convert them back to integers
    adj_sets = {int(u): set(neighbors) for u, neighbors in adjacency_list.items()}
    greedy_clique = set(data.get("greedy_clique", []))
    
    # 2. Setup Logger
    logger = None
    if log_path and SolutionLogger:
        logger = SolutionLogger(log_path, sense="maximize")

    # Handle edge case for empty graph
    if not vertices:
        solution = {
            "objective_value": 0.0,
            "clique_vertices": []
        }
        with open(solution_path, 'w') as f:
            json.dump(solution, f, indent=4)
        return

    # 3. Setup Gurobi Environment and Model
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    
    model = gp.Model("max_clique", env=env)
    
    # Force single-core execution as required
    model.setParam("Threads", 1)
    
    if time_limit is not None and time_limit > 0:
        model.setParam("TimeLimit", time_limit)
        
    # Max clique benefits significantly from aggressive clique cuts
    model.setParam("CliqueCuts", 2)
    
    # Create decision variables: x[v] = 1 if vertex v is in the clique, else 0
    x = model.addVars(vertices, vtype=GRB.BINARY, name="x")
    
    # Objective: Maximize the size of the clique
    model.setObjective(x.sum(), GRB.MAXIMIZE)
    
    # Inject Initial MIP Start using the provided greedy heuristic bounds
    for v in vertices:
        x[v].Start = 1.0 if v in greedy_clique else 0.0
        
    # 4. Formulate the problem globally using complement-graph (independent set) method
    # Constraint: If edge (u, v) does not exist, u and v cannot be simultaneously part of a clique.
    def generate_non_edges():
        num = len(vertices)
        for i in range(num):
            u = vertices[i]
            u_adj = adj_sets.get(u, set())
            for j in range(i + 1, num):
                v = vertices[j]
                if v not in u_adj:
                    yield u, v

    model.addConstrs((x[u] + x[v] <= 1 for u, v in generate_non_edges()), name="ne")
    
    # 5. Optimization Process Output via Callbacks
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            # Retrieved a new integer feasible incumbent
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(float(obj))
                
    model.optimize(callback)
    
    # 6. Extract the solution
    if model.SolCount > 0:
        clique_vertices = [int(v) for v in vertices if x[v].X > 0.5]
        objective_value = float(len(clique_vertices))
    else:
        # Fallback if no solution generated (unlikely with valid MIP start, but safe practice)
        clique_vertices = [int(v) for v in greedy_clique]
        if not clique_vertices and vertices:
            clique_vertices = [int(vertices[0])]
        objective_value = float(len(clique_vertices))
        
    # Dump result mapping exactly to schemas
    solution = {
        "objective_value": objective_value,
        "clique_vertices": clique_vertices
    }
    
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Max Clique Undirected Graph Optimizer")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to instance JSON")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to solution JSON")
    parser.add_argument("--time_limit", type=int, required=True, help="Time limit in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Path to logging file")
    
    args = parser.parse_args()
    
    solve_max_clique(
        instance_path=args.instance_path,
        solution_path=args.solution_path,
        time_limit=args.time_limit,
        log_path=args.log_path
    )