import argparse
import json
import numpy as np
import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None

def main():
    parser = argparse.ArgumentParser(description="Portfolio Optimization Solver")
    parser.add_argument("--instance_path", type=str, required=True, help="Path to instance JSON file")
    parser.add_argument("--solution_path", type=str, required=True, help="Path where the solution JSON is written")
    parser.add_argument("--time_limit", type=int, required=True, help="Time limit in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Path for logging intermediate solutions")
    args = parser.parse_args()

    # Initialize Logger
    if args.log_path and SolutionLogger is not None:
        logger = SolutionLogger(args.log_path, sense="minimize")
    else:
        logger = None

    # Load Instance Data
    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    n = instance["n"]
    k = instance["k"]
    b = float(instance["b"])
    
    # Convert lists into continuous numpy arrays for efficient matrix vectorizations
    r = np.array(instance["r"], dtype=float)
    Q = np.array(instance["Q"], dtype=float)

    # Setup Gurobi Core Environment
    model = gp.Model("Portfolio_Optimization")
    model.setParam("TimeLimit", args.time_limit)
    model.setParam("Threads", 1)  # Strictly adhering to restricted CPU cores

    # Callback mechanism to log best feasible integer solutions internally recognized
    def cb(model, where):
        if where == GRB.Callback.MIPSOL:
            try:
                obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if logger is not None:
                    logger.log(obj)
            except Exception:
                pass

    # Variables
    # The portfolio weights `x` can take any real value unrestricted (long or short positions allowed).
    x = model.addMVar(n, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="x")

    # Constraints
    # The weights must sum to exactly one
    model.addConstr(x.sum() == 1.0, name="sum_1")
    
    # Portfolio expected return must meet minimum acceptable average `b`
    model.addConstr(r @ x >= b, name="min_return")

    # Cardinality constraints & Variables linkage
    # Apply Cardinality Constraint entirely when k < n, avoiding useless integer variables when k >= n.
    if k < n:
        y = model.addMVar(n, vtype=GRB.BINARY, name="y")
        # `z` signifies the negated state of our actual binary active-variables `y`
        z = model.addMVar(n, vtype=GRB.BINARY, name="z")
        
        # At most `k` positions assigned non-zero
        model.addConstr(y.sum() <= k, name="cardinality")
        
        # Relational toggle constraint enforcing binary opposite: z_i = 1 - y_i
        model.addConstr(y + z == np.ones(n), name="yz_relation")
        
        # SOS Type 1 implicitly guards against Unbounded Indicator Variables:
        # If y[i] == 0, then z[i] == 1 => x[i] must absolutely act as 0
        for i in range(n):
            model.addSOS(GRB.SOS_TYPE1, [x[i].item(), z[i].item()])

    # Objective Function
    # Formulation calculates exact half of weighted portfolio return covariance measure (x_i * Q_ij * x_j / 2)
    model.setObjective(0.5 * (x @ Q @ x), GRB.MINIMIZE)

    # Execution Call Path
    if k < n and logger is not None:
        model.optimize(cb)
    else:
        model.optimize()
        if logger is not None and model.SolCount > 0:
            logger.log(model.ObjVal)

    # Build Response if feasible incumbent is secured
    if model.SolCount > 0:
        obj_val = model.ObjVal
        x_val = x.X.tolist()
        
        solution = {
            "objective_value": obj_val,
            "x": x_val
        }
        
        with open(args.solution_path, "w") as f:
            json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()