import argparse
import json
import math
import sys
import os
import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    logger = None
    if args.log_path and SolutionLogger:
        logger = SolutionLogger(args.log_path, sense="minimize")

    L = data["machine_length"]
    K = data["num_patterns"]
    M = data["M"]
    cost_setup = data["cost_setup_pattern"]
    cost_rep = data["cost_repetition"]
    cost_knife = data.get("cost_knife_setup", 0.0)
    prob_type = data["problem_type"]
    is_closed = data.get("x_i_forced_zero", False)
    if prob_type == "closed_ended_cscp":
        is_closed = True

    products = data["products"]
    product_ids = [p["id"] for p in products]
    
    prod_len = {p["id"]: p["length"] for p in products}
    prod_Nj = {p["id"]: p["N_i"] for p in products}
    prod_dem = {p["id"]: p["demand"] for p in products}
    
    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 1)
    env.start()
    
    model = gp.Model("cscp", env=env)
    model.setParam("Seed", 0)
    model.setParam("MIPGap", 1e-4)
    model.setParam("NumericFocus", 0)
    model.setParam("Threads", 1)
    if args.time_limit > 0:
        model.setParam("TimeLimit", args.time_limit)

    u = {}
    r = {}
    for k in range(K):
        u[k] = model.addVar(vtype=GRB.BINARY, name=f"u_{k}")
        r[k] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=M, name=f"r_{k}")

    y = {}
    p_var = {}
    x = {}
    for j in product_ids:
        x[j] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"x_{j}")
        Nj = prod_Nj[j]
        for k in range(K):
            y[k, j] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=Nj, name=f"y_{k}_{j}")
            p_var[k, j] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"p_{k}_{j}")

    # Linearization variables for y_kj * r_k via binary expansion of y_kj
    b = {}
    w = {}
    for j in product_ids:
        Nj = prod_Nj[j]
        Bj = int(math.floor(math.log2(Nj))) if Nj > 0 else -1
        for k in range(K):
            for v in range(Bj + 1):
                b[k, j, v] = model.addVar(vtype=GRB.BINARY, name=f"b_{k}_{j}_{v}")
                w[k, j, v] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"w_{k}_{j}_{v}")

    # Constraints
    for k in range(K):
        # Machine length constraint
        model.addConstr(
            gp.quicksum(prod_len[j] * y[k, j] for j in product_ids) <= L * u[k],
            name=f"cap_{k}"
        )
        # Relationship between repetitions and usage
        model.addConstr(r[k] <= M * u[k], name=f"r_ub_{k}")
        model.addConstr(u[k] <= r[k], name=f"u_lb_{k}")
        
        # Symmetry breaking constraints
        if k < K - 1:
            model.addConstr(r[k] >= r[k+1], name=f"sym_r_{k}")
            model.addConstr(u[k] >= u[k+1], name=f"sym_u_{k}")

        for j in product_ids:
            Nj = prod_Nj[j]
            model.addConstr(y[k, j] <= Nj * u[k], name=f"y_ub_{k}_{j}")
            
            if Nj == 0:
                model.addConstr(y[k, j] == 0)
                model.addConstr(p_var[k, j] == 0)
            else:
                Bj = int(math.floor(math.log2(Nj)))
                # Binary expansion of y_kj
                model.addConstr(y[k, j] == gp.quicksum((2**v) * b[k, j, v] for v in range(Bj + 1)))
                
                # Big-M constraints for w_kjv = b_kjv * r_k
                for v in range(Bj + 1):
                    model.addConstr(w[k, j, v] <= M * b[k, j, v])
                    model.addConstr(w[k, j, v] <= r[k])
                    model.addConstr(w[k, j, v] >= r[k] - M * (1 - b[k, j, v]))
                
                # Production pieces equation p_kj = y_kj * r_k
                model.addConstr(p_var[k, j] == gp.quicksum((2**v) * w[k, j, v] for v in range(Bj + 1)))

    # Demand fulfillment
    for j in product_ids:
        model.addConstr(
            gp.quicksum(p_var[k, j] for k in range(K)) + x[j] >= prod_dem[j],
            name=f"demand_{j}"
        )

    # Extra pieces limits based on problem configuration
    if is_closed:
        for j in product_ids:
            model.addConstr(x[j] == 0, name=f"closed_{j}")
    else:
        model.addConstr(
            gp.quicksum(x[j] for j in product_ids) <= gp.quicksum(r[k] for k in range(K)),
            name="open_extra"
        )

    # Objective function
    obj = gp.LinExpr()
    for k in range(K):
        obj += cost_setup * u[k]
        obj += cost_rep * r[k]
        if prob_type == "cscp_with_knife_dependent_setups":
            for j in product_ids:
                obj += cost_knife * y[k, j]

    model.setObjective(obj, GRB.MINIMIZE)

    # Helper function to project the solution fields
    def create_sol_dict(get_val, obj_val):
        is_knife = (prob_type == "cscp_with_knife_dependent_setups")
        patterns = {}
        for k in range(K):
            used = bool(round(get_val(u[k])) == 1)
            reps = int(round(get_val(r[k])))
            pieces = {}
            knife_count = 0
            for j in product_ids:
                val = int(round(get_val(y[k, j])))
                pieces[str(j)] = val
                knife_count += val
            
            pat_dict = {
                "used": used,
                "repetitions": reps,
                "pieces": pieces
            }
            if is_knife:
                pat_dict["knife_count"] = knife_count
            patterns[str(k)] = pat_dict
            
        extra_pieces = {}
        for j in product_ids:
            extra_pieces[str(j)] = int(round(get_val(x[j])))
            
        prod_quants = {}
        for j in product_ids:
            for k in range(K):
                val = int(round(get_val(p_var[k, j])))
                prod_quants[f"product_{j}_pattern_{k}"] = val
                
        return {
            "objective_value": float(obj_val),
            "patterns": patterns,
            "extra_pieces": extra_pieces,
            "production_quantities": prod_quants
        }

    def cb(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                try:
                    sol = create_sol_dict(model.cbGetSolution, obj_val)
                    logger.log_solution(obj_val, sol)
                except Exception:
                    logger.log(obj_val)

    model.optimize(cb)

    if model.SolCount > 0:
        final_sol = create_sol_dict(lambda v: v.X, model.ObjVal)
        with open(args.solution_path, "w") as f:
            json.dump(final_sol, f, indent=2)
    else:
        print("No feasible solution found.")
        sys.exit(1)

if __name__ == "__main__":
    main()