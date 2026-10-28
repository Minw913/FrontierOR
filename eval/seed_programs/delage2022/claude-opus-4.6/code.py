import argparse
import json
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def solve(instance_path, solution_path, time_limit, log_path):
    with open(instance_path, 'r') as f:
        data = json.load(f)

    logger = SolutionLogger(log_path, sense="minimize") if log_path else None

    problem_type = data["problem_type"]
    machine_length = data["machine_length"]
    num_products = data["num_products"]
    num_patterns = data["num_patterns"]
    cost_setup = data["cost_setup_pattern"]
    cost_rep = data["cost_repetition"]
    M_val = data["M"]
    products = data["products"]

    is_open = problem_type == "open_ended_cscp"
    is_closed = problem_type == "closed_ended_cscp"
    is_knife = problem_type == "cscp_with_knife_dependent_setups"
    cost_knife = data.get("cost_knife_setup", 0) if is_knife else 0

    prod_ids = [p["id"] for p in products]
    prod_length = {p["id"]: p["length"] for p in products}
    prod_demand = {p["id"]: p["demand"] for p in products}
    prod_N = {p["id"]: p["N_i"] for p in products}

    start_time = time.time()

    model = gp.Model("CSCP")
    model.setParam("Seed", 0)
    model.setParam("MIPGap", 1e-4)
    model.setParam("NumericFocus", 0)
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", max(1, time_limit - 5))

    # Decision variables
    # y[j] = 1 if pattern j is used
    y = {}
    # r[j] = number of repetitions of pattern j
    r = {}
    # a[i,j] = number of pieces of product i in pattern j
    a = {}
    # x[i] = extra pieces of product i (open-ended variants)
    x = {}

    pat_indices = list(range(num_patterns))

    for j in pat_indices:
        y[j] = model.addVar(vtype=GRB.BINARY, name=f"y_{j}")
        r[j] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=M_val, name=f"r_{j}")

    for i in prod_ids:
        for j in pat_indices:
            a[i, j] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=prod_N[i], name=f"a_{i}_{j}")

    for i in prod_ids:
        if is_closed:
            x[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=0, name=f"x_{i}")
        else:
            x[i] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"x_{i}")

    # Knife count variables for knife-dependent
    k = {}
    if is_knife:
        for j in pat_indices:
            k[j] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"k_{j}")

    model.update()

    # Objective: minimize setup + repetition costs (+ knife costs)
    obj = gp.LinExpr()
    for j in pat_indices:
        obj += cost_setup * y[j]
        obj += cost_rep * r[j]
    if is_knife:
        for j in pat_indices:
            obj += cost_knife * k[j]
    model.setObjective(obj, GRB.MINIMIZE)

    # Constraints

    # 1. Pattern length constraint: sum of lengths of pieces <= machine_length * y[j]
    # Also, if y[j]=0 then a[i,j]=0
    for j in pat_indices:
        model.addConstr(
            gp.quicksum(prod_length[i] * a[i, j] for i in prod_ids) <= machine_length * y[j],
            name=f"length_{j}"
        )

    # 2. Repetitions only if pattern is used: r[j] <= M * y[j]
    for j in pat_indices:
        model.addConstr(r[j] <= M_val * y[j], name=f"rep_limit_{j}")

    # 3. If pattern is not used, no pieces: a[i,j] <= N_i * y[j]
    for i in prod_ids:
        for j in pat_indices:
            model.addConstr(a[i, j] <= prod_N[i] * y[j], name=f"piece_limit_{i}_{j}")

    # 4. Demand satisfaction: sum_j(a[i,j]*r[j]) + x[i] >= demand[i]
    for i in prod_ids:
        model.addConstr(
            gp.quicksum(a[i, j] * r[j] for j in pat_indices) + x[i] >= prod_demand[i],
            name=f"demand_{i}"
        )

    # 5. Extra pieces constraint (open-ended): sum_i x[i] <= sum_j r[j]
    if is_open or is_knife:
        model.addConstr(
            gp.quicksum(x[i] for i in prod_ids) <= gp.quicksum(r[j] for j in pat_indices),
            name="extra_limit"
        )

    # 6. Symmetry breaking: r[j] >= r[j+1]
    for j in range(num_patterns - 1):
        model.addConstr(r[j] >= r[j + 1], name=f"sym_{j}")

    # 7. Knife count constraints
    if is_knife:
        for j in pat_indices:
            model.addConstr(
                k[j] == gp.quicksum(a[i, j] for i in prod_ids),
                name=f"knife_{j}"
            )

    # Store best solution found
    best_solution = [None]
    best_obj = [float('inf')]

    def extract_solution():
        sol = {}
        obj_val = model.ObjVal

        patterns_dict = {}
        for j in pat_indices:
            used = y[j].X > 0.5
            reps = int(round(r[j].X))
            pieces = {}
            for i in prod_ids:
                pieces[str(i)] = int(round(a[i, j].X))
            pat_entry = {
                "used": used,
                "repetitions": reps,
                "pieces": pieces
            }
            if is_knife:
                pat_entry["knife_count"] = int(round(sum(a[i, j].X for i in prod_ids)))
            patterns_dict[str(j)] = pat_entry

        extra = {}
        for i in prod_ids:
            extra[str(i)] = x[i].X if (is_open or is_knife) else 0.0

        prod_quant = {}
        for i in prod_ids:
            for j in pat_indices:
                key = f"product_{i}_pattern_{j}"
                prod_quant[key] = int(round(a[i, j].X)) * int(round(r[j].X))

        sol["objective_value"] = obj_val
        sol["patterns"] = patterns_dict
        sol["extra_pieces"] = extra
        sol["production_quantities"] = prod_quant
        return sol, obj_val

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj_val < best_obj[0]:
                best_obj[0] = obj_val
                if logger:
                    logger.log(obj_val)

    model.optimize(callback)

    solution = None
    if model.SolCount > 0:
        solution, obj_val = extract_solution()
        if logger:
            logger.log_solution(obj_val, solution)
    else:
        # Return a trivial solution if no feasible solution found
        solution = {
            "objective_value": float('inf'),
            "patterns": {},
            "extra_pieces": {},
            "production_quantities": {}
        }

    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)


if __name__ == "__main__":
    main()