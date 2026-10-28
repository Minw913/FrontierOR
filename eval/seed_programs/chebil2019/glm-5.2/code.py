import argparse
import json
import time
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    start_time = time.time()

    # Load instance
    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    families = instance["families"]
    capacity = instance["knapsack_capacity"]

    # Trivial feasible solution: activate nothing, select nothing -> profit 0
    best_obj = [0.0]
    best_solution = {
        "objective_value": 0.0,
        "families_selected": [],
        "items_selected": [],
    }
    if logger:
        logger.log_solution(0.0, best_solution)

    # Try to solve with Gurobi
    try:
        import gurobipy as gp
        from gurobipy import GRB

        env = gp.Env(empty=True)
        env.setParam("Threads", 1)
        env.setParam("Seed", 0)
        env.setParam("MIPGap", 1e-4)
        env.setParam("NumericFocus", 0)
        env.setParam("OutputFlag", 0)
        env.start()

        model = gp.Model(env=env)

        # Reserve some time for output writing
        remaining = max(1.0, args.time_limit - (time.time() - start_time) - 1.0)
        model.setParam("TimeLimit", remaining)

        x = {}
        y = {}
        item_meta = {}

        for fam in families:
            fid = fam["family_id"]
            x[fid] = model.addVar(vtype=GRB.BINARY, name=f"x_{fid}")
            for it in fam["items"]:
                iid = it["item_id"]
                y[(fid, iid)] = model.addVar(vtype=GRB.BINARY, name=f"y_{fid}_{iid}")
                item_meta[(fid, iid)] = it

        model.update()

        # Capacity constraint
        weight_expr = gp.LinExpr()
        for fam in families:
            fid = fam["family_id"]
            weight_expr.add(fam["setup_capacity"] * x[fid])
            for it in fam["items"]:
                iid = it["item_id"]
                weight_expr.add(it["weight"] * y[(fid, iid)])
        model.addConstr(weight_expr <= capacity, "capacity")

        # Linking constraints
        for fam in families:
            fid = fam["family_id"]
            for it in fam["items"]:
                iid = it["item_id"]
                model.addConstr(y[(fid, iid)] <= x[fid], f"link_{fid}_{iid}")

        # Objective: maximize sum profit*y - sum setup_cost*x
        obj = gp.LinExpr()
        for fam in families:
            fid = fam["family_id"]
            obj.add(-fam["setup_cost"] * x[fid])
            for it in fam["items"]:
                iid = it["item_id"]
                obj.add(it["profit"] * y[(fid, iid)])
        model.setObjective(obj, GRB.MAXIMIZE)

        # Provide trivial warm start (all zero) - feasible
        for fid in x:
            x[fid].Start = 0.0
        for key in y:
            y[key].Start = 0.0

        # Callback to log incumbents during the search
        def callback(model, where):
            if where == GRB.Callback.MIPSOL:
                try:
                    obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                except Exception:
                    return
                if obj_val > best_obj[0] + 1e-9:
                    families_selected = []
                    items_selected = []
                    for fam in families:
                        fid = fam["family_id"]
                        xv = model.cbGetSolution(x[fid])
                        if xv > 0.5:
                            families_selected.append(fid)
                            for it in fam["items"]:
                                iid = it["item_id"]
                                yv = model.cbGetSolution(y[(fid, iid)])
                                if yv > 0.5:
                                    items_selected.append({"family": fid, "item": iid})
                    sol = {
                        "objective_value": float(obj_val),
                        "families_selected": families_selected,
                        "items_selected": items_selected,
                    }
                    best_obj[0] = float(obj_val)
                    best_solution_ref[0] = sol
                    if logger:
                        logger.log_solution(float(obj_val), sol)

        best_solution_ref = [best_solution]

        model.optimize(callback)

        # After optimization, prefer the best incumbent we have tracked
        if model.SolCount > 0:
            obj_val = model.ObjVal
            if obj_val > best_obj[0] + 1e-9:
                families_selected = []
                items_selected = []
                for fam in families:
                    fid = fam["family_id"]
                    if x[fid].X > 0.5:
                        families_selected.append(fid)
                        for it in fam["items"]:
                            iid = it["item_id"]
                            if y[(fid, iid)].X > 0.5:
                                items_selected.append({"family": fid, "item": iid})
                best_solution = {
                    "objective_value": float(obj_val),
                    "families_selected": families_selected,
                    "items_selected": items_selected,
                }
                if logger:
                    logger.log_solution(float(obj_val), best_solution)
            else:
                best_solution = best_solution_ref[0]

    except Exception as e:
        sys.stderr.write(f"Gurobi solve failed: {e}\n")
        best_solution = best_solution

    # Ensure final solution written
    with open(args.solution_path, "w") as f:
        json.dump(best_solution, f, indent=2)

if __name__ == "__main__":
    main()