import argparse
import json
import numpy as np
import time
import sys
import os


def solve_bilevel(instance, time_limit, logger=None):
    start_time = time.time()

    n = instance['n']
    n1 = instance['n1']
    n2 = instance['n2']
    m1 = instance['m1']
    m2 = instance['m2']
    c = np.array(instance['c'], dtype=float)
    d2 = np.array(instance['d2'], dtype=float)
    A1 = np.array(instance['A1'], dtype=float)
    B1 = np.array(instance['B1'], dtype=float)
    b1 = np.array(instance['b1'], dtype=float)
    A2 = np.array(instance['A2'], dtype=float)
    B2 = np.array(instance['B2'], dtype=float)
    b2 = np.array(instance['b2'], dtype=float)
    integer_follower_vars = set(i - 1 for i in instance['integer_follower_vars'])  # 0-based

    best_obj = -float('inf')
    best_solution = None

    def time_remaining():
        return max(time_limit - (time.time() - start_time), 0.1)

    def update_best(x_val, y_val, leader_obj, follower_obj):
        nonlocal best_obj, best_solution
        if leader_obj > best_obj + 1e-8:
            best_obj = leader_obj
            best_solution = {
                "objective_value": float(leader_obj),
                "x": [float(v) for v in x_val],
                "y": [float(v) for v in y_val],
                "follower_objective_value": float(follower_obj)
            }
            if logger:
                logger.log_solution(best_obj, best_solution)

    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception as e:
        print(f"Cannot import gurobipy: {e}", file=sys.stderr)
        return None

    def create_env():
        env = gp.Env(empty=True)
        env.setParam('OutputFlag', 0)
        env.start()
        return env

    try:
        env = create_env()
    except Exception as e:
        print(f"Cannot create Gurobi env: {e}", file=sys.stderr)
        return None

    def create_model(name):
        m = gp.Model(name, env=env)
        m.Params.Seed = 0
        m.Params.MIPGap = 1e-4
        m.Params.NumericFocus = 0
        m.Params.Threads = 1
        m.Params.OutputFlag = 0
        return m

    # Compute variable bounds
    M_x = np.full(n1, 500.0)
    A_all = np.vstack([A1, A2])
    b_all = np.concatenate([b1, b2])
    for j in range(n1):
        col = A_all[:, j]
        mask = col > 1e-10
        if np.any(mask):
            M_x[j] = min(np.min(b_all[mask] / col[mask]) + 10, 500)

    M_y = np.full(n2, 500.0)
    B_all = np.vstack([B1, B2])
    for j in range(n2):
        col = B_all[:, j]
        mask = col > 1e-10
        if np.any(mask):
            M_y[j] = min(np.min(b_all[mask] / col[mask]) + 10, 500)

    max_iterations = 100

    # Build HPR master model
    try:
        master = create_model("HPR")

        x = master.addVars(n1, vtype=GRB.INTEGER, lb=0, name="x")
        for j in range(n1):
            x[j].UB = M_x[j]

        y = master.addVars(n2, lb=0, name="y")
        for j in range(n2):
            y[j].UB = M_y[j]
            if j in integer_follower_vars:
                y[j].vtype = GRB.INTEGER

        # Upper-level constraints
        for i in range(m1):
            master.addConstr(
                gp.quicksum(A1[i, j] * x[j] for j in range(n1)) +
                gp.quicksum(B1[i, j] * y[j] for j in range(n2)) <= b1[i]
            )

        # Lower-level constraints
        for i in range(m2):
            master.addConstr(
                gp.quicksum(A2[i, j] * x[j] for j in range(n1)) +
                gp.quicksum(B2[i, j] * y[j] for j in range(n2)) <= b2[i]
            )

        master.setObjective(gp.quicksum(c[j] * x[j] for j in range(n1)), GRB.MAXIMIZE)

        for iteration in range(max_iterations):
            if time_remaining() < 5:
                break

            master.Params.TimeLimit = max(time_remaining() * 0.3, 5)
            master.optimize()

            if master.SolCount == 0:
                break

            x_val = [round(x[j].X) for j in range(n1)]
            y_val_hpr = [y[j].X for j in range(n2)]
            for j in integer_follower_vars:
                y_val_hpr[j] = round(y_val_hpr[j])

            leader_obj_candidate = sum(c[j] * x_val[j] for j in range(n1))

            # If HPR objective <= best known, no improvement possible
            hpr_obj = master.ObjVal
            if hpr_obj <= best_obj + 1e-6:
                break

            x_arr = np.array(x_val, dtype=float)

            # Solve follower's MILP for this x
            rhs_lower = b2 - A2 @ x_arr

            if time_remaining() < 3:
                break

            # Solve follower
            fmodel = create_model("Follower")
            fmodel.Params.TimeLimit = max(min(time_remaining() * 0.2, 30), 2)

            fy = fmodel.addVars(n2, lb=0, name="fy")
            for j in range(n2):
                fy[j].UB = M_y[j]
                if j in integer_follower_vars:
                    fy[j].vtype = GRB.INTEGER

            feasible = True
            for i in range(m2):
                if rhs_lower[i] < -1e-6:
                    if np.all(B2[i, :] >= -1e-10):
                        feasible = False
                        break
                fmodel.addConstr(
                    gp.quicksum(B2[i, j] * fy[j] for j in range(n2)) <= rhs_lower[i]
                )

            if not feasible:
                fmodel.dispose()
                master.addConstr(
                    gp.quicksum(x[j] for j in range(n1) if x_val[j] > 0.5) -
                    gp.quicksum(x[j] for j in range(n1) if x_val[j] < 0.5) <=
                    sum(1 for j in range(n1) if x_val[j] > 0.5) - 1
                )
                continue

            fmodel.setObjective(gp.quicksum(d2[j] * fy[j] for j in range(n2)), GRB.MAXIMIZE)
            fmodel.optimize()

            if fmodel.SolCount == 0:
                fmodel.dispose()
                master.addConstr(
                    gp.quicksum(x[j] for j in range(n1) if x_val[j] > 0.5) -
                    gp.quicksum(x[j] for j in range(n1) if x_val[j] < 0.5) <=
                    sum(1 for j in range(n1) if x_val[j] > 0.5) - 1
                )
                continue

            fopt = fmodel.ObjVal
            fy_opt = [fy[j].X for j in range(n2)]
            fmodel.dispose()

            if time_remaining() < 2:
                y_arr = np.array(fy_opt)
                for j in integer_follower_vars:
                    y_arr[j] = round(y_arr[j])
                upper_check = A1 @ x_arr + B1 @ y_arr
                if np.all(upper_check <= b1 + 1e-6):
                    fobj = float(d2 @ y_arr)
                    update_best(x_val, y_arr.tolist(), leader_obj_candidate, fobj)
                break

            # Optimistic: find follower y that also satisfies upper-level constraints
            omodel = create_model("Optimistic")
            omodel.Params.TimeLimit = max(min(time_remaining() * 0.2, 30), 2)

            oy = omodel.addVars(n2, lb=0, name="oy")
            for j in range(n2):
                oy[j].UB = M_y[j]
                if j in integer_follower_vars:
                    oy[j].vtype = GRB.INTEGER

            for i in range(m2):
                omodel.addConstr(
                    gp.quicksum(B2[i, j] * oy[j] for j in range(n2)) <= rhs_lower[i]
                )

            omodel.addConstr(
                gp.quicksum(d2[j] * oy[j] for j in range(n2)) >= fopt - 1e-3
            )

            rhs_upper = b1 - A1 @ x_arr
            for i in range(m1):
                omodel.addConstr(
                    gp.quicksum(B1[i, j] * oy[j] for j in range(n2)) <= rhs_upper[i]
                )

            omodel.setObjective(gp.quicksum(d2[j] * oy[j] for j in range(n2)), GRB.MAXIMIZE)
            omodel.optimize()

            if omodel.SolCount > 0:
                y_opt = [oy[j].X for j in range(n2)]
                for j in integer_follower_vars:
                    y_opt[j] = round(y_opt[j])
                fobj = sum(d2[j] * y_opt[j] for j in range(n2))
                update_best(x_val, y_opt, leader_obj_candidate, fobj)
                omodel.dispose()
            else:
                omodel.dispose()

            master.addConstr(
                gp.quicksum(x[j] for j in range(n1) if x_val[j] > 0.5) -
                gp.quicksum(x[j] for j in range(n1) if x_val[j] < 0.5) <=
                sum(1 for j in range(n1) if x_val[j] > 0.5) - 1
            )

        master.dispose()
    except Exception as e:
        print(f"Error in main solve loop: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)

    # If no solution found, try x=0
    if best_solution is None and time_remaining() > 2:
        try:
            x_zero = [0.0] * n1
            x_arr = np.zeros(n1)
            rhs_lower = b2 - A2 @ x_arr

            fmodel = create_model("F0")
            fmodel.Params.TimeLimit = max(time_remaining() - 1, 1)

            fy = fmodel.addVars(n2, lb=0, name="fy")
            for j in range(n2):
                if j in integer_follower_vars:
                    fy[j].vtype = GRB.INTEGER

            for i in range(m2):
                fmodel.addConstr(
                    gp.quicksum(B2[i, j] * fy[j] for j in range(n2)) <= rhs_lower[i]
                )

            for i in range(m1):
                fmodel.addConstr(
                    gp.quicksum(B1[i, j] * fy[j] for j in range(n2)) <= b1[i]
                )

            fmodel.setObjective(gp.quicksum(d2[j] * fy[j] for j in range(n2)), GRB.MAXIMIZE)
            fmodel.optimize()

            if fmodel.SolCount > 0:
                y_opt = [fy[j].X for j in range(n2)]
                for j in integer_follower_vars:
                    y_opt[j] = round(y_opt[j])
                fobj = sum(d2[j] * y_opt[j] for j in range(n2))
                update_best(x_zero, y_opt, 0.0, fobj)

            fmodel.dispose()
        except Exception as e:
            print(f"Error in fallback: {e}", file=sys.stderr)

    try:
        env.dispose()
    except:
        pass

    return best_solution


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    solution = None
    try:
        solution = solve_bilevel(instance, args.time_limit, logger)
    except Exception as e:
        print(f"Error during solve: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)

    if solution is None:
        n1 = instance['n1']
        n2 = instance['n2']
        solution = {
            "objective_value": 0.0,
            "x": [0.0] * n1,
            "y": [0.0] * n2,
            "follower_objective_value": 0.0
        }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()