import argparse
import json
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def main():
    parser = argparse.ArgumentParser(description="Multi-item capacitated lot-sizing solver")
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    t_start = time.time()

    with open(args.instance_path, "r") as fh:
        data = json.load(fh)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # ---------------- Instance data ----------------
    n = int(data["num_items"])
    T = int(data["num_periods"])
    d = [[float(v) for v in row] for row in data["demand"]]
    sc = [float(v) for v in data["setup_cost"]]
    h = [float(v) for v in data["holding_cost"]]
    pc = [float(v) for v in data["variable_cost"]]
    st = [float(v) for v in data["setup_time"]]
    vt = [float(v) for v in data["variable_time"]]
    cap = [float(v) for v in data["capacity"]]
    ic = [float(v) for v in data["initial_inventory_cost"]]

    items = range(n)
    periods = range(T)

    # Remaining demand RD[i][t] = sum_{s >= t} d[i][s]
    RD = [[0.0] * T for _ in items]
    for i in items:
        run = 0.0
        for t in range(T - 1, -1, -1):
            run += d[i][t]
            RD[i][t] = run
    total_demand = [RD[i][0] for i in items]

    # Big-M: min of capacity-based bound and remaining demand
    M = [[0.0] * T for _ in items]
    for i in items:
        for t in periods:
            cap_bound = (cap[t] - st[i]) / vt[i] if vt[i] > 0.0 else RD[i][t]
            M[i][t] = min(cap_bound, RD[i][t])

    # ---------------- Solution assembly helpers ----------------
    def compute_objective(production, setup, inventory):
        obj = 0.0
        for i in items:
            obj += ic[i] * inventory[i][0]
            for t in periods:
                obj += sc[i] * setup[i][t]
                obj += pc[i] * production[i][t]
                obj += h[i] * inventory[i][t + 1]
        return float(obj)

    def assemble(i0_vals, x_vals, y_vals):
        """Clean solver values and rebuild inventory via balance equations."""
        production = []
        setup = []
        inventory = []
        for i in items:
            prow = []
            srow = []
            for t in periods:
                xv = round(float(x_vals[i][t]), 9)
                if abs(xv) < 1e-8:
                    xv = 0.0
                yv = 1 if float(y_vals[i][t]) > 0.5 else 0
                if yv == 1 and xv <= 0.0:
                    yv = 0  # setup without production is never beneficial
                if yv == 0:
                    xv = 0.0
                prow.append(xv)
                srow.append(yv)
            inv = [max(0.0, round(float(i0_vals[i]), 9))]
            for t in periods:
                nxt = inv[-1] + prow[t] - d[i][t]
                if abs(nxt) < 1e-7:
                    nxt = 0.0
                inv.append(max(0.0, round(nxt, 9)))
            inv[-1] = 0.0  # no stock after the last period
            production.append(prow)
            setup.append(srow)
            inventory.append(inv)
        sol = {
            "objective_value": compute_objective(production, setup, inventory),
            "production": production,
            "setup": setup,
            "inventory": inventory,
        }
        return sol

    def unflatten(flat):
        return [flat[i * T:(i + 1) * T] for i in items]

    # ---------------- Guaranteed-feasible fallback ----------------
    # Cover all demand with initial inventory: no setups, no production,
    # always feasible since initial inventory is uncapacitated.
    best = {
        "sol": assemble(
            [total_demand[i] for i in items],
            [[0.0] * T for _ in items],
            [[0] * T for _ in items],
        )
    }
    if logger:
        logger.log_solution(best["sol"]["objective_value"], best["sol"])

    # ---------------- MIP ----------------
    try:
        model = gp.Model("capacitated_lot_sizing")
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1
        model.Params.LogToConsole = 0
        remaining = float(args.time_limit) - (time.time() - t_start)
        model.Params.TimeLimit = max(0.5, remaining)

        X, Y, I, I0 = {}, {}, {}, {}
        for i in items:
            I0[i] = model.addVar(lb=0.0, name=f"I0_{i}")
            for t in periods:
                X[i, t] = model.addVar(lb=0.0, name=f"X_{i}_{t}")
                if M[i][t] > 1e-9:
                    Y[i, t] = model.addVar(vtype=GRB.BINARY, name=f"Y_{i}_{t}")
                else:
                    # Production impossible in this period: fix setup to 0.
                    Y[i, t] = model.addVar(lb=0.0, ub=0.0, vtype=GRB.BINARY,
                                           name=f"Y_{i}_{t}")
                ub_inv = RD[i][t + 1] if t + 1 < T else 0.0
                I[i, t] = model.addVar(lb=0.0, ub=ub_inv, name=f"I_{i}_{t}")

        obj = gp.LinExpr()
        for i in items:
            obj.add(ic[i] * I0[i])
            for t in periods:
                obj.add(sc[i] * Y[i, t])
                obj.add(pc[i] * X[i, t])
                obj.add(h[i] * I[i, t])
        model.setObjective(obj, GRB.MINIMIZE)

        # Inventory balance and setup linkage
        for i in items:
            for t in periods:
                prev = I0[i] if t == 0 else I[i, t - 1]
                model.addConstr(prev + X[i, t] == d[i][t] + I[i, t],
                                name=f"bal_{i}_{t}")
                model.addConstr(X[i, t] <= M[i][t] * Y[i, t],
                                name=f"link_{i}_{t}")

        # Period capacity
        for t in periods:
            model.addConstr(
                gp.quicksum(st[i] * Y[i, t] + vt[i] * X[i, t] for i in items) <= cap[t],
                name=f"cap_{t}",
            )

        # Valid inequality: cumulative (prefix) capacity
        for k in periods:
            model.addConstr(
                gp.quicksum(
                    st[i] * Y[i, t] + vt[i] * X[i, t]
                    for i in items for t in range(k + 1)
                ) <= sum(cap[0:k + 1]),
                name=f"prefixcap_{k}",
            )

        # Valid inequalities: inventory must cover next-period demand if no setup
        for i in items:
            I0[i] >= 0.0
            model.addConstr(I0[i] >= d[i][0] * (1 - Y[i, 0]), name=f"cover0_{i}")
            for t in range(T - 1):
                model.addConstr(I[i, t] >= d[i][t + 1] * (1 - Y[i, t + 1]),
                                name=f"cover_{i}_{t}")

        # MIP start: trivial feasible solution (all demand from initial inventory)
        for i in items:
            I0[i].Start = total_demand[i]
            for t in periods:
                X[i, t].Start = 0.0
                Y[i, t].Start = 0.0
                I[i, t].Start = RD[i][t + 1] if t + 1 < T else 0.0

        # Flat variable lists for fast callback extraction
        i0_list = [I0[i] for i in items]
        x_list = [X[i, t] for i in items for t in periods]
        y_list = [Y[i, t] for i in items for t in periods]

        state = {"last": best["sol"]["objective_value"]}

        def callback(m, where):
            if where == GRB.Callback.MIPSOL:
                try:
                    obj_val = m.cbGet(GRB.Callback.MIPSOL_OBJ)
                    if obj_val < state["last"] - 1e-7:
                        i0v = m.cbGetSolution(i0_list)
                        xv = unflatten(m.cbGetSolution(x_list))
                        yv = unflatten(m.cbGetSolution(y_list))
                        sol = assemble(i0v, xv, yv)
                        if sol["objective_value"] < state["last"] - 1e-7:
                            state["last"] = sol["objective_value"]
                            best["sol"] = sol
                            if logger:
                                logger.log_solution(sol["objective_value"], sol)
                except Exception:
                    pass

        model.optimize(callback)

        # Retrieve final incumbent
        if model.SolCount > 0:
            i0v = [v.X for v in i0_list]
            xv = unflatten([v.X for v in x_list])
            yv = unflatten([v.X for v in y_list])
            sol = assemble(i0v, xv, yv)
            if sol["objective_value"] < best["sol"]["objective_value"] - 1e-9:
                best["sol"] = sol
                if logger:
                    logger.log_solution(sol["objective_value"], sol)

    except Exception:
        # Any solver failure: keep the guaranteed-feasible fallback.
        pass

    # ---------------- Write final solution ----------------
    with open(args.solution_path, "w") as fh:
        json.dump(best["sol"], fh)


if __name__ == "__main__":
    main()