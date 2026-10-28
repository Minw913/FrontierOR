import argparse
import json
import math
from collections import defaultdict

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def build_and_solve(instance, time_limit, logger=None):
    n = int(instance["dimensions"]["n"])
    m = int(instance["dimensions"]["m"])
    p = int(instance["dimensions"]["p"])

    demands = instance["demands"]  # [n][p]
    inventory_costs = instance["inventory_costs"]  # [n]
    processing_time = float(instance["processing_time"])
    setup_times = instance["setup_times"]  # [n][n][m]
    setup_costs = instance["setup_costs"]  # [n][n][m]
    machine_capacities = instance["machine_capacities"]  # [m][p]
    max_production = instance["max_production"]  # [n][p]
    max_setups_per_item = instance["max_setups_per_item"]  # [n][m][p]
    minimum_lot_sizes = instance["minimum_lot_sizes"]  # [n]
    machine_eligibility = instance["machine_eligibility"]  # [n][m]

    model = gp.Model("parallel_machine_lot_sizing_sd_setup")
    model.Params.Threads = 1
    model.Params.TimeLimit = int(time_limit)
    model.Params.OutputFlag = 0

    # -------------------------
    # Decision variables
    # -------------------------
    q = {}   # production qty of item i on machine mm in period t
    z = {}   # start-of-period setup state
    e = {}   # end-of-period setup state
    u = {}   # MTZ ordering variable
    inv = {} # inventory of item i at end of period t

    for i in range(n):
        for mm in range(m):
            elig = 1 if machine_eligibility[i][mm] == 1 else 0
            for t in range(p):
                q[(i, mm, t)] = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"q_{i}_{mm}_{t}")
                z[(i, mm, t)] = model.addVar(lb=0.0, ub=elig, vtype=GRB.BINARY, name=f"z_{i}_{mm}_{t}")
                e[(i, mm, t)] = model.addVar(lb=0.0, ub=elig, vtype=GRB.BINARY, name=f"e_{i}_{mm}_{t}")
                u[(i, mm, t)] = model.addVar(lb=0.0, ub=float(n), vtype=GRB.CONTINUOUS, name=f"u_{i}_{mm}_{t}")

    for i in range(n):
        for t in range(p):
            inv[(i, t)] = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"inv_{i}_{t}")

    # Setup transition arcs y[j,i,mm,t] (from item j to item i)
    y = {}
    incoming_arcs = defaultdict(list)  # key: (i,mm,t) -> list of arc keys
    outgoing_arcs = defaultdict(list)  # key: (i,mm,t) -> list of arc keys

    for mm in range(m):
        eligible_items = [i for i in range(n) if machine_eligibility[i][mm] == 1]
        for t in range(p):
            for j in eligible_items:
                for i in eligible_items:
                    if i == j:
                        continue
                    # If max setups into i is 0 this period, entering i via setup is forbidden.
                    if max_setups_per_item[i][mm][t] <= 0:
                        continue
                    key = (j, i, mm, t)
                    y[key] = model.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY, name=f"y_{j}_{i}_{mm}_{t}")
                    incoming_arcs[(i, mm, t)].append(key)
                    outgoing_arcs[(j, mm, t)].append(key)

    model.update()

    # -------------------------
    # Constraints
    # -------------------------

    # Exactly one start and one end setup state per machine and period
    for mm in range(m):
        for t in range(p):
            model.addConstr(
                gp.quicksum(z[(i, mm, t)] for i in range(n)) == 1,
                name=f"one_start_{mm}_{t}",
            )
            model.addConstr(
                gp.quicksum(e[(i, mm, t)] for i in range(n)) == 1,
                name=f"one_end_{mm}_{t}",
            )

    # Carry-over between consecutive periods
    for mm in range(m):
        for t in range(1, p):
            for i in range(n):
                model.addConstr(
                    z[(i, mm, t)] == e[(i, mm, t - 1)],
                    name=f"carry_{i}_{mm}_{t}",
                )

    # Sequence flow, degree limits, production/setup linking, lot sizes
    for mm in range(m):
        for t in range(p):
            for i in range(n):
                in_expr = gp.quicksum(y[a] for a in incoming_arcs[(i, mm, t)]) if incoming_arcs[(i, mm, t)] else 0.0
                out_expr = gp.quicksum(y[a] for a in outgoing_arcs[(i, mm, t)]) if outgoing_arcs[(i, mm, t)] else 0.0

                # Flow conservation from start state to end state
                model.addConstr(
                    z[(i, mm, t)] + in_expr == e[(i, mm, t)] + out_expr,
                    name=f"flow_{i}_{mm}_{t}",
                )

                # At most one entering and one leaving transition per item/machine/period
                model.addConstr(in_expr <= 1, name=f"in_deg_{i}_{mm}_{t}")
                model.addConstr(out_expr <= 1, name=f"out_deg_{i}_{mm}_{t}")

                # Prevent re-entering the start item in the same period
                model.addConstr(in_expr + z[(i, mm, t)] <= 1, name=f"no_reenter_start_{i}_{mm}_{t}")

                # Setup count cap into item
                model.addConstr(
                    in_expr <= float(max_setups_per_item[i][mm][t]),
                    name=f"max_setup_in_{i}_{mm}_{t}",
                )

                # Can only produce if machine is in setup for i (either carry-in z or entered via setup)
                setup_state_expr = z[(i, mm, t)] + in_expr
                model.addConstr(
                    q[(i, mm, t)] <= float(max_production[i][t]) * setup_state_expr,
                    name=f"prod_if_setup_{i}_{mm}_{t}",
                )

                # If item is produced because of a setup transition into it, enforce min lot size
                model.addConstr(
                    q[(i, mm, t)] >= float(minimum_lot_sizes[i]) * in_expr,
                    name=f"min_lot_if_setup_{i}_{mm}_{t}",
                )

                # If max_setups is zero, item cannot be produced in this machine-period
                can_produce_here = 1 if (machine_eligibility[i][mm] == 1 and max_setups_per_item[i][mm][t] > 0) else 0
                model.addConstr(
                    q[(i, mm, t)] <= float(max_production[i][t]) * can_produce_here,
                    name=f"forbid_prod_{i}_{mm}_{t}",
                )

                # MTZ anchor: start node has order 0
                model.addConstr(
                    u[(i, mm, t)] <= float(n) * (1 - z[(i, mm, t)]),
                    name=f"mtz_anchor_{i}_{mm}_{t}",
                )

            # Machine-time capacity (processing + setup times)
            proc_time = processing_time * gp.quicksum(q[(i, mm, t)] for i in range(n))
            setup_time = gp.quicksum(setup_times[j][i][mm] * y[(j, i, mm, t)] for (j, i, mm2, t2) in y if mm2 == mm and t2 == t)
            model.addConstr(
                proc_time + setup_time <= float(machine_capacities[mm][t]),
                name=f"cap_{mm}_{t}",
            )

    # MTZ subtour elimination on each setup arc
    for (j, i, mm, t), yvar in y.items():
        model.addConstr(
            u[(i, mm, t)] >= u[(j, mm, t)] + 1 - float(n) * (1 - yvar),
            name=f"mtz_{j}_{i}_{mm}_{t}",
        )

    # Inventory balance and item-period production limits
    for i in range(n):
        for t in range(p):
            prod_it = gp.quicksum(q[(i, mm, t)] for mm in range(m))
            if t == 0:
                model.addConstr(
                    prod_it == float(demands[i][t]) + inv[(i, t)],
                    name=f"inv_bal_{i}_{t}",
                )
            else:
                model.addConstr(
                    inv[(i, t - 1)] + prod_it == float(demands[i][t]) + inv[(i, t)],
                    name=f"inv_bal_{i}_{t}",
                )

            model.addConstr(
                prod_it <= float(max_production[i][t]),
                name=f"max_prod_item_{i}_{t}",
            )

    # -------------------------
    # Objective
    # -------------------------
    inv_cost_expr = gp.quicksum(inventory_costs[i] * inv[(i, t)] for i in range(n) for t in range(p))
    setup_cost_expr = gp.quicksum(setup_costs[j][i][mm] * y[(j, i, mm, t)] for (j, i, mm, t) in y)
    model.setObjective(inv_cost_expr + setup_cost_expr, GRB.MINIMIZE)

    # -------------------------
    # Incumbent logging callback
    # -------------------------
    best_logged = [math.inf]

    def cb(mdl, where):
        if logger is None:
            return
        if where == GRB.Callback.MIPSOL:
            obj = mdl.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj + 1e-9 < best_logged[0]:
                best_logged[0] = obj
                logger.log(float(obj))

    model.optimize(cb)

    if model.SolCount == 0:
        return None, None, None, None

    if logger is not None and model.ObjVal + 1e-9 < best_logged[0]:
        logger.log(float(model.ObjVal))

    # Extract solution values
    q_val = {}
    for i in range(n):
        for mm in range(m):
            for t in range(p):
                v = q[(i, mm, t)].X
                if v > 1e-8:
                    q_val[(i, mm, t)] = float(v)

    y_val = {}
    for key, var in y.items():
        v = var.X
        if v > 0.5:
            y_val[key] = int(round(v))

    z_val = {}
    for i in range(n):
        for mm in range(m):
            for t in range(p):
                z_val[(i, mm, t)] = int(round(z[(i, mm, t)].X))

    return float(model.ObjVal), q_val, y_val, z_val


def allocate_production_to_demands(n, m, p, demands, q_val):
    """
    Build x_{item}_{machine}_{production_period}_{demand_period}
    from machine-period production q by greedy earliest-demand assignment.
    """
    production = {}
    tol = 1e-8

    for i in range(n):
        rem = [float(demands[i][u]) for u in range(p)]
        for t in range(p):
            for mm in range(m):
                amt = q_val.get((i, mm, t), 0.0)
                if amt <= tol:
                    continue
                u = t
                while amt > tol and u < p:
                    if rem[u] > tol:
                        a = min(amt, rem[u])
                        key = f"x_{i}_{mm}_{t}_{u}"
                        production[key] = production.get(key, 0.0) + float(a)
                        amt -= a
                        rem[u] -= a
                    else:
                        u += 1
                if amt > 1e-6:
                    # Numerical fallback: place residual on last feasible demand period.
                    key = f"x_{i}_{mm}_{t}_{p-1}"
                    production[key] = production.get(key, 0.0) + float(amt)
                    amt = 0.0

    # Clean tiny values
    production = {k: float(v) for k, v in production.items() if v > 1e-8}
    return production


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    n = int(instance["dimensions"]["n"])
    m = int(instance["dimensions"]["m"])
    p = int(instance["dimensions"]["p"])
    demands = instance["demands"]

    obj, q_val, y_val, z_val = build_and_solve(instance, args.time_limit, logger=logger)

    if obj is None:
        solution = {
            "objective_value": 1e30,
            "production": {},
            "setups": {},
            "carryover": {}
        }
    else:
        production = allocate_production_to_demands(n, m, p, demands, q_val)

        setups = {}
        for (j, i, mm, t), v in y_val.items():
            if v != 0:
                setups[f"y_{j}_{i}_{mm}_{t}"] = int(v)

        carryover = {}
        # Include all carryover keys explicitly
        for i in range(n):
            for mm in range(m):
                for t in range(p):
                    carryover[f"z_{i}_{mm}_{t}"] = int(z_val[(i, mm, t)])

        solution = {
            "objective_value": float(obj),
            "production": production,
            "setups": setups,
            "carryover": carryover
        }

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()