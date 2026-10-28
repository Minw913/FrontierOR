import argparse
import json
import time
from collections import defaultdict

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def build_and_solve(instance, time_limit, logger=None):
    start_time = time.time()

    num_customers = instance["num_customers"]
    num_nodes = instance["num_nodes"]  # depot + customers
    depot = instance["depot_index"]
    customers = list(instance["customer_indices"])

    params = instance["parameters"]
    num_drones = params["num_drones"]
    drone_caps = params["drone_battery_capacities"]

    truck_time = instance["truck_travel_times"]
    drone_batt = instance["drone_battery_consumption"]
    drone_time = instance["drone_delivery_time"]

    sink = num_nodes  # virtual sink index required by output schema
    launch_nodes = [depot] + customers
    drones = list(range(num_drones))

    # ----- Build truck arc set (path from depot to sink through subset of customers) -----
    arcs = []
    # depot -> customer/sink
    for j in customers:
        arcs.append((depot, j))
    arcs.append((depot, sink))
    # customer -> customer/sink
    for i in customers:
        for j in customers:
            if i != j:
                arcs.append((i, j))
        arcs.append((i, sink))

    # Arc costs
    arc_cost = {}
    for (i, j) in arcs:
        if j == sink:
            arc_cost[(i, j)] = truck_time[i][depot]  # virtual sink same as depot
        else:
            arc_cost[(i, j)] = truck_time[i][j]

    # ----- Build feasible drone assignment keys -----
    # z[k,l,c] = 1 if drone k dispatched from launch node l serves customer c
    # Skip impossible single-trip assignments (consumption > capacity).
    z_keys = []
    z_by_customer = defaultdict(list)
    z_by_drone = defaultdict(list)
    z_by_drone_launch = defaultdict(list)
    z_by_launch = defaultdict(list)

    for k in drones:
        cap_k = drone_caps[k]
        for l in launch_nodes:
            for c in customers:
                # If l is same customer c, this is impossible because:
                # - if truck visits c then c served by truck (coverage), not drone
                # - if truck doesn't visit c then cannot dispatch from c
                if l == c and l in customers:
                    continue
                cons = drone_batt[k][l][c]
                if cons <= cap_k:
                    key = (k, l, c)
                    z_keys.append(key)
                    z_by_customer[c].append(key)
                    z_by_drone[k].append(key)
                    z_by_drone_launch[(k, l)].append(key)
                    z_by_launch[l].append(key)

    # ----- Model -----
    m = gp.Model("truck_drones")

    x = m.addVars(arcs, vtype=GRB.BINARY, name="x")
    y = m.addVars(customers, vtype=GRB.BINARY, name="y")
    z = m.addVars(z_keys, vtype=GRB.BINARY, name="z")
    w = m.addVars(launch_nodes, lb=0.0, vtype=GRB.CONTINUOUS, name="w")
    # MTZ order variables for optional customers
    U = num_customers
    u = m.addVars(customers, lb=0.0, ub=float(U), vtype=GRB.CONTINUOUS, name="u")

    # ----- Truck routing constraints -----
    m.addConstr(gp.quicksum(x[depot, j] for j in customers + [sink] if (depot, j) in x) == 1, name="start_once")
    m.addConstr(gp.quicksum(x[i, sink] for i in [depot] + customers if (i, sink) in x) == 1, name="end_once")

    for c in customers:
        in_expr = gp.quicksum(x[i, c] for i in [depot] + customers if i != c and (i, c) in x)
        out_expr = gp.quicksum(x[c, j] for j in customers + [sink] if j != c and (c, j) in x)
        m.addConstr(in_expr == y[c], name=f"in_{c}")
        m.addConstr(out_expr == y[c], name=f"out_{c}")

    # MTZ subtour elimination on visited customers
    for i in customers:
        m.addConstr(u[i] >= y[i], name=f"u_lb_{i}")
        m.addConstr(u[i] <= U * y[i], name=f"u_ub_{i}")

    for i in customers:
        for j in customers:
            if i == j:
                continue
            if (i, j) in x:
                m.addConstr(
                    u[i] - u[j] + U * x[i, j] <= U - 1 + U * (2 - y[i] - y[j]),
                    name=f"mtz_{i}_{j}"
                )

    # ----- Service coverage -----
    for c in customers:
        m.addConstr(
            y[c] + gp.quicksum(z[key] for key in z_by_customer[c]) == 1,
            name=f"cover_{c}"
        )

    # Drone can dispatch from customer launch node only if truck visits it
    for (k, l, c) in z_keys:
        if l in y:
            m.addConstr(z[k, l, c] <= y[l], name=f"launch_visit_{k}_{l}_{c}")

    # Battery per drone
    for k in drones:
        m.addConstr(
            gp.quicksum(drone_batt[k][l][c] * z[k, l, c] for (kk, l, c) in z_by_drone[k] for _ in [0] if kk == k)
            <= drone_caps[k],
            name=f"battery_{k}"
        )

    # Waiting times: for each launch node, wait must dominate each drone's total time from that node
    max_single_time = 0
    for k in drones:
        for l in launch_nodes:
            for c in customers:
                max_single_time = max(max_single_time, drone_time[k][l][c])
    big_wait = max_single_time * max(1, num_customers)

    for l in launch_nodes:
        for k in drones:
            keys = z_by_drone_launch.get((k, l), [])
            if keys:
                m.addConstr(
                    gp.quicksum(drone_time[k][l2][c] * z[k, l2, c] for (k2, l2, c) in keys for _ in [0] if k2 == k and l2 == l)
                    <= w[l],
                    name=f"wait_{l}_{k}"
                )
            else:
                # no assignment variables for (k,l), no need constraint
                pass
        if l in y:
            m.addConstr(w[l] <= big_wait * y[l], name=f"wait_zero_if_not_visited_{l}")

    # Objective
    truck_cost_expr = gp.quicksum(arc_cost[a] * x[a] for a in arcs)
    wait_cost_expr = gp.quicksum(w[l] for l in launch_nodes)
    m.setObjective(truck_cost_expr + wait_cost_expr, GRB.MINIMIZE)

    # Required Gurobi params
    m.Params.Seed = 0
    m.Params.MIPGap = 1e-4
    m.Params.NumericFocus = 0
    m.Params.Threads = 1

    elapsed = time.time() - start_time
    remaining = max(1, int(time_limit - elapsed))
    m.Params.TimeLimit = remaining

    # ----- MIP start: all customers served by truck in nearest-neighbor order -----
    if customers:
        rem = set(customers)
        cur = depot
        seq = []
        while rem:
            nxt = min(rem, key=lambda c: truck_time[cur][c])
            seq.append(nxt)
            rem.remove(nxt)
            cur = nxt

        # y start
        for c in customers:
            y[c].Start = 1.0

        # x start
        if seq:
            first = seq[0]
            if (depot, first) in x:
                x[depot, first].Start = 1.0
            for i in range(len(seq) - 1):
                a = (seq[i], seq[i + 1])
                if a in x:
                    x[a].Start = 1.0
            last = seq[-1]
            if (last, sink) in x:
                x[last, sink].Start = 1.0
        else:
            x[depot, sink].Start = 1.0

        # u start
        for pos, c in enumerate(seq, start=1):
            u[c].Start = float(pos)

    for l in launch_nodes:
        w[l].Start = 0.0
    for key in z_keys:
        z[key].Start = 0.0

    # ----- Helpers for extracting solution -----
    def extract_solution_from_values(xv, yv, zv, wv, obj_val):
        truck_route = []
        for (i, j) in arcs:
            val = xv[(i, j)] if isinstance(xv, dict) else xv[i, j]
            if val > 0.5:
                truck_route.append([int(i), int(j)])

        drone_assignments = []
        for (k, l, c) in z_keys:
            val = zv[(k, l, c)] if isinstance(zv, dict) else zv[k, l, c]
            if val > 0.5:
                drone_assignments.append({
                    "drone": int(k),
                    "dispatch_node": int(l),
                    "delivery_node": int(c),
                })

        visited = {depot}
        for c in customers:
            val = yv[c] if isinstance(yv, dict) else yv[c]
            if val > 0.5:
                visited.add(c)

        waiting_times = {}
        for l in visited:
            val = wv[l] if isinstance(wv, dict) else wv[l]
            waiting_times[str(int(l))] = float(val)

        return {
            "objective_value": float(obj_val),
            "truck_route": truck_route,
            "drone_assignments": drone_assignments,
            "waiting_times": waiting_times,
        }

    # ----- Incumbent logging callback -----
    m._x = x
    m._y = y
    m._z = z
    m._w = w
    m._extract_solution_from_values = extract_solution_from_values
    m._logger = logger
    m._arcs = arcs
    m._z_keys = z_keys

    def cb(model, where):
        if where == GRB.Callback.MIPSOL and model._logger is not None:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            xv = model.cbGetSolution(model._x)
            yv = model.cbGetSolution(model._y)
            zv = model.cbGetSolution(model._z)
            wv = model.cbGetSolution(model._w)
            sol = model._extract_solution_from_values(xv, yv, zv, wv, obj)
            model._logger.log_solution(float(obj), sol)

    m.optimize(cb)

    if m.SolCount == 0:
        # Extremely unlikely for this model; provide safe empty fallback
        return {
            "objective_value": float("inf"),
            "truck_route": [],
            "drone_assignments": [],
            "waiting_times": {str(depot): 0.0},
        }

    final_sol = extract_solution_from_values(
        {(i, j): x[i, j].X for (i, j) in arcs},
        {c: y[c].X for c in customers},
        {(k, l, c): z[k, l, c].X for (k, l, c) in z_keys},
        {l: w[l].X for l in launch_nodes},
        m.ObjVal
    )

    return final_sol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    solution = build_and_solve(instance, args.time_limit, logger=logger)

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()