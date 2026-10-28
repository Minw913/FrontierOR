import argparse
import json
import math
import time
from collections import defaultdict

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def load_instance(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def build_greedy_heuristic(data):
    """
    Build a feasible (or near-feasible) heuristic solution:
    1) Assign customers to sites with per-site capacity Q.
    2) Pack active sites into vehicles with capacity Q.
    3) Build simple nearest-neighbor route per vehicle.
    """
    n = data["num_delivery_sites"]
    m = data["num_customers"]
    Q = data["vehicle_capacity"]
    K = data["num_vehicles"]

    demands = [int(c["demand"]) for c in data["customers"]]
    a_cost = data["assignment_cost_matrix"]
    r_cost = data["routing_cost_matrix"]

    customers = list(range(m))
    sites = list(range(1, n + 1))
    vehicles = list(range(K))

    # ---------- Customer -> site assignment with per-site cap ----------
    residual = {s: Q for s in sites}
    cust_to_site = {}
    site_demand = {s: 0 for s in sites}
    site_customers = defaultdict(list)

    # sort by descending demand, then hardest (cost spread) first
    def spread(c):
        row = a_cost[c]
        mn = min(row)
        mx = max(row)
        return mx - mn

    customers_sorted = sorted(customers, key=lambda c: (-demands[c], -spread(c)))

    for c in customers_sorted:
        d = demands[c]
        feasible_sites = [s for s in sites if residual[s] >= d]

        if not feasible_sites:
            # fallback (may become infeasible if instance itself infeasible)
            s_best = min(sites, key=lambda s: a_cost[c][s - 1])
        else:
            s_best = min(feasible_sites, key=lambda s: a_cost[c][s - 1])

        cust_to_site[c] = s_best
        site_demand[s_best] += d
        residual[s_best] -= d
        site_customers[s_best].append(c)

    active_sites = [s for s in sites if site_demand[s] > 0]

    # ---------- Pack active sites into vehicles ----------
    # Best-fit decreasing
    sites_sorted = sorted(active_sites, key=lambda s: -site_demand[s])

    veh_sites = {v: [] for v in vehicles}
    veh_load = {v: 0 for v in vehicles}
    used_vehicles = []

    for s in sites_sorted:
        d = site_demand[s]
        candidates = [v for v in used_vehicles if veh_load[v] + d <= Q]
        if candidates:
            # best fit: minimal remaining space after insert
            v_best = min(candidates, key=lambda v: Q - (veh_load[v] + d))
        else:
            # open new vehicle
            unopened = [v for v in vehicles if v not in used_vehicles]
            if unopened:
                v_best = unopened[0]
                used_vehicles.append(v_best)
            else:
                # forced fallback (should be rare if data is feasible)
                v_best = min(vehicles, key=lambda v: veh_load[v])

        veh_sites[v_best].append(s)
        veh_load[v_best] += d

    # ---------- Build route per used vehicle ----------
    veh_seq = {}
    veh_arcs = {}

    for v in used_vehicles:
        stop_sites = veh_sites[v]
        if not stop_sites:
            continue

        unvisited = set(stop_sites)
        cur = 0
        seq = [0]

        while unvisited:
            nxt = min(unvisited, key=lambda s: r_cost[cur][s])
            seq.append(nxt)
            unvisited.remove(nxt)
            cur = nxt

        seq.append(0)
        arcs = [(seq[i], seq[i + 1]) for i in range(len(seq) - 1)]

        veh_seq[v] = seq
        veh_arcs[v] = arcs

    # site -> vehicle
    site_to_vehicle = {}
    for v in used_vehicles:
        for s in veh_sites[v]:
            site_to_vehicle[s] = v

    # objective
    assign_obj = sum(a_cost[c][cust_to_site[c] - 1] for c in customers)
    route_obj = 0.0
    for v in used_vehicles:
        for i, j in veh_arcs[v]:
            route_obj += r_cost[i][j]

    obj = assign_obj + route_obj

    return {
        "objective": float(obj),
        "cust_to_site": cust_to_site,
        "site_demand": site_demand,
        "veh_sites": veh_sites,
        "veh_load": veh_load,
        "used_vehicles": used_vehicles,
        "veh_seq": veh_seq,
        "veh_arcs": veh_arcs,
        "site_to_vehicle": site_to_vehicle,
    }


def solve_instance(data, time_limit, logger=None):
    n = data["num_delivery_sites"]
    m = data["num_customers"]
    Q = data["vehicle_capacity"]
    K = data["num_vehicles"]

    # Internal indexing:
    # node 0 = depot, nodes 1..n = sites in given list order
    nodes = list(range(n + 1))
    sites = list(range(1, n + 1))
    customers = list(range(m))
    vehicles = list(range(K))

    demands = [int(c["demand"]) for c in data["customers"]]
    routing_cost = data["routing_cost_matrix"]        # (n+1)x(n+1)
    assign_cost = data["assignment_cost_matrix"]      # m x n

    # Heuristic for MIP start and fallback
    heuristic = build_greedy_heuristic(data)

    model = gp.Model("location_routing_pickup")
    model.Params.Threads = 1
    model.Params.TimeLimit = int(time_limit)
    model.Params.OutputFlag = 0

    # Variables
    x = model.addVars(
        ((v, i, j) for v in vehicles for i in nodes for j in nodes if i != j),
        vtype=GRB.BINARY,
        name="x",
    )
    z = model.addVars(vehicles, sites, vtype=GRB.BINARY, name="z")
    use = model.addVars(vehicles, vtype=GRB.BINARY, name="use")
    y = model.addVars(customers, sites, vtype=GRB.BINARY, name="y")
    D = model.addVars(sites, vtype=GRB.CONTINUOUS, lb=0.0, ub=Q, name="D")
    h = model.addVars(vehicles, sites, vtype=GRB.CONTINUOUS, lb=0.0, ub=Q, name="h")

    # flow only to non-depot destination (j in sites)
    F = model.addVars(
        ((v, i, j) for v in vehicles for i in nodes for j in sites if i != j),
        vtype=GRB.CONTINUOUS,
        lb=0.0,
        name="F",
    )

    # Objective
    routing_obj = gp.quicksum(
        routing_cost[i][j] * x[v, i, j]
        for v in vehicles for i in nodes for j in nodes if i != j
    )
    assign_obj = gp.quicksum(
        assign_cost[c][s - 1] * y[c, s]
        for c in customers for s in sites
    )
    model.setObjective(routing_obj + assign_obj, GRB.MINIMIZE)

    # Constraints

    # Each customer assigned to exactly one site
    for c in customers:
        model.addConstr(gp.quicksum(y[c, s] for s in sites) == 1, name=f"assign_{c}")

    # Site demand definition and bound
    for s in sites:
        model.addConstr(
            D[s] == gp.quicksum(demands[c] * y[c, s] for c in customers),
            name=f"demand_def_{s}",
        )
        model.addConstr(D[s] <= Q, name=f"site_cap_{s}")

    # Link assignment and site visit:
    # if any customer assigned -> some vehicle visits
    # if visited -> at least one customer assigned
    for s in sites:
        model.addConstr(
            gp.quicksum(y[c, s] for c in customers) <= m * gp.quicksum(z[v, s] for v in vehicles),
            name=f"assigned_implies_visit_{s}",
        )
        model.addConstr(
            gp.quicksum(y[c, s] for c in customers) >= gp.quicksum(z[v, s] for v in vehicles),
            name=f"visit_implies_assigned_{s}",
        )

    # Each site visited by at most one vehicle
    for s in sites:
        model.addConstr(gp.quicksum(z[v, s] for v in vehicles) <= 1, name=f"one_vehicle_site_{s}")

    # Vehicle route degree + use
    for v in vehicles:
        model.addConstr(gp.quicksum(x[v, 0, j] for j in sites) == use[v], name=f"dep_out_{v}")
        model.addConstr(gp.quicksum(x[v, i, 0] for i in sites) == use[v], name=f"dep_in_{v}")

        for s in sites:
            model.addConstr(
                gp.quicksum(x[v, s, j] for j in nodes if j != s) == z[v, s],
                name=f"out_deg_v{v}_s{s}",
            )
            model.addConstr(
                gp.quicksum(x[v, i, s] for i in nodes if i != s) == z[v, s],
                name=f"in_deg_v{v}_s{s}",
            )
            model.addConstr(z[v, s] <= use[v], name=f"z_use_v{v}_s{s}")

    # Demand allocation to vehicles for each site
    for s in sites:
        model.addConstr(gp.quicksum(h[v, s] for v in vehicles) == D[s], name=f"h_sum_{s}")
        for v in vehicles:
            model.addConstr(h[v, s] <= Q * z[v, s], name=f"h_link_v{v}_s{s}")

    # Vehicle capacity
    for v in vehicles:
        model.addConstr(gp.quicksum(h[v, s] for s in sites) <= Q * use[v], name=f"veh_cap_{v}")

    # Single-commodity flow constraints for connectivity and capacity
    for v in vehicles:
        # depot flow starts with total delivered quantity
        model.addConstr(
            gp.quicksum(F[v, 0, j] for j in sites) == gp.quicksum(h[v, s] for s in sites),
            name=f"flow_depot_{v}",
        )

        for s in sites:
            inflow = gp.quicksum(F[v, i, s] for i in nodes if i != s)
            outflow = gp.quicksum(F[v, s, j] for j in sites if j != s)
            model.addConstr(inflow - outflow == h[v, s], name=f"flow_cons_v{v}_s{s}")

    # Arc-flow capacity linking
    for v in vehicles:
        for i in nodes:
            for j in sites:
                if i == j:
                    continue
                model.addConstr(F[v, i, j] <= Q * x[v, i, j], name=f"flow_cap_v{v}_{i}_{j}")

    # Symmetry breaking for vehicle usage
    for v in range(K - 1):
        model.addConstr(use[v] >= use[v + 1], name=f"use_order_{v}")

    # ---------- MIP Start from heuristic ----------
    hs_c2s = heuristic["cust_to_site"]
    hs_site_d = heuristic["site_demand"]
    hs_site2v = heuristic["site_to_vehicle"]
    hs_used = set(heuristic["used_vehicles"])
    hs_arcs = heuristic["veh_arcs"]
    hs_seq = heuristic["veh_seq"]

    for c in customers:
        s = hs_c2s[c]
        y[c, s].Start = 1.0

    for s in sites:
        D[s].Start = float(hs_site_d.get(s, 0.0))

    for v in vehicles:
        use[v].Start = 1.0 if v in hs_used else 0.0
        for s in sites:
            served = (hs_site2v.get(s, None) == v)
            z[v, s].Start = 1.0 if served else 0.0
            h[v, s].Start = float(hs_site_d[s]) if served else 0.0

    for v in hs_used:
        for i, j in hs_arcs.get(v, []):
            x[v, i, j].Start = 1.0

        # set flow start along route
        seq = hs_seq.get(v, [0, 0])
        rem = sum(hs_site_d[s] for s in heuristic["veh_sites"][v])
        for idx in range(len(seq) - 1):
            i, j = seq[idx], seq[idx + 1]
            if j != 0 and i != j:
                F[v, i, j].Start = float(rem)
            if j != 0:
                rem -= hs_site_d[j]

    # ---------- Logging callback ----------
    model._logger = logger
    model._best_logged = float("inf")

    def cb(m, where):
        if where == GRB.Callback.MIPSOL and m._logger is not None:
            obj = m.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj + 1e-9 < m._best_logged:
                m._best_logged = obj
                m._logger.log(float(obj))

    model.optimize(cb)

    if model.SolCount == 0:
        # fallback to heuristic output
        return None, heuristic

    return model, heuristic


def extract_solution(data, model, heuristic):
    n = data["num_delivery_sites"]
    m = data["num_customers"]
    K = data["num_vehicles"]

    depot_id = data["depot"]["id"]
    site_ids = [s["id"] for s in data["delivery_sites"]]
    customer_ids = [c["id"] for c in data["customers"]]

    def node_id(internal_node):
        if internal_node == 0:
            return depot_id
        return site_ids[internal_node - 1]

    routes = {}
    assignments = {}
    customer_tours = {}

    if model is None:
        # Use heuristic
        c2s = heuristic["cust_to_site"]
        s2v = heuristic["site_to_vehicle"]
        veh_seq = heuristic["veh_seq"]

        # Routes
        for v, seq in veh_seq.items():
            if len(seq) <= 2:
                continue
            arcs_internal = [(seq[i], seq[i + 1]) for i in range(len(seq) - 1)]
            routes[str(v)] = {
                "arcs": [[node_id(i), node_id(j)] for i, j in arcs_internal],
                "sequence": [node_id(i) for i in seq],
            }

        # Customer assignment + tour
        for c in range(m):
            s = c2s[c]
            cid = str(customer_ids[c])
            sid = node_id(s)
            assignments[cid] = sid
            v = s2v.get(s, 0)
            customer_tours[cid] = {"vehicle": int(v), "site": sid}

        objective_value = float(heuristic["objective"])
        return {
            "objective_value": objective_value,
            "routes": routes,
            "assignments": assignments,
            "customer_tours": customer_tours,
        }

    # Pull variable values from model
    x_val = model.getAttr("X", model.getVars())
    # Better access by name via tuples for clarity
    # We'll fetch from tupledicts by scanning var names is expensive; rebuild through getVarByName would be slow.
    # Instead use model variables through structured extraction below using getAttr on model vars impossible by dict.
    # We'll use model.getVars() only if needed. Simpler: parse via variable names from model.
    # But we can directly evaluate by name if small; here we do robust explicit loop through model vars map.
    var_map = {v.VarName: v.X for v in model.getVars()}

    # Determine site->vehicle from z
    site_to_vehicle = {}
    for s in range(1, n + 1):
        best_v = None
        best_val = -1.0
        for v in range(K):
            val = var_map.get(f"z[{v},{s}]", 0.0)
            if val > best_val:
                best_val = val
                best_v = v
        if best_val > 0.5:
            site_to_vehicle[s] = best_v

    # Assignments
    for c in range(m):
        best_s = 1
        best_val = -1.0
        for s in range(1, n + 1):
            val = var_map.get(f"y[{c},{s}]", 0.0)
            if val > best_val:
                best_val = val
                best_s = s
        cid = str(customer_ids[c])
        sid = node_id(best_s)
        assignments[cid] = sid
        v = site_to_vehicle.get(best_s, 0)
        customer_tours[cid] = {"vehicle": int(v), "site": sid}

    # Routes
    for v in range(K):
        use_val = var_map.get(f"use[{v}]", 0.0)
        if use_val < 0.5:
            continue

        arcs_internal = []
        succ = {}

        for i in range(0, n + 1):
            for j in range(0, n + 1):
                if i == j:
                    continue
                val = var_map.get(f"x[{v},{i},{j}]", 0.0)
                if val > 0.5:
                    arcs_internal.append((i, j))
                    succ[i] = j

        if not arcs_internal:
            continue

        # Build main sequence from depot
        seq = [0]
        cur = 0
        seen = {0}
        for _ in range(n + 2):
            if cur not in succ:
                break
            nxt = succ[cur]
            seq.append(nxt)
            if nxt == 0:
                break
            if nxt in seen:
                break
            seen.add(nxt)
            cur = nxt

        if seq[-1] != 0:
            seq.append(0)

        routes[str(v)] = {
            "arcs": [[node_id(i), node_id(j)] for i, j in arcs_internal],
            "sequence": [node_id(i) for i in seq],
        }

    return {
        "objective_value": float(model.ObjVal),
        "routes": routes,
        "assignments": assignments,
        "customer_tours": customer_tours,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    data = load_instance(args.instance_path)
    model, heuristic = solve_instance(data, args.time_limit, logger=logger)
    solution = extract_solution(data, model, heuristic)

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()