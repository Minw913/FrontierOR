import argparse
import json
import time
from collections import defaultdict, Counter

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def build_and_solve(instance, time_limit, logger=None):
    t_start = time.time()

    network = instance["network"]
    time_params = instance["time_parameters"]
    demand_data = instance["demand"]
    fleet = instance["fleet"]

    T = int(time_params["time_horizon_T"])  # time steps 0..T
    trans_times = range(T)                  # transition steps 0..T-1
    all_times = range(T + 1)

    nodes = network["nodes"]
    links = network["links"]

    node_type = {str(n["id"]): n["type"] for n in nodes}
    centroid_ids = [str(n["id"]) for n in nodes if n["type"] == "centroid"]

    L = list(range(len(links)))
    link_from = {}
    link_to = {}
    link_type = {}
    tau_ff = {}
    tau_w = {}
    cap_step = {}
    jam_occ = {}
    is_road = {}
    is_outgoing_conn = {}
    is_incoming_conn = {}
    incoming_conn_centroid = {}

    dt_sec = int(time_params["time_step_duration_seconds"])
    for li, lk in enumerate(links):
        u = str(lk["from"])
        v = str(lk["to"])
        typ = lk["type"]

        link_from[li] = u
        link_to[li] = v
        link_type[li] = typ
        tau_ff[li] = int(lk["free_flow_travel_time_steps"])
        tau_w[li] = int(lk.get("congested_travel_time_steps", 1))
        jam_occ[li] = float(lk.get("jam_density_vehicles", 1e9))
        cap_vph = float(lk.get("capacity_vph", 1e9))
        cap_step[li] = cap_vph * dt_sec / 3600.0

        road = (typ == "road")
        outc = (typ == "centroid_connector" and node_type.get(u) == "centroid")
        inc = (typ == "centroid_connector" and node_type.get(v) == "centroid")

        is_road[li] = road
        is_outgoing_conn[li] = outc
        is_incoming_conn[li] = inc
        if inc:
            incoming_conn_centroid[li] = v

    # Node-link incidence
    links_from_node = defaultdict(list)
    links_to_node = defaultdict(list)
    for li in L:
        links_from_node[link_from[li]].append(li)
        links_to_node[link_to[li]].append(li)

    # Successor and predecessor links
    succ_links = {}
    pred_links = {}
    for li in L:
        # vehicles do not pass through incoming centroid connectors
        if is_incoming_conn[li]:
            succ_links[li] = []
        else:
            succ_links[li] = list(links_from_node.get(link_to[li], []))

        # outgoing centroid connectors are fed by centroid parking, not predecessor links
        if is_outgoing_conn[li]:
            pred_links[li] = []
        else:
            pred_links[li] = list(links_to_node.get(link_from[li], []))

    outgoing_links_by_centroid = defaultdict(list)
    incoming_links_by_centroid = defaultdict(list)
    for li in L:
        if is_outgoing_conn[li]:
            outgoing_links_by_centroid[link_from[li]].append(li)
        if is_incoming_conn[li]:
            incoming_links_by_centroid[link_to[li]].append(li)

    # Demand aggregation by (o,d,t)
    demand_t = defaultdict(float)
    for od in demand_data["od_pairs"]:
        o = str(od["origin"])
        d = str(od["destination"])
        dep_times = od.get("departure_times", [])
        cnt = Counter(int(tt) for tt in dep_times)
        for tt, c in cnt.items():
            if 0 <= tt < T:
                demand_t[(o, d, tt)] += float(c)

    # OD set for waiting/pickup variables
    OD = [(o, d) for o in centroid_ids for d in centroid_ids if o != d]

    model = gp.Model("SAV_LTM")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = max(1, int(time_limit))
    model.Params.Threads = 1

    # Decision vars
    # y: turning flows from link l to successor m for destination d at time t
    y_index = []
    for l in L:
        for m in succ_links[l]:
            for d in centroid_ids:
                # incoming centroid connector can only receive matching destination
                if is_incoming_conn[m] and incoming_conn_centroid[m] != d:
                    continue
                for t in trans_times:
                    y_index.append((l, m, d, t))
    y = model.addVars(y_index, lb=0.0, name="y")

    # y_centroid: departures from centroid along outgoing connector
    yc_index = []
    for l in L:
        if is_outgoing_conn[l]:
            for d in centroid_ids:
                for t in trans_times:
                    yc_index.append((l, d, t))
    y_centroid = model.addVars(yc_index, lb=0.0, name="y_centroid")

    # Cumulative counts
    N_U = model.addVars(L, centroid_ids, all_times, lb=0.0, name="N_U")
    N_D = model.addVars(L, centroid_ids, all_times, lb=0.0, name="N_D")

    # Parking
    p = model.addVars(centroid_ids, all_times, lb=0.0, name="p")

    # Pickup and waiting
    e = model.addVars(OD, trans_times, lb=0.0, name="e")
    omega = model.addVars(OD, all_times, lb=0.0, name="omega")

    # Initial conditions for cumulative counts
    for l in L:
        for d in centroid_ids:
            model.addConstr(N_U[l, d, 0] == 0.0)
            model.addConstr(N_D[l, d, 0] == 0.0)

    # Link flow dynamics
    for l in L:
        for d in centroid_ids:
            for t in trans_times:
                if is_outgoing_conn[l]:
                    inflow_expr = y_centroid[l, d, t]
                else:
                    inflow_expr = y.sum("*", l, d, t)

                model.addConstr(N_U[l, d, t + 1] == N_U[l, d, t] + inflow_expr)

                if is_incoming_conn[l]:
                    # Special incoming connector rule
                    model.addConstr(N_D[l, d, t + 1] == N_U[l, d, t])
                else:
                    outflow_expr = y.sum(l, "*", d, t)
                    model.addConstr(N_D[l, d, t + 1] == N_D[l, d, t] + outflow_expr)

            for t in all_times:
                model.addConstr(N_U[l, d, t] >= N_D[l, d, t])

    # Sending-flow constraints for road + outgoing centroid connectors
    for l in L:
        if not (is_road[l] or is_outgoing_conn[l]):
            continue
        tau = tau_ff[l]
        for d in centroid_ids:
            for t in trans_times:
                outflow_expr = y.sum(l, "*", d, t)
                if t < tau:
                    model.addConstr(outflow_expr == 0.0)
                tt = t - tau + 1
                rhs = (N_U[l, d, tt] if tt >= 0 else 0.0) - N_D[l, d, t]
                model.addConstr(outflow_expr <= rhs)

    # Road receiving/capacity constraints
    for l in L:
        if not is_road[l]:
            continue
        cap = cap_step[l]
        wtau = tau_w[l]
        jam = jam_occ[l]

        for t in trans_times:
            out_tot = gp.quicksum(y.sum(l, "*", d, t) for d in centroid_ids)
            in_tot = gp.quicksum(y.sum("*", l, d, t) for d in centroid_ids)

            model.addConstr(out_tot <= cap)
            model.addConstr(in_tot <= cap)

            tt = t - wtau + 1
            wave_rhs = gp.quicksum(
                (N_U[l, d, tt] if tt >= 0 else 0.0) - N_D[l, d, t]
                for d in centroid_ids
            ) + jam
            model.addConstr(in_tot <= wave_rhs)

    # Parking initial condition
    init_dist = {str(k): float(v) for k, v in fleet["initial_distribution"].items()}
    for c in centroid_ids:
        model.addConstr(p[c, 0] == init_dist.get(c, 0.0))

    # Parking evolution and departure limits
    for c in centroid_ids:
        out_conns = outgoing_links_by_centroid.get(c, [])
        in_conns = incoming_links_by_centroid.get(c, [])

        for t in trans_times:
            arrivals = gp.quicksum(N_D[l, c, t + 1] - N_D[l, c, t] for l in in_conns)
            departures = gp.quicksum(y_centroid[l, d, t] for l in out_conns for d in centroid_ids)

            model.addConstr(p[c, t + 1] == p[c, t] + arrivals - departures)
            model.addConstr(departures <= p[c, t])

    # Fleet conservation at final time
    init_total = sum(init_dist.get(c, 0.0) for c in centroid_ids)
    model.addConstr(gp.quicksum(p[c, T] for c in centroid_ids) == init_total)

    # Waiting/pickup dynamics
    for (o, d) in OD:
        out_conns_o = outgoing_links_by_centroid.get(o, [])

        model.addConstr(omega[o, d, 0] == 0.0)
        for t in trans_times:
            dem = demand_t.get((o, d, t), 0.0)
            depart_od = gp.quicksum(y_centroid[l, d, t] for l in out_conns_o)

            model.addConstr(omega[o, d, t + 1] == omega[o, d, t] + dem - e[o, d, t])
            model.addConstr(e[o, d, t] <= omega[o, d, t] + dem)
            model.addConstr(e[o, d, t] <= depart_od)

        model.addConstr(omega[o, d, T] == 0.0)

    # Objective: vehicle time + waiting time
    veh_time = gp.quicksum(N_U[l, d, t] - N_D[l, d, t] for l in L for d in centroid_ids for t in all_times)
    wait_time = gp.quicksum(omega[o, d, t] for (o, d) in OD for t in all_times)
    model.setObjective(veh_time + wait_time, GRB.MINIMIZE)

    # Adjust remaining time limit after build
    elapsed_build = time.time() - t_start
    remaining = max(1, int(time_limit - elapsed_build))
    model.Params.TimeLimit = remaining

    model.optimize()

    solved = model.SolCount > 0
    if solved and logger:
        logger.log(float(model.ObjVal))

    # Build output
    sol = {
        "objective_value": float(model.ObjVal) if solved else 0.0,
        "y": {},
        "y_centroid": {},
        "N_U": {},
        "N_D": {},
        "p": {},
        "e": {},
        "omega": {}
    }

    if not solved:
        return sol

    EPS = 1e-7

    for (l, m, d, t) in y_index:
        v = y[l, m, d, t].X
        if abs(v) > EPS:
            key = f"{link_from[l]}|{link_to[l]}|{link_to[m]}|{d}|{t}"
            sol["y"][key] = float(v)

    for (l, d, t) in yc_index:
        v = y_centroid[l, d, t].X
        if abs(v) > EPS:
            key = f"{link_from[l]}|{link_to[l]}|{d}|{t}"
            sol["y_centroid"][key] = float(v)

    for l in L:
        for d in centroid_ids:
            for t in all_times:
                vu = N_U[l, d, t].X
                vd = N_D[l, d, t].X
                if abs(vu) > EPS:
                    key_u = f"{link_from[l]}|{link_to[l]}|{d}|{t}"
                    sol["N_U"][key_u] = float(vu)
                if abs(vd) > EPS:
                    key_d = f"{link_from[l]}|{link_to[l]}|{d}|{t}"
                    sol["N_D"][key_d] = float(vd)

    for c in centroid_ids:
        for t in all_times:
            v = p[c, t].X
            if abs(v) > EPS:
                key = f"{c}|{t}"
                sol["p"][key] = float(v)

    for (o, d) in OD:
        for t in trans_times:
            v = e[o, d, t].X
            if abs(v) > EPS:
                key = f"{o}|{d}|{t}"
                sol["e"][key] = float(v)
        for t in all_times:
            v = omega[o, d, t].X
            if abs(v) > EPS:
                key = f"{o}|{d}|{t}"
                sol["omega"][key] = float(v)

    return sol


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

    solution = build_and_solve(instance, args.time_limit, logger=logger)

    with open(args.solution_path, "w") as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()