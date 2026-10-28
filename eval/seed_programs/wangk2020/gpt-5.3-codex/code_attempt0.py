import argparse
import json
import math
from collections import defaultdict

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def build_and_solve(instance, time_limit, logger=None):
    T = int(instance["num_time_periods"])
    delta_global = int(instance["maximum_displacement_delta"])
    airports = instance["airports"]
    flights_data = instance["flights"]
    aircraft_connections = instance.get("aircraft_connections", [])
    passenger_connections = instance.get("passenger_connections", [])
    capacity_envelopes = instance["capacity_envelopes"]
    scenarios = instance["scenarios"]
    rho = float(instance.get("rho", 0.5))

    # Flight maps
    flights = [f["id"] for f in flights_data]
    fdict = {f["id"]: f for f in flights_data}

    dep_airport = {f["id"]: f["dep_airport"] for f in flights_data}
    arr_airport = {f["id"]: f["arr_airport"] for f in flights_data}

    # Candidate scheduled departure times per flight (within displacement window and horizon)
    sched_candidates = {}
    for f in flights_data:
        fid = f["id"]
        req_dep = int(f["dep_period"])
        delta_sch = int(f["delta_sch"])
        lo = max(0, req_dep - delta_global)
        hi = min(T - 1, req_dep + delta_global)

        cand = [t for t in range(lo, hi + 1) if 0 <= t + delta_sch <= T - 1]
        if not cand:
            # Fallback to any horizon-feasible departure time
            cand = [t for t in range(T) if 0 <= t + delta_sch <= T - 1]
        if not cand:
            # As absolute fallback (should not happen in valid instances)
            cand = [max(0, min(T - 1, req_dep))]
        sched_candidates[fid] = cand

    # Group flights by airport for capacity constraints
    dep_flights_by_airport = defaultdict(list)
    arr_flights_by_airport = defaultdict(list)
    for fid in flights:
        dep_flights_by_airport[dep_airport[fid]].append(fid)
        arr_flights_by_airport[arr_airport[fid]].append(fid)

    m = gp.Model("stochastic_air_traffic")
    m.Params.Threads = 1
    m.Params.TimeLimit = time_limit
    m.Params.OutputFlag = 0
    m.Params.MIPFocus = 1

    # First-stage variables: scheduled departure choice
    z = {}
    for fid in flights:
        for t in sched_candidates[fid]:
            z[fid, t] = m.addVar(vtype=GRB.BINARY, name=f"z_{fid}_{t}")

    # One scheduled departure per flight
    for fid in flights:
        m.addConstr(gp.quicksum(z[fid, t] for t in sched_candidates[fid]) == 1, name=f"sched_one_{fid}")

    # Expressions for scheduled dep/arr times
    t_sched_dep = {}
    t_sched_arr = {}
    for fid in flights:
        t_sched_dep[fid] = gp.quicksum(t * z[fid, t] for t in sched_candidates[fid])
        t_sched_arr[fid] = t_sched_dep[fid] + int(fdict[fid]["delta_sch"])

    # First-stage connections: aircraft + passenger
    for c in aircraft_connections:
        i = c["flight_i"]
        j = c["flight_j"]
        mct = int(c["min_connection_time"])
        if i in fdict and j in fdict:
            m.addConstr(t_sched_dep[j] - t_sched_arr[i] >= mct, name=f"conn_ac_sched_{i}_{j}")

    for c in passenger_connections:
        i = c["flight_i"]
        j = c["flight_j"]
        mct = int(c["min_connection_time"])
        if i in fdict and j in fdict:
            m.addConstr(t_sched_dep[j] - t_sched_arr[i] >= mct, name=f"conn_pax_sched_{i}_{j}")

    # Second-stage variables per scenario
    y_dep = {}
    y_arr = {}
    t_dep = {}
    t_arr = {}
    v_dep = {}
    v_arr = {}

    scen_prob = {}
    scen_ids = []
    scen_cond = {}  # scen_id -> airport -> list conditions length T

    for s in scenarios:
        sid = s["id"]
        scen_ids.append(sid)
        scen_prob[sid] = float(s["probability"])
        scen_cond[sid] = s["operating_conditions"]

    # Build vars
    for sid in scen_ids:
        for fid in flights:
            for t in range(T):
                y_dep[fid, t, sid] = m.addVar(vtype=GRB.BINARY, name=f"yd_{fid}_{t}_{sid}")
                y_arr[fid, t, sid] = m.addVar(vtype=GRB.BINARY, name=f"ya_{fid}_{t}_{sid}")

            t_dep[fid, sid] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=T - 1, name=f"tdep_{fid}_{sid}")
            t_arr[fid, sid] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=T - 1, name=f"tarr_{fid}_{sid}")

            v_dep[fid, sid] = m.addVar(
                vtype=GRB.CONTINUOUS,
                lb=0,
                ub=float(fdict[fid]["max_dep_delay"]),
                name=f"vdep_{fid}_{sid}",
            )
            v_arr[fid, sid] = m.addVar(
                vtype=GRB.CONTINUOUS,
                lb=0,
                ub=float(fdict[fid]["max_arr_delay"]),
                name=f"varr_{fid}_{sid}",
            )

    # Second-stage constraints
    for sid in scen_ids:
        for fid in flights:
            # Unique operated dep/arr period
            m.addConstr(gp.quicksum(y_dep[fid, t, sid] for t in range(T)) == 1, name=f"dep_one_{fid}_{sid}")
            m.addConstr(gp.quicksum(y_arr[fid, t, sid] for t in range(T)) == 1, name=f"arr_one_{fid}_{sid}")

            # Time definitions
            m.addConstr(
                t_dep[fid, sid] == gp.quicksum(t * y_dep[fid, t, sid] for t in range(T)),
                name=f"tdep_def_{fid}_{sid}",
            )
            m.addConstr(
                t_arr[fid, sid] == gp.quicksum(t * y_arr[fid, t, sid] for t in range(T)),
                name=f"tarr_def_{fid}_{sid}",
            )

            # Departure delay exactly relative to schedule (and thus no early departure)
            m.addConstr(v_dep[fid, sid] == t_dep[fid, sid] - t_sched_dep[fid], name=f"vdep_def_{fid}_{sid}")

            # Arrival delay lower bound relative to schedule
            m.addConstr(v_arr[fid, sid] >= t_arr[fid, sid] - t_sched_arr[fid], name=f"varr_lb_{fid}_{sid}")

            # En-route duration bounds
            m.addConstr(
                t_arr[fid, sid] - t_dep[fid, sid] >= int(fdict[fid]["delta_min"]),
                name=f"dur_min_{fid}_{sid}",
            )
            m.addConstr(
                t_arr[fid, sid] - t_dep[fid, sid] <= int(fdict[fid]["delta_max"]),
                name=f"dur_max_{fid}_{sid}",
            )

        # Aircraft-only connection constraints in operations
        for c in aircraft_connections:
            i = c["flight_i"]
            j = c["flight_j"]
            mct = int(c["min_connection_time"])
            if i in fdict and j in fdict:
                m.addConstr(t_dep[j, sid] - t_arr[i, sid] >= mct, name=f"conn_ac_ops_{i}_{j}_{sid}")

        # Capacity constraints by airport, period, scenario and envelope segment
        for a in airports:
            dep_list = dep_flights_by_airport.get(a, [])
            arr_list = arr_flights_by_airport.get(a, [])
            for t in range(T):
                cond = scen_cond[sid][a][t]  # "VMC" or "IMC"
                segments = capacity_envelopes[a][cond]
                dep_count = gp.quicksum(y_dep[fid, t, sid] for fid in dep_list) if dep_list else 0
                arr_count = gp.quicksum(y_arr[fid, t, sid] for fid in arr_list) if arr_list else 0
                for k, seg in enumerate(segments):
                    aa = float(seg["a"])
                    bb = float(seg["b"])
                    qq = float(seg["Q"])
                    m.addConstr(
                        aa * dep_count + bb * arr_count <= qq,
                        name=f"cap_{a}_{t}_{sid}_{k}",
                    )

    # Objective components
    sched_cost = gp.quicksum(
        abs(int(t) - int(fdict[fid]["dep_period"])) * z[fid, t]
        for fid in flights
        for t in sched_candidates[fid]
    )

    exp_delay_cost = gp.quicksum(
        scen_prob[sid]
        * gp.quicksum(
            float(fdict[fid]["cost_dep_delay"]) * v_dep[fid, sid]
            + float(fdict[fid]["cost_arr_delay"]) * v_arr[fid, sid]
            for fid in flights
        )
        for sid in scen_ids
    )

    m.setObjective(rho * sched_cost + (1.0 - rho) * exp_delay_cost, GRB.MINIMIZE)

    # Incumbent logging callback
    if logger is not None:
        m._logger = logger

        def cb(model, where):
            if where == GRB.Callback.MIPSOL:
                obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                model._logger.log(float(obj))

        m.optimize(cb)
    else:
        m.optimize()

    return m, {
        "flights": flights,
        "fdict": fdict,
        "T": T,
        "sched_candidates": sched_candidates,
        "scen_ids": scen_ids,
        "z": z,
        "y_dep": y_dep,
        "y_arr": y_arr,
        "v_dep": v_dep,
        "v_arr": v_arr,
    }


def extract_solution(model, ctx):
    flights = ctx["flights"]
    fdict = ctx["fdict"]
    T = ctx["T"]
    sched_candidates = ctx["sched_candidates"]
    scen_ids = ctx["scen_ids"]
    z = ctx["z"]
    y_dep = ctx["y_dep"]
    y_arr = ctx["y_arr"]
    v_dep = ctx["v_dep"]
    v_arr = ctx["v_arr"]

    if model.SolCount == 0:
        # No feasible solution found
        return {
            "objective_value": 1e30,
            "schedule": {str(fid): {"scheduled_dep": int(fdict[fid]["dep_period"]), "scheduled_arr": int(fdict[fid]["arr_period"])} for fid in flights},
            "scenario_solutions": [
                {
                    "scenario_id": sid,
                    "x_dep": {f"{fid}_{t}": 0 for fid in flights for t in range(T)},
                    "x_arr": {f"{fid}_{t}": 0 for fid in flights for t in range(T)},
                    "v_dep": {str(fid): 0 for fid in flights},
                    "v_arr": {str(fid): 0 for fid in flights},
                }
                for sid in scen_ids
            ],
        }

    # Scheduled times
    sched_dep_time = {}
    sched_arr_time = {}
    for fid in flights:
        best_t = None
        best_v = -1.0
        for t in sched_candidates[fid]:
            val = z[fid, t].X
            if val > best_v:
                best_v = val
                best_t = t
        if best_t is None:
            best_t = int(round(sum(t * z[fid, t].X for t in sched_candidates[fid])))
        sched_dep_time[fid] = int(best_t)
        sched_arr_time[fid] = int(best_t + int(fdict[fid]["delta_sch"]))

    schedule_out = {
        str(fid): {"scheduled_dep": sched_dep_time[fid], "scheduled_arr": sched_arr_time[fid]}
        for fid in flights
    }

    # Scenario solutions
    scenario_solutions = []
    for sid in scen_ids:
        dep_time = {}
        arr_time = {}

        for fid in flights:
            dep_t = max(range(T), key=lambda t: y_dep[fid, t, sid].X)
            arr_t = max(range(T), key=lambda t: y_arr[fid, t, sid].X)
            dep_time[fid] = int(dep_t)
            arr_time[fid] = int(arr_t)

        x_dep_out = {}
        x_arr_out = {}
        for fid in flights:
            dt = dep_time[fid]
            at = arr_time[fid]
            for t in range(T):
                x_dep_out[f"{fid}_{t}"] = 1 if dt >= t else 0
                x_arr_out[f"{fid}_{t}"] = 1 if at >= t else 0

        v_dep_out = {str(fid): int(round(v_dep[fid, sid].X)) for fid in flights}
        v_arr_out = {str(fid): int(round(v_arr[fid, sid].X)) for fid in flights}

        scenario_solutions.append(
            {
                "scenario_id": sid,
                "x_dep": x_dep_out,
                "x_arr": x_arr_out,
                "v_dep": v_dep_out,
                "v_arr": v_arr_out,
            }
        )

    return {
        "objective_value": float(model.ObjVal),
        "schedule": schedule_out,
        "scenario_solutions": scenario_solutions,
    }


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

    model, ctx = build_and_solve(instance, args.time_limit, logger)
    solution = extract_solution(model, ctx)

    with open(args.solution_path, "w") as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()