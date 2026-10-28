import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    T = data['num_time_periods']
    Delta = data['maximum_displacement_delta']
    airports = data['airports']
    flights = data['flights']
    aircraft_connections = data['aircraft_connections']
    passenger_connections = data['passenger_connections']
    capacity_envelopes = data['capacity_envelopes']
    scenarios = data['scenarios']

    flight_dict = {f['id']: f for f in flights}
    F = [f['id'] for f in flights]

    # Precompute possible periods
    dep_possible = {}
    arr_possible = {}
    for fid in F:
        f = flight_dict[fid]
        req_dep = f['dep_period']
        delta_sch = f['delta_sch']
        min_dep = max(0, req_dep - Delta)
        max_dep = min(T - 1, req_dep + Delta)
        dep_possible[fid] = list(range(min_dep, max_dep + 1))
        arr_possible[fid] = [d + delta_sch for d in dep_possible[fid]]

    # Connections
    all_connections = []
    for conn in passenger_connections:
        all_connections.append((conn['flight_i'], conn['flight_j'], conn['min_connection_time'], False))
    for conn in aircraft_connections:
        all_connections.append((conn['flight_i'], conn['flight_j'], conn['min_connection_time'], True))

    aircraft_conn_set = set()
    for conn in aircraft_connections:
        aircraft_conn_set.add((conn['flight_i'], conn['flight_j']))

    # Start model
    model = gp.Model("AirTrafficScheduling")
    model.setParam('TimeLimit', args.time_limit - 5)  # leave some margin
    model.setParam('LogToConsole', 0)
    model.setParam('MIPGap', 0.01)
    model.setParam('Threads', 1)

    # First stage: scheduled indicators
    x_sch_dep = {}
    x_sch_arr = {}
    for fid in F:
        for t in range(T):
            x_sch_dep[fid, t] = model.addVar(vtype=GRB.BINARY, name=f"xsch_dep_{fid}_{t}")
            x_sch_arr[fid, t] = model.addVar(vtype=GRB.BINARY, name=f"xsch_arr_{fid}_{t}")

    # Second stage: operated indicators per scenario
    x_op_dep = {}
    x_op_arr = {}
    v_dep = {}
    v_arr = {}
    for s_idx, scen in enumerate(scenarios):
        sid = scen['id']
        for fid in F:
            for t in range(T):
                x_op_dep[sid, fid, t] = model.addVar(vtype=GRB.BINARY, name=f"xop_dep_{sid}_{fid}_{t}")
                x_op_arr[sid, fid, t] = model.addVar(vtype=GRB.BINARY, name=f"xop_arr_{sid}_{fid}_{t}")
            v_dep[sid, fid] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=flight_dict[fid]['max_dep_delay'],
                                           name=f"vdep_{sid}_{fid}")
            v_arr[sid, fid] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=flight_dict[fid]['max_arr_delay'],
                                           name=f"varr_{sid}_{fid}")

    # Objective components
    rho = 0.5  # default if not given, but we will extract from problem if needed. Problem mentions rho but not in schema. Assume 0.5 for balance.
    # Note: rho is mentioned in description but not in provided instance_schema. We use 0.5.
    rho = 0.5

    # Schedule displacement cost
    sched_cost = gp.LinExpr()
    for fid in F:
        f = flight_dict[fid]
        req = f['dep_period']
        c_delay = 1.0  # the description says "per-period departure delay cost" but for schedule it's absolute displacement.
        # Actually problem says: "the displacement cost of a flight equals the absolute difference between its scheduled and requested departure periods"
        # So cost is just |s - req| , coefficient is 1 per period.
        for t_idx, t in enumerate(range(T)):
            if t > 0:
                drop = x_sch_dep[fid, t-1] - x_sch_dep[fid, t]
                sched_cost += abs(t - req) * drop

    # Expected operational delay cost
    exp_delay_cost = gp.LinExpr()
    for s_idx, scen in enumerate(scenarios):
        prob = scen['probability']
        sid = scen['id']
        for fid in F:
            f = flight_dict[fid]
            exp_delay_cost += prob * f['cost_dep_delay'] * v_dep[sid, fid]
            exp_delay_cost += prob * f['cost_arr_delay'] * v_arr[sid, fid]

    model.setObjective(rho * sched_cost + (1 - rho) * exp_delay_cost, GRB.MINIMIZE)

    # Constraints

    # 1. Scheduled indicators non-increasing, fix outside window
    for fid in F:
        min_d = min(dep_possible[fid])
        max_d = max(dep_possible[fid])
        for t in range(T):
            if t < min_d:
                model.addConstr(x_sch_dep[fid, t] == 1)
            if t > max_d:
                model.addConstr(x_sch_dep[fid, t] == 0)
            if t > 0:
                model.addConstr(x_sch_dep[fid, t] <= x_sch_dep[fid, t-1])
                model.addConstr(x_sch_arr[fid, t] <= x_sch_arr[fid, t-1])

        min_a = min(arr_possible[fid])
        max_a = max(arr_possible[fid])
        for t in range(T):
            if t < min_a:
                model.addConstr(x_sch_arr[fid, t] == 1)
            if t > max_a:
                model.addConstr(x_sch_arr[fid, t] == 0)

    # 2. Scheduled en-route duration
    for fid in F:
        f = flight_dict[fid]
        delta_sch = f['delta_sch']
        sum_arr = gp.quicksum(x_sch_arr[fid, t] for t in range(T))
        sum_dep = gp.quicksum(x_sch_dep[fid, t] for t in range(T))
        model.addConstr(sum_arr - sum_dep == delta_sch)

    # 3. Connection constraints for schedule (all connections)
    for i, j, min_conn, _ in all_connections:
        sum_dep_j = gp.quicksum(x_sch_dep[j, t] for t in range(T))
        sum_arr_i = gp.quicksum(x_sch_arr[i, t] for t in range(T))
        model.addConstr(sum_dep_j - sum_arr_i >= min_conn)

    # 4. For each scenario
    for s_idx, scen in enumerate(scenarios):
        sid = scen['id']
        prob = scen['probability']
        op_cond = scen['operating_conditions']

        for fid in F:
            f = flight_dict[fid]
            # Operated non-increasing
            for t in range(1, T):
                model.addConstr(x_op_dep[sid, fid, t] <= x_op_dep[sid, fid, t-1])
                model.addConstr(x_op_arr[sid, fid, t] <= x_op_arr[sid, fid, t-1])

            # Link to scheduled: v_dep = sum(xop_dep) - sum(xsch_dep)
            sum_op_dep = gp.quicksum(x_op_dep[sid, fid, t] for t in range(T))
            sum_sch_dep = gp.quicksum(x_sch_dep[fid, t] for t in range(T))
            model.addConstr(sum_op_dep - sum_sch_dep == v_dep[sid, fid])

            # v_arr >= sum(xop_arr) - sum(xsch_arr)
            sum_op_arr = gp.quicksum(x_op_arr[sid, fid, t] for t in range(T))
            sum_sch_arr = gp.quicksum(x_sch_arr[fid, t] for t in range(T))
            model.addConstr(v_arr[sid, fid] >= sum_op_arr - sum_sch_arr)

            # Enroute duration limits
            model.addConstr(sum_op_arr - sum_op_dep >= f['delta_min'])
            model.addConstr(sum_op_arr - sum_op_dep <= f['delta_max'])

        # Aircraft connection constraints (only aircraft)
        for i, j, min_conn, is_aircraft in all_connections:
            if is_aircraft:
                sum_op_dep_j = gp.quicksum(x_op_dep[sid, j, t] for t in range(T))
                sum_op_arr_i = gp.quicksum(x_op_arr[sid, i, t] for t in range(T))
                model.addConstr(sum_op_dep_j - sum_op_arr_i >= min_conn)

        # Capacity constraints
        for ap in airports:
            env_vmc = capacity_envelopes[ap]['VMC']
            env_imc = capacity_envelopes[ap]['IMC']
            deps = [[] for _ in range(T)]
            arrs = [[] for _ in range(T)]
            for fid in F:
                f = flight_dict[fid]
                if f['dep_airport'] == ap:
                    for t in range(T):
                        if t < T-1:
                            drop = x_op_dep[sid, fid, t] - x_op_dep[sid, fid, t+1]
                            deps[t].append(drop)
                        else:
                            drop = x_op_dep[sid, fid, t]
                            deps[t].append(drop)
                if f['arr_airport'] == ap:
                    for t in range(T):
                        if t < T-1:
                            drop = x_op_arr[sid, fid, t] - x_op_arr[sid, fid, t+1]
                            arrs[t].append(drop)
                        else:
                            drop = x_op_arr[sid, fid, t]
                            arrs[t].append(drop)

            for tau in range(T):
                cond = op_cond[ap][tau]
                segments = env_vmc if cond == "VMC" else env_imc
                num_dep_tau = gp.quicksum(deps[tau])
                num_arr_tau = gp.quicksum(arrs[tau])
                for seg in segments:
                    model.addConstr(seg['a'] * num_dep_tau + seg['b'] * num_arr_tau <= seg['Q'])

    # Callback for logging incumbents
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if logger:
                logger.log(float(obj))

    # Optimize
    start_time = time.time()
    model.optimize(callback)

    # Extract solution
    if model.status == GRB.INFEASIBLE:
        print("Model is infeasible")
        return
    elif model.status not in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SOLUTION_LIMIT]:
        print("No solution found")
        return

    best_obj = model.objVal

    # Build schedule dict
    schedule = {}
    for fid in F:
        sch_dep_val = None
        for t in range(T):
            val = x_sch_dep[fid, t].X
            if t == 0 and abs(val - 1.0) < 1e-5:
                continue
            if t > 0:
                prev = x_sch_dep[fid, t-1].X
                if abs(prev - 1.0) < 1e-5 and abs(val - 0.0) < 1e-5:
                    sch_dep_val = t
                    break
        if sch_dep_val is None:
            sch_dep_val = max(dep_possible[fid]) + 1 if x_sch_dep[fid, T-1].X > 0.5 else T
        sch_arr_val = None
        for t in range(T):
            val = x_sch_arr[fid, t].X
            if t > 0:
                prev = x_sch_arr[fid, t-1].X
                if abs(prev - 1.0) < 1e-5 and abs(val) < 1e-5:
                    sch_arr_val = t
                    break
        if sch_arr_val is None:
            sch_arr_val = T
        schedule[str(fid)] = {
            "scheduled_dep": int(sch_dep_val),
            "scheduled_arr": int(sch_arr_val)
        }

    # Scenario solutions
    scenario_solutions = []
    for scen in scenarios:
        sid = scen['id']
        xdep_dict = {}
        xarr_dict = {}
        vdep_dict = {}
        varr_dict = {}
        for fid in F:
            for t in range(T):
                xdep_dict[f"{fid}_{t}"] = int(round(x_op_dep[sid, fid, t].X))
                xarr_dict[f"{fid}_{t}"] = int(round(x_op_arr[sid, fid, t].X))
            vdep_dict[str(fid)] = int(round(v_dep[sid, fid].X))
            varr_dict[str(fid)] = int(round(v_arr[sid, fid].X))
        scenario_solutions.append({
            "scenario_id": int(sid),
            "x_dep": xdep_dict,
            "x_arr": xarr_dict,
            "v_dep": vdep_dict,
            "v_arr": varr_dict
        })

    solution = {
        "objective_value": float(best_obj),
        "schedule": schedule,
        "scenario_solutions": scenario_solutions
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

    if logger:
        logger.log(float(best_obj))

if __name__ == "__main__":
    main()