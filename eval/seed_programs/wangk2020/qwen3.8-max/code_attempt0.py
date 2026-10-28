import json
import argparse
import time
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    start_time = time.time()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    T = instance['num_time_periods']
    max_disp = instance['maximum_displacement_delta']
    airports = instance['airports']
    flights = instance['flights']
    aircraft_connections = instance.get('aircraft_connections', [])
    passenger_connections = instance.get('passenger_connections', [])
    capacity_envelopes = instance['capacity_envelopes']
    scenarios = instance['scenarios']
    rho = instance.get('rho', 0.5)

    F = len(flights)
    S = len(scenarios)

    # Flight data indexed by id
    flight_ids = [f['id'] for f in flights]
    dep_airport = {f['id']: f['dep_airport'] for f in flights}
    arr_airport = {f['id']: f['arr_airport'] for f in flights}
    dep_period_req = {f['id']: f['dep_period'] for f in flights}
    arr_period_req = {f['id']: f['arr_period'] for f in flights}
    delta_sch = {f['id']: f['delta_sch'] for f in flights}
    delta_min = {f['id']: f['delta_min'] for f in flights}
    delta_max = {f['id']: f['delta_max'] for f in flights}
    max_dep_delay = {f['id']: f['max_dep_delay'] for f in flights}
    max_arr_delay = {f['id']: f['max_arr_delay'] for f in flights}
    cost_dep_delay = {f['id']: f['cost_dep_delay'] for f in flights}
    cost_arr_delay = {f['id']: f['cost_arr_delay'] for f in flights}

    # Time windows for scheduled departure/arrival
    sch_dep_lo = {}
    sch_dep_hi = {}
    sch_arr_lo = {}
    sch_arr_hi = {}
    for f in flights:
        fid = f['id']
        sch_dep_lo[fid] = max(0, f['dep_period'] - max_disp)
        sch_dep_hi[fid] = min(T - 1, f['dep_period'] + max_disp)
        sch_arr_lo[fid] = max(0, f['arr_period'] - max_disp)
        sch_arr_hi[fid] = min(T - 1, f['arr_period'] + max_disp)

    # Time windows for operated departure/arrival
    op_dep_lo = {}
    op_dep_hi = {}
    op_arr_lo = {}
    op_arr_hi = {}
    for f in flights:
        fid = f['id']
        op_dep_lo[fid] = sch_dep_lo[fid]
        op_dep_hi[fid] = min(T - 1, sch_dep_hi[fid] + max_dep_delay[fid])
        op_arr_lo[fid] = sch_arr_lo[fid]
        op_arr_hi[fid] = min(T - 1, sch_arr_hi[fid] + max_arr_delay[fid])

    # Tighten windows based on en-route duration constraints
    for f in flights:
        fid = f['id']
        op_arr_lo[fid] = max(op_arr_lo[fid], op_dep_lo[fid] + delta_min[fid])
        op_dep_hi[fid] = min(op_dep_hi[fid], op_arr_hi[fid] - delta_min[fid])
        op_dep_lo[fid] = max(op_dep_lo[fid], op_arr_lo[fid] - delta_max[fid])
        op_arr_hi[fid] = min(op_arr_hi[fid], op_dep_hi[fid] + delta_max[fid])
        # Ensure valid ranges
        op_dep_lo[fid] = max(0, op_dep_lo[fid])
        op_dep_hi[fid] = min(T - 1, op_dep_hi[fid])
        op_arr_lo[fid] = max(0, op_arr_lo[fid])
        op_arr_hi[fid] = min(T - 1, op_arr_hi[fid])

    # All first-stage connections
    all_connections = aircraft_connections + passenger_connections

    # Flights by airport
    dep_flights_at = {a: [] for a in airports}
    arr_flights_at = {a: [] for a in airports}
    for f in flights:
        fid = f['id']
        dep_flights_at[dep_airport[fid]].append(fid)
        arr_flights_at[arr_airport[fid]].append(fid)

    # Build MIP
    import gurobipy as gp
    from gurobipy import GRB, quicksum, LinExpr

    model = gp.Model("ATM_Stochastic")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = args.time_limit - 5  # Leave some buffer
    model.Params.OutputFlag = 0

    # First-stage variables: u_dep[fid, t], u_arr[fid, t]
    u_dep = {}
    u_arr = {}
    for f in flights:
        fid = f['id']
        for t in range(sch_dep_lo[fid], sch_dep_hi[fid] + 1):
            u_dep[fid, t] = model.addVar(vtype=GRB.BINARY, name=f"ud_{fid}_{t}")
        for t in range(sch_arr_lo[fid], sch_arr_hi[fid] + 1):
            u_arr[fid, t] = model.addVar(vtype=GRB.BINARY, name=f"ua_{fid}_{t}")

    # Second-stage variables: x_dep[fid, t, sid], x_arr[fid, t, sid]
    x_dep = {}
    x_arr = {}
    for s in scenarios:
        sid = s['id']
        for f in flights:
            fid = f['id']
            for t in range(op_dep_lo[fid], op_dep_hi[fid] + 1):
                x_dep[fid, t, sid] = model.addVar(vtype=GRB.BINARY, name=f"xd_{fid}_{t}_{sid}")
            for t in range(op_arr_lo[fid], op_arr_hi[fid] + 1):
                x_arr[fid, t, sid] = model.addVar(vtype=GRB.BINARY, name=f"xa_{fid}_{t}_{sid}")

    # Delay variables
    v_dep = {}
    v_arr = {}
    for s in scenarios:
        sid = s['id']
        for f in flights:
            fid = f['id']
            v_dep[fid, sid] = model.addVar(lb=0, ub=max_dep_delay[fid], vtype=GRB.CONTINUOUS, name=f"vd_{fid}_{sid}")
            v_arr[fid, sid] = model.addVar(lb=0, ub=max_arr_delay[fid], vtype=GRB.CONTINUOUS, name=f"va_{fid}_{sid}")

    model.update()

    # Helper: get indicator value (variable or constant)
    def get_u_dep(fid, t):
        if t < sch_dep_lo[fid]:
            return 1
        elif t > sch_dep_hi[fid]:
            return 0
        return u_dep[fid, t]

    def get_u_arr(fid, t):
        if t < sch_arr_lo[fid]:
            return 1
        elif t > sch_arr_hi[fid]:
            return 0
        return u_arr[fid, t]

    def get_x_dep(fid, t, sid):
        if t < op_dep_lo[fid]:
            return 1
        elif t > op_dep_hi[fid]:
            return 0
        return x_dep[fid, t, sid]

    def get_x_arr(fid, t, sid):
        if t < op_arr_lo[fid]:
            return 1
        elif t > op_arr_hi[fid]:
            return 0
        return x_arr[fid, t, sid]

    # Constraints
    # 1. Non-increasing scheduled departure indicators
    for f in flights:
        fid = f['id']
        for t in range(sch_dep_lo[fid], sch_dep_hi[fid]):
            model.addConstr(u_dep[fid, t] >= u_dep[fid, t + 1])

    # 2. Non-increasing scheduled arrival indicators
    for f in flights:
        fid = f['id']
        for t in range(sch_arr_lo[fid], sch_arr_hi[fid]):
            model.addConstr(u_arr[fid, t] >= u_arr[fid, t + 1])

    # 3. En-route duration (first stage)
    for f in flights:
        fid = f['id']
        lhs = quicksum(u_arr[fid, t] for t in range(sch_arr_lo[fid], sch_arr_hi[fid] + 1)) \
            - quicksum(u_dep[fid, t] for t in range(sch_dep_lo[fid], sch_dep_hi[fid] + 1))
        rhs = delta_sch[fid] - (sch_arr_lo[fid] - sch_dep_lo[fid])
        model.addConstr(lhs == rhs)

    # 4. Connection constraints (first stage)
    for conn in all_connections:
        fi = conn['flight_i']
        fj = conn['flight_j']
        mct = conn['min_connection_time']
        lhs = quicksum(u_dep[fj, t] for t in range(sch_dep_lo[fj], sch_dep_hi[fj] + 1)) \
            - quicksum(u_arr[fi, t] for t in range(sch_arr_lo[fi], sch_arr_hi[fi] + 1))
        rhs = mct - (sch_dep_lo[fj] - sch_arr_lo[fi])
        model.addConstr(lhs >= rhs)

    # 5. Non-increasing operated indicators (second stage)
    for s in scenarios:
        sid = s['id']
        for f in flights:
            fid = f['id']
            for t in range(op_dep_lo[fid], op_dep_hi[fid]):
                model.addConstr(x_dep[fid, t, sid] >= x_dep[fid, t + 1, sid])
            for t in range(op_arr_lo[fid], op_arr_hi[fid]):
                model.addConstr(x_arr[fid, t, sid] >= x_arr[fid, t + 1, sid])

    # 6. Departure delay definition
    for s in scenarios:
        sid = s['id']
        for f in flights:
            fid = f['id']
            op_sum = quicksum(x_dep[fid, t, sid] for t in range(op_dep_lo[fid], op_dep_hi[fid] + 1))
            sch_sum = quicksum(u_dep[fid, t] for t in range(sch_dep_lo[fid], sch_dep_hi[fid] + 1))
            offset = op_dep_lo[fid] - sch_dep_lo[fid]
            model.addConstr(v_dep[fid, sid] == op_sum - sch_sum + offset)

    # 7. Arrival delay constraint (>=)
    for s in scenarios:
        sid = s['id']
        for f in flights:
            fid = f['id']
            op_sum = quicksum(x_arr[fid, t, sid] for t in range(op_arr_lo[fid], op_arr_hi[fid] + 1))
            sch_sum = quicksum(u_arr[fid, t] for t in range(sch_arr_lo[fid], sch_arr_hi[fid] + 1))
            offset = op_arr_lo[fid] - sch_arr_lo[fid]
            model.addConstr(v_arr[fid, sid] >= op_sum - sch_sum + offset)

    # 8. Aircraft connections (second stage)
    for s in scenarios:
        sid = s['id']
        for conn in aircraft_connections:
            fi = conn['flight_i']
            fj = conn['flight_j']
            mct = conn['min_connection_time']
            lhs = quicksum(x_dep[fj, t, sid] for t in range(op_dep_lo[fj], op_dep_hi[fj] + 1)) \
                - quicksum(x_arr[fi, t, sid] for t in range(op_arr_lo[fi], op_arr_hi[fi] + 1))
            rhs = mct - (op_dep_lo[fj] - op_arr_lo[fi])
            model.addConstr(lhs >= rhs)

    # 9. En-route duration bounds (second stage)
    for s in scenarios:
        sid = s['id']
        for f in flights:
            fid = f['id']
            arr_sum = quicksum(x_arr[fid, t, sid] for t in range(op_arr_lo[fid], op_arr_hi[fid] + 1))
            dep_sum = quicksum(x_dep[fid, t, sid] for t in range(op_dep_lo[fid], op_dep_hi[fid] + 1))
            offset = op_arr_lo[fid] - op_dep_lo[fid]
            model.addConstr(arr_sum - dep_sum + offset >= delta_min[fid])
            model.addConstr(arr_sum - dep_sum + offset <= delta_max[fid])

    # 10. Capacity constraints
    for s in scenarios:
        sid = s['id']
        for airport in airports:
            for t in range(T):
                condition = s['operating_conditions'][airport][t]
                if airport not in capacity_envelopes:
                    continue
                segments = capacity_envelopes[airport].get(condition, [])
                if not segments:
                    continue

                for seg in segments:
                    a_coeff = seg['a']
                    b_coeff = seg['b']
                    Q = seg['Q']

                    dep_expr = LinExpr()
                    for fid in dep_flights_at[airport]:
                        val_t = get_x_dep(fid, t, sid)
                        val_t1 = get_x_dep(fid, t + 1, sid)
                        if isinstance(val_t, int) and isinstance(val_t1, int):
                            dep_expr += (val_t - val_t1)
                        else:
                            dep_expr += (val_t - val_t1)

                    arr_expr = LinExpr()
                    for fid in arr_flights_at[airport]:
                        val_t = get_x_arr(fid, t, sid)
                        val_t1 = get_x_arr(fid, t + 1, sid)
                        if isinstance(val_t, int) and isinstance(val_t1, int):
                            arr_expr += (val_t - val_t1)
                        else:
                            arr_expr += (val_t - val_t1)

                    model.addConstr(a_coeff * dep_expr + b_coeff * arr_expr <= Q)

    # Objective
    disp_cost = LinExpr()
    for f in flights:
        fid = f['id']
        r = dep_period_req[fid]
        for t in range(sch_dep_lo[fid], sch_dep_hi[fid] + 1):
            if t > r:
                disp_cost += u_dep[fid, t]
            else:
                disp_cost += (1 - u_dep[fid, t])

    delay_cost = LinExpr()
    for s in scenarios:
        sid = s['id']
        prob = s['probability']
        for f in flights:
            fid = f['id']
            delay_cost += prob * cost_dep_delay[fid] * v_dep[fid, sid]
            delay_cost += prob * cost_arr_delay[fid] * v_arr[fid, sid]

    model.setObjective(rho * disp_cost + (1 - rho) * delay_cost, GRB.MINIMIZE)

    # Callback to log solutions
    best_obj = [float('inf')]
    best_solution = [None]

    def callback(model_cb, where):
        if where == GRB.Callback.MIPSOL:
            obj = model_cb.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj < best_obj[0]:
                best_obj[0] = obj
                # Extract solution
                sol = extract_solution(model_cb)
                best_solution[0] = sol
                if logger:
                    logger.log_solution(obj, sol)

    def extract_solution(m):
        solution = {}
        schedule = {}
        for f in flights:
            fid = f['id']
            sd = sum(m.cbGetSolution(u_dep[fid, t]) if isinstance(u_dep[fid, t], gp.Var) else u_dep[fid, t]
                     for t in range(sch_dep_lo[fid], sch_dep_hi[fid] + 1)) + sch_dep_lo[fid] - 1
            sa = sd + delta_sch[fid]
            schedule[str(fid)] = {"scheduled_dep": int(round(sd)), "scheduled_arr": int(round(sa))}
        solution['schedule'] = schedule

        scenario_solutions = []
        for s in scenarios:
            sid = s['id']
            ss = {"scenario_id": sid, "x_dep": {}, "x_arr": {}, "v_dep": {}, "v_arr": {}}
            for f in flights:
                fid = f['id']
                for t in range(op_dep_lo[fid], op_dep_hi[fid] + 1):
                    val = m.cbGetSolution(x_dep[fid, t, sid])
                    ss["x_dep"][f"{fid}_{t}"] = int(round(val))
                for t in range(op_arr_lo[fid], op_arr_hi[fid] + 1):
                    val = m.cbGetSolution(x_arr[fid, t, sid])
                    ss["x_arr"][f"{fid}_{t}"] = int(round(val))
                vd = m.cbGetSolution(v_dep[fid, sid])
                va = m.cbGetSolution(v_arr[fid, sid])
                ss["v_dep"][str(fid)] = int(round(vd))
                ss["v_arr"][str(fid)] = int(round(va))
            scenario_solutions.append(ss)
        solution['scenario_solutions'] = scenario_solutions
        solution['objective_value'] = best_obj[0]
        return solution

    # Optimize
    model.optimize(callback)

    # Extract final solution
    if model.SolCount > 0:
        obj_val = model.ObjVal
        schedule = {}
        for f in flights:
            fid = f['id']
            sd = sum(u_dep[fid, t].X for t in range(sch_dep_lo[fid], sch_dep_hi[fid] + 1)) + sch_dep_lo[fid] - 1
            sa = sd + delta_sch[fid]
            schedule[str(fid)] = {"scheduled_dep": int(round(sd)), "scheduled_arr": int(round(sa))}

        scenario_solutions = []
        for s in scenarios:
            sid = s['id']
            ss = {"scenario_id": sid, "x_dep": {}, "x_arr": {}, "v_dep": {}, "v_arr": {}}
            for f in flights:
                fid = f['id']
                for t in range(op_dep_lo[fid], op_dep_hi[fid] + 1):
                    ss["x_dep"][f"{fid}_{t}"] = int(round(x_dep[fid, t, sid].X))
                for t in range(op_arr_lo[fid], op_arr_hi[fid] + 1):
                    ss["x_arr"][f"{fid}_{t}"] = int(round(x_arr[fid, t, sid].X))
                ss["v_dep"][str(fid)] = int(round(v_dep[fid, sid].X))
                ss["v_arr"][str(fid)] = int(round(v_arr[fid, sid].X))
            scenario_solutions.append(ss)

        solution = {
            "objective_value": obj_val,
            "schedule": schedule,
            "scenario_solutions": scenario_solutions
        }

        if logger:
            logger.log_solution(obj_val, solution)

    else:
        # Fallback heuristic: use requested times
        schedule = {}
        for f in flights:
            fid = f['id']
            schedule[str(fid)] = {
                "scheduled_dep": dep_period_req[fid],
                "scheduled_arr": arr_period_req[fid]
            }

        scenario_solutions = []
        for s in scenarios:
            sid = s['id']
            ss = {"scenario_id": sid, "x_dep": {}, "x_arr": {}, "v_dep": {}, "v_arr": {}}
            for f in flights:
                fid = f['id']
                sd = dep_period_req[fid]
                sa = arr_period_req[fid]
                for t in range(op_dep_lo[fid], op_dep_hi[fid] + 1):
                    ss["x_dep"][f"{fid}_{t}"] = 1 if t <= sd else 0
                for t in range(op_arr_lo[fid], op_arr_hi[fid] + 1):
                    ss["x_arr"][f"{fid}_{t}"] = 1 if t <= sa else 0
                ss["v_dep"][str(fid)] = 0
                ss["v_arr"][str(fid)] = 0
            scenario_solutions.append(ss)

        # Compute objective for heuristic
        disp = sum(abs(dep_period_req[f['id']] - dep_period_req[f['id']]) for f in flights)
        solution = {
            "objective_value": 0.0,
            "schedule": schedule,
            "scenario_solutions": scenario_solutions
        }
        if logger:
            logger.log_solution(0.0, solution)

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == '__main__':
    main()