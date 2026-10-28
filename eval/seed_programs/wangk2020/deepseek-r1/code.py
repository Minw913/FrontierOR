import gurobipy as gp
from gurobipy import GRB
import json
import argparse
import os
from solution_logger import SolutionLogger

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

    num_time_periods = data['num_time_periods']
    maximum_displacement_delta = data['maximum_displacement_delta']
    airports = data['airports']
    flights = data['flights']
    aircraft_connections = data.get('aircraft_connections', [])
    passenger_connections = data.get('passenger_connections', [])
    capacity_envelopes = data['capacity_envelopes']
    scenarios = data['scenarios']
    rho = data.get('rho', 0.5)

    model = gp.Model()
    model.setParam('OutputFlag', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', args.time_limit)

    x_dep0 = {}
    x_arr0 = {}
    displacement_plus = {}
    displacement_minus = {}

    for flight in flights:
        fid = flight['id']
        dep_req = flight['dep_period']
        L_dep = max(0, dep_req - maximum_displacement_delta)
        U_dep = min(num_time_periods-1, dep_req + maximum_displacement_delta)
        x_dep0_f = []
        for t in range(num_time_periods):
            if t < L_dep:
                v = model.addVar(vtype=GRB.BINARY, lb=1, ub=1, name=f"x_dep0_{fid}_{t}")
            elif t > U_dep:
                v = model.addVar(vtype=GRB.BINARY, lb=0, ub=0, name=f"x_dep0_{fid}_{t}")
            else:
                v = model.addVar(vtype=GRB.BINARY, name=f"x_dep0_{fid}_{t}")
            x_dep0_f.append(v)
        x_dep0[fid] = x_dep0_f
        for t in range(num_time_periods-1):
            model.addConstr(x_dep0_f[t] >= x_dep0_f[t+1], name=f"noninc_dep0_{fid}_{t}")

        arr_req = flight['arr_period']
        L_arr = max(0, L_dep + flight['delta_sch'])
        U_arr = min(num_time_periods-1, U_dep + flight['delta_sch'])
        x_arr0_f = []
        for t in range(num_time_periods):
            if t < L_arr:
                v = model.addVar(vtype=GRB.BINARY, lb=1, ub=1, name=f"x_arr0_{fid}_{t}")
            elif t > U_arr:
                v = model.addVar(vtype=GRB.BINARY, lb=0, ub=0, name=f"x_arr0_{fid}_{t}")
            else:
                v = model.addVar(vtype=GRB.BINARY, name=f"x_arr0_{fid}_{t}")
            x_arr0_f.append(v)
        x_arr0[fid] = x_arr0_f
        for t in range(num_time_periods-1):
            model.addConstr(x_arr0_f[t] >= x_arr0_f[t+1], name=f"noninc_arr0_{fid}_{t}")

        model.addConstr(
            gp.quicksum(x_arr0_f) - gp.quicksum(x_dep0_f) == flight['delta_sch'],
            name=f"enroute_{fid}"
        )

        sched_dep = gp.quicksum(x_dep0_f)
        d_plus = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"d_plus_{fid}")
        d_minus = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"d_minus_{fid}")
        model.addConstr(sched_dep - dep_req == d_plus - d_minus, name=f"abs_constr_{fid}")
        displacement_plus[fid] = d_plus
        displacement_minus[fid] = d_minus

    first_set_connections = aircraft_connections + passenger_connections
    for conn in first_set_connections:
        fid_i = conn['flight_i']
        fid_j = conn['flight_j']
        min_conn = conn['min_connection_time']
        sum_dep_j = gp.quicksum(x_dep0[fid_j])
        sum_arr_i = gp.quicksum(x_arr0[fid_i])
        model.addConstr(sum_dep_j - sum_arr_i >= min_conn, name=f"conn_sched_{fid_i}_{fid_j}")

    scenario_delay_costs = []
    scenario_vars = {}

    for scenario in scenarios:
        sid = scenario['id']
        prob = scenario['probability']
        operating_conditions = scenario['operating_conditions']
        x_dep_s = {}
        x_arr_s = {}
        v_dep_s = {}
        v_arr_s = {}

        for flight in flights:
            fid = flight['id']
            x_dep_f = []
            for t in range(num_time_periods):
                x = model.addVar(vtype=GRB.BINARY, name=f"x_dep_{fid}_{t}_s{sid}")
                x_dep_f.append(x)
            x_dep_s[fid] = x_dep_f
            for t in range(num_time_periods-1):
                model.addConstr(x_dep_f[t] >= x_dep_f[t+1], name=f"noninc_dep_{fid}_{t}_s{sid}")

            x_arr_f = []
            for t in range(num_time_periods):
                x = model.addVar(vtype=GRB.BINARY, name=f"x_arr_{fid}_{t}_s{sid}")
                x_arr_f.append(x)
            x_arr_s[fid] = x_arr_f
            for t in range(num_time_periods-1):
                model.addConstr(x_arr_f[t] >= x_arr_f[t+1], name=f"noninc_arr_{fid}_{t}_s{sid}")

            v_dep = model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name=f"v_dep_{fid}_s{sid}")
            v_arr = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"v_arr_{fid}_s{sid}")
            v_dep_s[fid] = v_dep
            v_arr_s[fid] = v_arr

            sum_dep_oper = gp.quicksum(x_dep_f)
            sum_dep_sched = gp.quicksum(x_dep0[fid])
            model.addConstr(v_dep == sum_dep_oper - sum_dep_sched, name=f"def_v_dep_{fid}_s{sid}")

            sum_arr_oper = gp.quicksum(x_arr_f)
            sum_arr_sched = gp.quicksum(x_arr0[fid])
            model.addConstr(v_arr >= sum_arr_oper - sum_arr_sched, name=f"def_v_arr_{fid}_s{sid}")

            model.addConstr(v_dep <= flight['max_dep_delay'], name=f"max_dep_{fid}_s{sid}")
            model.addConstr(v_arr <= flight['max_arr_delay'], name=f"max_arr_{fid}_s{sid}")

            enroute_oper = sum_arr_oper - sum_dep_oper
            model.addConstr(enroute_oper >= flight['delta_min'], name=f"enroute_min_{fid}_s{sid}")
            model.addConstr(enroute_oper <= flight['delta_max'], name=f"enroute_max_{fid}_s{sid}")

        for conn in aircraft_connections:
            fid_i = conn['flight_i']
            fid_j = conn['flight_j']
            min_conn = conn['min_connection_time']
            sum_dep_j = gp.quicksum(x_dep_s[fid_j])
            sum_arr_i = gp.quicksum(x_arr_s[fid_i])
            model.addConstr(sum_dep_j - sum_arr_i >= min_conn, name=f"conn_oper_{fid_i}_{fid_j}_s{sid}")

        for airport in airports:
            cond_list = operating_conditions[airport]
            for t in range(num_time_periods):
                cond = cond_list[t]
                segments = capacity_envelopes[airport][cond]
                flights_dep = [f for f in flights if f['dep_airport'] == airport]
                flights_arr = [f for f in flights if f['arr_airport'] == airport]
                if t < num_time_periods - 1:
                    dep_count = gp.quicksum(x_dep_s[f['id']][t] - x_dep_s[f['id']][t+1] for f in flights_dep)
                    arr_count = gp.quicksum(x_arr_s[f['id']][t] - x_arr_s[f['id']][t+1] for f in flights_arr)
                else:
                    dep_count = gp.quicksum(x_dep_s[f['id']][t] for f in flights_dep)
                    arr_count = gp.quicksum(x_arr_s[f['id']][t] for f in flights_arr)
                for seg in segments:
                    a_seg = seg['a']
                    b_seg = seg['b']
                    Q_seg = seg['Q']
                    model.addConstr(
                        a_seg * dep_count + b_seg * arr_count <= Q_seg,
                        name=f"cap_{airport}_{t}_s{sid}_seg"
                    )

        scenario_cost = gp.quicksum(
            flight['cost_dep_delay'] * v_dep_s[flight['id']] + 
            flight['cost_arr_delay'] * v_arr_s[flight['id']] 
            for flight in flights
        )
        scenario_delay_costs.append(prob * scenario_cost)
        scenario_vars[sid] = {
            'x_dep_s': x_dep_s,
            'x_arr_s': x_arr_s,
            'v_dep_s': v_dep_s,
            'v_arr_s': v_arr_s
        }

    total_displacement_cost = gp.quicksum(
        displacement_plus[f['id']] + displacement_minus[f['id']] for f in flights
    )
    total_expected_delay = gp.quicksum(scenario_delay_costs)
    model.setObjective(rho * total_displacement_cost + (1 - rho) * total_expected_delay, GRB.MINIMIZE)

    def callback_wrapper(model, where):
        if where == GRB.Callback.MIPSOL:
            if logger:
                obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                logger.log(obj)

    try:
        if logger:
            model.optimize(callback_wrapper)
        else:
            model.optimize()
    except gp.GurobiError as e:
        print(f"Gurobi error: {e}")
    except Exception as e:
        print(f"Error: {e}")

    solution = None
    if model.SolCount > 0:
        schedule = {}
        for flight in flights:
            fid = flight['id']
            dep_sum = sum(x_dep0[fid][t].X for t in range(num_time_periods))
            s_dep = int(round(dep_sum)) - 1
            arr_sum = sum(x_arr0[fid][t].X for t in range(num_time_periods))
            s_arr = int(round(arr_sum)) - 1
            schedule[fid] = {
                'scheduled_dep': s_dep,
                'scheduled_arr': s_arr
            }

        scenario_solutions = []
        for scenario in scenarios:
            sid = scenario['id']
            sol_dict = {
                'scenario_id': sid,
                'x_dep': {},
                'x_arr': {},
                'v_dep': {},
                'v_arr': {}
            }
            for flight in flights:
                fid = flight['id']
                for t in range(num_time_periods):
                    key_dep = f"{fid}_{t}"
                    sol_dict['x_dep'][key_dep] = int(round(scenario_vars[sid]['x_dep_s'][fid][t].X))
                    key_arr = f"{fid}_{t}"
                    sol_dict['x_arr'][key_arr] = int(round(scenario_vars[sid]['x_arr_s'][fid][t].X))
                key_v_dep = str(fid)
                sol_dict['v_dep'][key_v_dep] = scenario_vars[sid]['v_dep_s'][fid].X
                key_v_arr = str(fid)
                sol_dict['v_arr'][key_v_arr] = scenario_vars[sid]['v_arr_s'][fid].X
            scenario_solutions.append(sol_dict)

        solution = {
            'objective_value': model.ObjVal,
            'schedule': schedule,
            'scenario_solutions': scenario_solutions
        }
    else:
        schedule = {}
        for flight in flights:
            schedule[flight['id']] = {
                'scheduled_dep': flight['dep_period'],
                'scheduled_arr': flight['arr_period']
            }
        scenario_solutions = []
        for scenario in scenarios:
            sid = scenario['id']
            sol_dict = {
                'scenario_id': sid,
                'x_dep': {},
                'x_arr': {},
                'v_dep': {},
                'v_arr': {}
            }
            for flight in flights:
                fid = flight['id']
                dep_req = flight['dep_period']
                arr_req = flight['arr_period']
                for t in range(num_time_periods):
                    key_dep = f"{fid}_{t}"
                    sol_dict['x_dep'][key_dep] = 1 if t <= dep_req else 0
                    key_arr = f"{fid}_{t}"
                    sol_dict['x_arr'][key_arr] = 1 if t <= arr_req else 0
                key_v_dep = str(fid)
                sol_dict['v_dep'][key_v_dep] = 0.0
                key_v_arr = str(fid)
                sol_dict['v_arr'][key_v_arr] = 0.0
            scenario_solutions.append(sol_dict)
        solution = {
            'objective_value': None,
            'schedule': schedule,
            'scenario_solutions': scenario_solutions
        }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == "__main__":
    main()