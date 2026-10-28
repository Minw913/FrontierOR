import json
import argparse
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    # Load instance data
    with open(args.instance_path, "r") as f:
        data = json.load(f)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    T = data["num_time_periods"]
    max_disp = data["maximum_displacement_delta"]
    airports = data["airports"]
    flights = data["flights"]
    num_flights = len(flights)
    aircraft_conn = data.get("aircraft_connections", [])
    passenger_conn = data.get("passenger_connections", [])
    envelopes = data["capacity_envelopes"]
    scenarios = data["scenarios"]
    num_scenarios = len(scenarios)
    rho = data.get("rho", 0.5)

    # Map flight ids to indices
    flight_dict = {f["id"]: idx for idx, f in enumerate(flights)}

    # Precompute flights by airport
    dep_flights = {a: [] for a in airports}
    arr_flights = {a: [] for a in airports}
    for idx, f in enumerate(flights):
        dep_flights[f["dep_airport"]].append(idx)
        arr_flights[f["arr_airport"]].append(idx)

    # Create Gurobi environment and model
    env = gp.Env(empty=True)
    env.setParam("Seed", 0)
    env.setParam("Threads", 1)
    env.start()
    model = gp.Model(env=env)
    model.setParam("MIPGap", 1e-4)
    model.setParam("NumericFocus", 0)
    model.setParam("TimeLimit", args.time_limit)

    # First-stage variables: scheduled departure times and displacement costs
    s_dep = model.addVars(range(num_flights), vtype=GRB.INTEGER, name="s_dep")
    disp = model.addVars(range(num_flights), vtype=GRB.CONTINUOUS, lb=0, name="disp")

    for f_idx, f in enumerate(flights):
        p = f["dep_period"]
        delta_sch = f["delta_sch"]
        lb = max(1, p - max_disp)
        ub = min(T - delta_sch, p + max_disp)
        s_dep[f_idx].lb = lb
        s_dep[f_idx].ub = ub
        model.addConstr(disp[f_idx] >= s_dep[f_idx] - p)
        model.addConstr(disp[f_idx] >= p - s_dep[f_idx])

    # Scheduled connections (passenger + aircraft)
    for conn in passenger_conn + aircraft_conn:
        i = flight_dict[conn["flight_i"]]
        j = flight_dict[conn["flight_j"]]
        min_t = conn["min_connection_time"]
        delta_sch_i = flights[i]["delta_sch"]
        model.addConstr(s_dep[j] - s_dep[i] >= min_t + delta_sch_i)

    # Second-stage variables: operated departure/arrival indicators and delays
    x_dep = model.addVars(range(num_scenarios), range(num_flights), range(T),
                          vtype=GRB.BINARY, name="x_dep")
    x_arr = model.addVars(range(num_scenarios), range(num_flights), range(T),
                          vtype=GRB.BINARY, name="x_arr")
    v_dep = model.addVars(range(num_scenarios), range(num_flights),
                          lb=0, vtype=GRB.CONTINUOUS, name="v_dep")
    v_arr = model.addVars(range(num_scenarios), range(num_flights),
                          lb=0, vtype=GRB.CONTINUOUS, name="v_arr")

    # Set delay upper bounds
    for s_idx in range(num_scenarios):
        for f_idx, f in enumerate(flights):
            v_dep[s_idx, f_idx].ub = f["max_dep_delay"]
            v_arr[s_idx, f_idx].ub = f["max_arr_delay"]

    # Second-stage constraints
    for s_idx in range(num_scenarios):
        for f_idx, f in enumerate(flights):
            # Non-increasing indicators
            for t in range(T - 1):
                model.addConstr(x_dep[s_idx, f_idx, t] >= x_dep[s_idx, f_idx, t + 1])
                model.addConstr(x_arr[s_idx, f_idx, t] >= x_arr[s_idx, f_idx, t + 1])
            # Must depart/arrive at some period
            model.addConstr(x_dep[s_idx, f_idx, 0] == 1)
            model.addConstr(x_arr[s_idx, f_idx, 0] == 1)

            sum_dep = gp.quicksum(x_dep[s_idx, f_idx, t] for t in range(T))
            sum_arr = gp.quicksum(x_arr[s_idx, f_idx, t] for t in range(T))

            # Delays
            model.addConstr(v_dep[s_idx, f_idx] == sum_dep - s_dep[f_idx])
            model.addConstr(v_arr[s_idx, f_idx] >= sum_arr - (s_dep[f_idx] + f["delta_sch"]))

            # En-route duration limits
            model.addConstr(sum_arr - sum_dep >= f["delta_min"])
            model.addConstr(sum_arr - sum_dep <= f["delta_max"])

    # Aircraft-only operated connections
    for s_idx in range(num_scenarios):
        for conn in aircraft_conn:
            i = flight_dict[conn["flight_i"]]
            j = flight_dict[conn["flight_j"]]
            min_t = conn["min_connection_time"]
            sum_dep_j = gp.quicksum(x_dep[s_idx, j, t] for t in range(T))
            sum_arr_i = gp.quicksum(x_arr[s_idx, i, t] for t in range(T))
            model.addConstr(sum_dep_j - sum_arr_i >= min_t)

    # Capacity envelopes
    for s_idx, sc in enumerate(scenarios):
        cond = sc["operating_conditions"]
        for a in airports:
            a_cond = cond[a]
            for t in range(T):
                weather = a_cond[t]
                seg_list = envelopes[a][weather]
                if t < T - 1:
                    dep_expr = gp.quicksum(
                        x_dep[s_idx, f_idx, t] - x_dep[s_idx, f_idx, t + 1]
                        for f_idx in dep_flights[a]
                    )
                    arr_expr = gp.quicksum(
                        x_arr[s_idx, f_idx, t] - x_arr[s_idx, f_idx, t + 1]
                        for f_idx in arr_flights[a]
                    )
                else:
                    dep_expr = gp.quicksum(
                        x_dep[s_idx, f_idx, t] for f_idx in dep_flights[a]
                    )
                    arr_expr = gp.quicksum(
                        x_arr[s_idx, f_idx, t] for f_idx in arr_flights[a]
                    )
                for seg in seg_list:
                    a_coef = seg["a"]
                    b_coef = seg["b"]
                    Q = seg["Q"]
                    model.addConstr(a_coef * dep_expr + b_coef * arr_expr <= Q)

    # Objective function
    obj = rho * gp.quicksum(disp[f_idx] for f_idx in range(num_flights))
    for s_idx, sc in enumerate(scenarios):
        prob = sc["probability"]
        for f_idx, f in enumerate(flights):
            obj += (1 - rho) * prob * (
                f["cost_dep_delay"] * v_dep[s_idx, f_idx] +
                f["cost_arr_delay"] * v_arr[s_idx, f_idx]
            )
    model.setObjective(obj, GRB.MINIMIZE)

    # Callback for logging incumbents
    class IncumbentCallback:
        def __init__(self, logger, x_dep, x_arr, v_dep, v_arr, s_dep,
                     flights, scenarios, T, rho):
            self.logger = logger
            self.x_dep = x_dep
            self.x_arr = x_arr
            self.v_dep = v_dep
            self.v_arr = v_arr
            self.s_dep = s_dep
            self.flights = flights
            self.scenarios = scenarios
            self.T = T
            self.rho = rho
            self.best_obj = float('inf')

        def __call__(self, model, where):
            if where == GRB.Callback.MIPSOL:
                obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                if obj_val < self.best_obj:
                    self.best_obj = obj_val
                    if self.logger:
                        s_dep_vals = model.cbGetSolution(self.s_dep)
                        x_dep_vals = model.cbGetSolution(self.x_dep)
                        x_arr_vals = model.cbGetSolution(self.x_arr)
                        v_dep_vals = model.cbGetSolution(self.v_dep)
                        v_arr_vals = model.cbGetSolution(self.v_arr)

                        sol = {
                            "objective_value": obj_val,
                            "schedule": {},
                            "scenario_solutions": []
                        }
                        for f_idx, f in enumerate(self.flights):
                            sd = int(round(s_dep_vals[f_idx]))
                            sol["schedule"][str(f["id"])] = {
                                "scheduled_dep": sd,
                                "scheduled_arr": sd + f["delta_sch"]
                            }
                        for s_idx, sc in enumerate(self.scenarios):
                            x_dep_dict = {}
                            x_arr_dict = {}
                            v_dep_dict = {}
                            v_arr_dict = {}
                            for f_idx, f in enumerate(self.flights):
                                fid = f["id"]
                                for t in range(self.T):
                                    period = t + 1
                                    x_dep_dict[f"{fid}_{period}"] = int(round(x_dep_vals[s_idx, f_idx, t]))
                                    x_arr_dict[f"{fid}_{period}"] = int(round(x_arr_vals[s_idx, f_idx, t]))
                                v_dep_dict[str(fid)] = v_dep_vals[s_idx, f_idx]
                                v_arr_dict[str(fid)] = v_arr_vals[s_idx, f_idx]
                            sol["scenario_solutions"].append({
                                "scenario_id": sc["id"],
                                "x_dep": x_dep_dict,
                                "x_arr": x_arr_dict,
                                "v_dep": v_dep_dict,
                                "v_arr": v_arr_dict
                            })
                        self.logger.log_solution(obj_val, sol)

    if logger:
        cb = IncumbentCallback(logger, x_dep, x_arr, v_dep, v_arr, s_dep,
                               flights, scenarios, T, rho)
        model.optimize(cb)
    else:
        model.optimize()

    # Build final solution
    if model.SolCount == 0:
        # Fallback trivial solution (requested times, zero delays)
        schedule = {}
        for f in flights:
            schedule[str(f["id"])] = {
                "scheduled_dep": f["dep_period"],
                "scheduled_arr": f["arr_period"]
            }
        scenario_solutions = []
        for sc in scenarios:
            x_dep_dict = {}
            x_arr_dict = {}
            v_dep_dict = {}
            v_arr_dict = {}
            for f in flights:
                fid = f["id"]
                s_dep_val = f["dep_period"]
                s_arr_val = f["arr_period"]
                for t in range(T):
                    period = t + 1
                    x_dep_dict[f"{fid}_{period}"] = 1 if period <= s_dep_val else 0
                    x_arr_dict[f"{fid}_{period}"] = 1 if period <= s_arr_val else 0
                v_dep_dict[str(fid)] = 0.0
                v_arr_dict[str(fid)] = 0.0
            scenario_solutions.append({
                "scenario_id": sc["id"],
                "x_dep": x_dep_dict,
                "x_arr": x_arr_dict,
                "v_dep": v_dep_dict,
                "v_arr": v_arr_dict
            })
        solution = {
            "objective_value": 0.0,
            "schedule": schedule,
            "scenario_solutions": scenario_solutions
        }
    else:
        obj_val = model.ObjVal
        schedule = {}
        for f_idx, f in enumerate(flights):
            sd = int(round(s_dep[f_idx].X))
            schedule[str(f["id"])] = {
                "scheduled_dep": sd,
                "scheduled_arr": sd + f["delta_sch"]
            }
        scenario_solutions = []
        for s_idx, sc in enumerate(scenarios):
            x_dep_dict = {}
            x_arr_dict = {}
            v_dep_dict = {}
            v_arr_dict = {}
            for f_idx, f in enumerate(flights):
                fid = f["id"]
                for t in range(T):
                    period = t + 1
                    x_dep_dict[f"{fid}_{period}"] = int(round(x_dep[s_idx, f_idx, t].X))
                    x_arr_dict[f"{fid}_{period}"] = int(round(x_arr[s_idx, f_idx, t].X))
                v_dep_dict[str(fid)] = v_dep[s_idx, f_idx].X
                v_arr_dict[str(fid)] = v_arr[s_idx, f_idx].X
            scenario_solutions.append({
                "scenario_id": sc["id"],
                "x_dep": x_dep_dict,
                "x_arr": x_arr_dict,
                "v_dep": v_dep_dict,
                "v_arr": v_arr_dict
            })
        solution = {
            "objective_value": obj_val,
            "schedule": schedule,
            "scenario_solutions": scenario_solutions
        }

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()