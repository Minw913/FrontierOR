import argparse
import json
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


class IncumbentLogger:
    def __init__(self, logger, num_facilities, num_customers, customer_demands,
                 customer_to_facilities, sense, problem_type):
        self.logger = logger
        self.num_facilities = num_facilities
        self.num_customers = num_customers
        self.customer_demands = customer_demands
        self.customer_to_facilities = customer_to_facilities
        self.sense = sense
        self.problem_type = problem_type
        self.best_obj = None

    def __call__(self, model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            improved = False
            if self.best_obj is None:
                improved = True
            elif self.sense == 'minimize' and obj < self.best_obj - 1e-9:
                improved = True
            elif self.sense == 'maximize' and obj > self.best_obj + 1e-9:
                improved = True

            if improved:
                self.best_obj = obj
                if self.logger:
                    x_vals = model.cbGetSolution(model._vars_x)
                    y_vals = model.cbGetSolution(model._vars_y)
                    open_facs = [j for j in range(self.num_facilities) if x_vals[j] > 0.5]
                    open_set = set(open_facs)
                    covered_demand = 0
                    for i in range(self.num_customers):
                        facs = self.customer_to_facilities[i]
                        if any(j in open_set for j in facs):
                            covered_demand += self.customer_demands[i]

                    sol = {
                        "objective_value": float(obj),
                        "primary_problem_type": self.problem_type,
                        "results": {
                            "PSCLP": {"objective_value": 0.0, "open_facilities": [], "covered_demand": 0.0},
                            "MCLP": {"objective_value": 0.0, "open_facilities": [], "covered_demand": 0.0}
                        }
                    }
                    if self.problem_type == "PSCLP":
                        sol["results"]["PSCLP"] = {
                            "objective_value": float(obj),
                            "open_facilities": open_facs,
                            "covered_demand": float(covered_demand)
                        }
                    else:
                        sol["results"]["MCLP"] = {
                            "objective_value": float(obj),
                            "open_facilities": open_facs,
                            "covered_demand": float(covered_demand)
                        }
                    self.logger.log_solution(float(obj), sol)


def solve_psclp(instance, time_limit, customer_to_facilities_list, logger=None):
    m = gp.Model("PSCLP")
    m.Params.Seed = 0
    m.Params.MIPGap = 1e-4
    m.Params.NumericFocus = 0
    m.Params.Threads = 1
    m.Params.TimeLimit = max(1.0, time_limit)
    m.Params.OutputFlag = 0

    num_facilities = instance["num_facilities"]
    num_customers = instance["num_customers"]
    facility_cost = instance["facility_cost"]
    customer_demands = instance["customer_demands"]
    covering_demand_D = instance.get("covering_demand_D")
    if covering_demand_D is None:
        covering_demand_D = instance["covering_demand_fraction"] * instance["total_demand_D_bar"]

    x = m.addVars(num_facilities, vtype=GRB.BINARY, name="x")
    y = m.addVars(num_customers, vtype=GRB.BINARY, name="y")
    m._vars_x = x
    m._vars_y = y

    m.setObjective(gp.quicksum(facility_cost[j] * x[j] for j in range(num_facilities)), GRB.MINIMIZE)

    for i in range(num_customers):
        facs = customer_to_facilities_list[i]
        if facs:
            m.addConstr(y[i] <= gp.quicksum(x[j] for j in facs))
        else:
            m.addConstr(y[i] == 0)

    m.addConstr(gp.quicksum(customer_demands[i] * y[i] for i in range(num_customers)) >= covering_demand_D)

    cb = None
    if logger:
        cb = IncumbentLogger(logger, num_facilities, num_customers, customer_demands,
                             customer_to_facilities_list, "minimize", "PSCLP")
    m.optimize(cb)

    if m.SolCount > 0:
        open_facs = [j for j in range(num_facilities) if x[j].X > 0.5]
        open_set = set(open_facs)
        covered_demand = 0
        for i in range(num_customers):
            facs = customer_to_facilities_list[i]
            if any(j in open_set for j in facs):
                covered_demand += customer_demands[i]
        return m.ObjVal, open_facs, covered_demand
    return None, [], 0.0


def solve_mclp(instance, time_limit, customer_to_facilities_list, logger=None):
    m = gp.Model("MCLP")
    m.Params.Seed = 0
    m.Params.MIPGap = 1e-4
    m.Params.NumericFocus = 0
    m.Params.Threads = 1
    m.Params.TimeLimit = max(1.0, time_limit)
    m.Params.OutputFlag = 0

    num_facilities = instance["num_facilities"]
    num_customers = instance["num_customers"]
    facility_cost = instance["facility_cost"]
    customer_demands = instance["customer_demands"]
    budget_B = instance["budget_B"]

    x = m.addVars(num_facilities, vtype=GRB.BINARY, name="x")
    y = m.addVars(num_customers, vtype=GRB.BINARY, name="y")
    m._vars_x = x
    m._vars_y = y

    m.setObjective(gp.quicksum(customer_demands[i] * y[i] for i in range(num_customers)), GRB.MAXIMIZE)

    for i in range(num_customers):
        facs = customer_to_facilities_list[i]
        if facs:
            m.addConstr(y[i] <= gp.quicksum(x[j] for j in facs))
        else:
            m.addConstr(y[i] == 0)

    m.addConstr(gp.quicksum(facility_cost[j] * x[j] for j in range(num_facilities)) <= budget_B)

    cb = None
    if logger:
        cb = IncumbentLogger(logger, num_facilities, num_customers, customer_demands,
                             customer_to_facilities_list, "maximize", "MCLP")
    m.optimize(cb)

    if m.SolCount > 0:
        open_facs = [j for j in range(num_facilities) if x[j].X > 0.5]
        open_set = set(open_facs)
        covered_demand = 0
        for i in range(num_customers):
            facs = customer_to_facilities_list[i]
            if any(j in open_set for j in facs):
                covered_demand += customer_demands[i]
        return m.ObjVal, open_facs, covered_demand
    return 0.0, [], 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    num_customers = instance["num_customers"]
    customer_to_facilities_raw = instance["customer_to_facilities_coverage"]
    customer_to_facilities_list = [customer_to_facilities_raw.get(str(i), []) for i in range(num_customers)]

    # Determine primary problem type
    primary_type = instance.get("primary_problem_type")
    if primary_type is None:
        problem_types = instance.get("problem_types")
        if problem_types:
            primary_type = problem_types[0]
        else:
            primary_type = "PSCLP"

    sense = "minimize" if primary_type == "PSCLP" else "maximize"
    logger = SolutionLogger(args.log_path, sense=sense) if args.log_path else None

    start_time = time.time()

    # Solve primary problem
    primary_time = max(1.0, args.time_limit * 0.5)
    primary_obj, primary_open, primary_covered = None, [], 0.0
    try:
        if primary_type == "PSCLP":
            primary_obj, primary_open, primary_covered = solve_psclp(
                instance, primary_time, customer_to_facilities_list, logger)
        else:
            primary_obj, primary_open, primary_covered = solve_mclp(
                instance, primary_time, customer_to_facilities_list, logger)
    except Exception as e:
        pass

    elapsed = time.time() - start_time
    secondary_time = max(1.0, args.time_limit - elapsed)

    # Solve secondary problem
    secondary_obj, secondary_open, secondary_covered = 0.0, [], 0.0
    try:
        if primary_type == "PSCLP":
            secondary_obj, secondary_open, secondary_covered = solve_mclp(
                instance, secondary_time, customer_to_facilities_list, None)
        else:
            secondary_obj, secondary_open, secondary_covered = solve_psclp(
                instance, secondary_time, customer_to_facilities_list, None)
    except Exception as e:
        pass

    if primary_type == "PSCLP":
        psclp_obj, psclp_open, psclp_covered = primary_obj, primary_open, primary_covered
        mclp_obj, mclp_open, mclp_covered = secondary_obj, secondary_open, secondary_covered
    else:
        mclp_obj, mclp_open, mclp_covered = primary_obj, primary_open, primary_covered
        psclp_obj, psclp_open, psclp_covered = secondary_obj, secondary_open, secondary_covered

    if psclp_obj is None:
        psclp_obj = 1e12

    output = {
        "objective_value": float(primary_obj) if primary_obj is not None else 0.0,
        "primary_problem_type": primary_type,
        "results": {
            "PSCLP": {
                "objective_value": float(psclp_obj),
                "open_facilities": psclp_open,
                "covered_demand": float(psclp_covered)
            },
            "MCLP": {
                "objective_value": float(mclp_obj),
                "open_facilities": mclp_open,
                "covered_demand": float(mclp_covered)
            }
        }
    }

    if logger and primary_obj is not None:
        final_dict = {
            "objective_value": float(primary_obj),
            "primary_problem_type": primary_type,
            "results": {
                "PSCLP": {
                    "objective_value": float(psclp_obj),
                    "open_facilities": psclp_open,
                    "covered_demand": float(psclp_covered)
                },
                "MCLP": {
                    "objective_value": float(mclp_obj),
                    "open_facilities": mclp_open,
                    "covered_demand": float(mclp_covered)
                }
            }
        }
        logger.log_solution(float(primary_obj), final_dict)

    with open(args.solution_path, 'w') as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()