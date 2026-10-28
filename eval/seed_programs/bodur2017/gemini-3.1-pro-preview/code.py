import sys
import json
import argparse
import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True, help="Path to the instance JSON file.")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to write the output solution JSON file.")
    parser.add_argument("--time_limit", type=int, required=True, help="Maximum execution time in seconds.")
    parser.add_argument("--log_path", type=str, default=None, help="Path for progressive logging.")
    args = parser.parse_args()

    # Initialize SolutionLogger if requested
    if SolutionLogger and args.log_path:
        logger = SolutionLogger(args.log_path, sense="minimize")
    else:
        logger = None

    # Load instance data
    with open(args.instance_path, "r") as f:
        data = json.load(f)

    num_f = data["num_facilities"]
    num_c = data["num_customers"]
    num_s = data["num_scenarios"]

    # Parse facilities robustly to assure order matches ID
    fac_opening_cost = [0.0] * num_f
    fac_capacity = [0.0] * num_f
    for f_info in data["facilities"]:
        f_id = f_info["id"]
        fac_opening_cost[f_id] = f_info["opening_cost"]
        fac_capacity[f_id] = f_info["capacity"]

    # Parse transportation costs
    trans_cost = data["transportation_costs"]

    # Parse scenarios robustly to assure order matches scenario ID
    demands = [[0.0] * num_c for _ in range(num_s)]
    prob = [0.0] * num_s
    for s in data["scenarios"]:
        s_id = s["id"]
        demands[s_id] = s["demands"]
        prob[s_id] = s["probability"]

    # Set up the Gurobi model
    m = gp.Model("stochastic_facility_location")
    m.Params.TimeLimit = args.time_limit
    m.Params.Threads = 1  # Mandatory single CPU core restriction

    # Facilities decision variables
    x = m.addVars(num_f, vtype=GRB.BINARY, name="x")
    for f in range(num_f):
        x[f].Obj = fac_opening_cost[f]

    # Allocation decision variables y[s, f, c] mapped correctly
    # Generating object coefficients in order: s varies slowest, c varies fastest
    y_obj_coeffs = [prob[s] * trans_cost[f][c] for s in range(num_s) for f in range(num_f) for c in range(num_c)]
    y = m.addVars(num_s, num_f, num_c, vtype=GRB.CONTINUOUS, obj=y_obj_coeffs, name="y")

    m.ModelSense = GRB.MINIMIZE

    # Constraints: Demand must be fulfilled under every scenario
    m.addConstrs(
        (y.sum(s, '*', c) == demands[s][c] for s in range(num_s) for c in range(num_c)),
        name="satisfy_demand"
    )

    # Constraints: Capacity restriction per facility for each scenario
    m.addConstrs(
        (y.sum(s, f, '*') <= fac_capacity[f] * x[f] for s in range(num_s) for f in range(num_f)),
        name="facility_capacity"
    )

    # Constraints: Total capacity of opened facilities >= maximum possible demand
    max_tot_demand = max(sum(demands[s]) for s in range(num_s))
    m.addConstr(
        gp.quicksum(fac_capacity[f] * x[f] for f in range(num_f)) >= max_tot_demand,
        name="max_total_demand_coverage"
    )

    # Callback definition for progressive solution logging
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            # New incumbent solution found
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj_val)

    # Solve the model
    m.optimize(callback)

    # Extract answers and dump into JSON per output schema
    if m.SolCount > 0:
        open_facs = [f for f in range(num_f) if x[f].X > 0.5]
        out_x = {str(f): int(round(x[f].X)) for f in range(num_f)}
        out_y = {}
        for s in range(num_s):
            str_s = str(s)
            out_y[str_s] = {}
            for f in range(num_f):
                str_f = str(f)
                out_y[str_s][str_f] = {}
                for c in range(num_c):
                    # Guard bounding against tiny negative values linked to floating-point imperfections
                    out_y[str_s][str_f][str(c)] = max(0.0, y[s, f, c].X)

        solution = {
            "objective_value": m.ObjVal,
            "open_facilities": open_facs,
            "x": out_x,
            "y": out_y
        }

        with open(args.solution_path, "w") as f_out:
            json.dump(solution, f_out, indent=2)
    else:
        # Failsafe in case no solution is retrieved (should not occur under given time limits typically)
        sys.exit("No feasible solution obtained within the time limit.")

if __name__ == "__main__":
    main()