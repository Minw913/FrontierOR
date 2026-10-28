import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    start_time = time.time()

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    n_fac = data['num_facilities']
    n_cust = data['num_customers']
    demands = data['demands']
    capacities = data['capacities']
    fc1 = data['fixed_costs_obj1']
    fc2 = data['fixed_costs_obj2']
    ac1 = data['assignment_costs_obj1']
    ac2 = data['assignment_costs_obj2']

    def elapsed():
        return time.time() - start_time

    def remaining():
        return max(args.time_limit - elapsed(), 0.5)

    def solve_model(obj_mode, z2_bound=None, z1_bound=None):
        """
        Solve the CFLP with specified objective and optional epsilon constraints.
        obj_mode: 'z1', 'z2', or 'weighted'
        Returns (z1, z2, open_facilities, assignments) or None
        """
        time_rem = remaining()
        if time_rem < 1:
            return None

        m = gp.Model()
        m.Params.Seed = 0
        m.Params.MIPGap = 1e-4
        m.Params.NumericFocus = 0
        m.Params.Threads = 1
        m.Params.OutputFlag = 0
        m.Params.TimeLimit = time_rem

        y = m.addVars(n_fac, vtype=GRB.BINARY, name="y")
        x = m.addVars(n_fac, n_cust, vtype=GRB.BINARY, name="x")

        z1_expr = gp.quicksum(fc1[i] * y[i] for i in range(n_fac)) + \
                  gp.quicksum(ac1[i][j] * x[i, j] for i in range(n_fac) for j in range(n_cust))
        z2_expr = gp.quicksum(fc2[i] * y[i] for i in range(n_fac)) + \
                  gp.quicksum(ac2[i][j] * x[i, j] for i in range(n_fac) for j in range(n_cust))

        if obj_mode == 'z1':
            m.setObjective(z1_expr, GRB.MINIMIZE)
        elif obj_mode == 'z2':
            m.setObjective(z2_expr, GRB.MINIMIZE)
        else:
            m.setObjective(z1_expr + z2_expr, GRB.MINIMIZE)

        # Each customer assigned to exactly one facility
        for j in range(n_cust):
            m.addConstr(gp.quicksum(x[i, j] for i in range(n_fac)) == 1)

        # Capacity constraints
        for i in range(n_fac):
            m.addConstr(gp.quicksum(demands[j] * x[i, j] for j in range(n_cust)) <= capacities[i] * y[i])

        # Linking constraints
        for i in range(n_fac):
            for j in range(n_cust):
                m.addConstr(x[i, j] <= y[i])

        # Epsilon constraints
        if z2_bound is not None:
            m.addConstr(z2_expr <= z2_bound)
        if z1_bound is not None:
            m.addConstr(z1_expr <= z1_bound)

        m.optimize()

        if m.SolCount > 0:
            y_val = [int(round(y[i].X)) for i in range(n_fac)]
            x_val = [[int(round(x[i, j].X)) for j in range(n_cust)] for i in range(n_fac)]

            z1 = sum(fc1[i] * y_val[i] for i in range(n_fac)) + \
                 sum(ac1[i][j] * x_val[i][j] for i in range(n_fac) for j in range(n_cust))
            z2 = sum(fc2[i] * y_val[i] for i in range(n_fac)) + \
                 sum(ac2[i][j] * x_val[i][j] for i in range(n_fac) for j in range(n_cust))

            assignments = {}
            for j in range(n_cust):
                for i in range(n_fac):
                    if x_val[i][j] == 1:
                        assignments[str(j)] = i
                        break

            open_facs = [i for i in range(n_fac) if y_val[i] == 1]

            return (z1, z2, open_facs, assignments)

        return None

    # Storage for Pareto points: (z1, z2) -> (open_facilities, assignments)
    pareto_set = {}

    def add_point(result):
        if result is None:
            return
        z1, z2, open_facs, assignments = result
        key = (z1, z2)
        if key not in pareto_set:
            pareto_set[key] = (open_facs, assignments)

    def is_dominated(z1, z2):
        for (pz1, pz2) in pareto_set:
            if pz1 <= z1 and pz2 <= z2 and (pz1 < z1 or pz2 < z2):
                return True
        return False

    def remove_dominated():
        keys = list(pareto_set.keys())
        to_remove = set()
        for i in range(len(keys)):
            if keys[i] in to_remove:
                continue
            for j in range(len(keys)):
                if i == j or keys[j] in to_remove:
                    continue
                # Check if keys[j] dominates keys[i]
                if keys[j][0] <= keys[i][0] and keys[j][1] <= keys[i][1] and \
                   (keys[j][0] < keys[i][0] or keys[j][1] < keys[i][1]):
                    to_remove.add(keys[i])
                    break
        for k in to_remove:
            del pareto_set[k]

    def build_output():
        pareto_front = sorted(pareto_set.keys(), key=lambda p: (p[0], p[1]))
        if not pareto_front:
            return None

        best_point = min(pareto_front, key=lambda p: 0.5 * p[0] + 0.5 * p[1])
        objective_value = 0.5 * best_point[0] + 0.5 * best_point[1]

        solutions = []
        for (z1, z2) in pareto_front:
            open_facs, assignments = pareto_set[(z1, z2)]
            solutions.append({
                "z1": z1,
                "z2": z2,
                "open_facilities": open_facs,
                "assignments": assignments
            })

        return {
            "objective_value": objective_value,
            "pareto_front": [[z1, z2] for z1, z2 in pareto_front],
            "solutions": solutions
        }

    def log_current_best():
        if logger and pareto_set:
            output = build_output()
            if output:
                logger.log_solution(output["objective_value"], output)

    # Step 1: Solve weighted sum for a quick initial solution
    result = solve_model('weighted')
    if result:
        add_point(result)
        log_current_best()

    # Step 2: Minimize z1 (leftmost Pareto point)
    if remaining() > 1:
        result = solve_model('z1')
        if result:
            add_point(result)
            log_current_best()

    # Step 3: Minimize z2 (rightmost Pareto point)
    if remaining() > 1:
        result = solve_model('z2')
        if result:
            add_point(result)
            log_current_best()

    # Step 4: Epsilon-constraint enumeration - trace from left to right
    # Start from the point with minimum z1, iteratively constrain z2
    if pareto_set and remaining() > 2:
        sorted_points = sorted(pareto_set.keys(), key=lambda p: p[0])
        # Find the point with minimum z1
        leftmost = sorted_points[0]
        z2_bound = leftmost[1] - 1

        while remaining() > 2 and z2_bound >= 0:
            result = solve_model('z1', z2_bound=z2_bound)
            if result is None:
                break
            z1, z2, _, _ = result
            add_point(result)
            log_current_best()
            z2_bound = z2 - 1

    # Step 5: Also trace from right to left (minimize z2 with z1 constraint)
    if pareto_set and remaining() > 2:
        sorted_points = sorted(pareto_set.keys(), key=lambda p: p[1])
        # Find the point with minimum z2
        bottommost = sorted_points[0]
        z1_bound = bottommost[0] - 1

        while remaining() > 2 and z1_bound >= 0:
            result = solve_model('z2', z1_bound=z1_bound)
            if result is None:
                break
            z1, z2, _, _ = result
            add_point(result)
            log_current_best()
            z1_bound = z1 - 1

    # Step 6: Fill gaps - for each pair of adjacent points, try to find points between them
    if pareto_set and remaining() > 2:
        sorted_points = sorted(pareto_set.keys(), key=lambda p: p[0])
        for idx in range(len(sorted_points) - 1):
            if remaining() <= 2:
                break
            p_left = sorted_points[idx]
            p_right = sorted_points[idx + 1]
            # Try to find a point with z1 between p_left[0] and p_right[0]
            # and z2 between p_right[1] and p_left[1]
            # Use epsilon constraint: z2 <= p_left[1] - 1, minimize z1
            # But we already did this in step 4. Let's try the other direction.
            # Constrain z1 <= p_right[0] - 1, minimize z2
            if p_right[0] - 1 > p_left[0]:
                result = solve_model('z2', z1_bound=p_right[0] - 1)
                if result:
                    z1, z2, _, _ = result
                    if not is_dominated(z1, z2):
                        add_point(result)
                        log_current_best()

    # Final cleanup: remove any dominated points
    remove_dominated()

    # Build final output
    output = build_output()

    if output is None:
        # Fallback: shouldn't happen but just in case
        output = {
            "objective_value": 0.0,
            "pareto_front": [],
            "solutions": []
        }

    # Final log
    if logger:
        logger.log_solution(output["objective_value"], output)

    with open(args.solution_path, 'w') as f:
        json.dump(output, f, indent=2)


if __name__ == '__main__':
    main()