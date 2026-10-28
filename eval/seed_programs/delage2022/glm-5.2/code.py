import argparse
import json
import math
from gurobipy import Model, GRB, quicksum

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path) as f:
        instance = json.load(f)

    problem_type = instance['problem_type']
    machine_length = instance['machine_length']
    num_products = instance['num_products']
    num_patterns = instance['num_patterns']
    cost_setup = instance['cost_setup_pattern']
    cost_rep = instance['cost_repetition']
    M = instance['M']

    products = instance['products']
    product_ids = [p['id'] for p in products]
    lengths = {p['id']: p['length'] for p in products}
    demands = {p['id']: p['demand'] for p in products}
    N_i = {p['id']: p['N_i'] for p in products}

    is_closed = (problem_type == 'closed_ended_cscp')
    is_knife = (problem_type == 'cscp_with_knife_dependent_setups')
    cost_knife = instance.get('cost_knife_setup', 0) if is_knife else 0

    def build_solution(y_vals, r_vals, a_vals, x_vals, obj_val):
        patterns_out = {}
        for j in range(num_patterns):
            used = bool(round(y_vals[j]))
            reps = int(round(r_vals[j]))
            pieces = {str(pid): int(round(a_vals[pid, j])) for pid in product_ids}
            pattern_dict = {
                'used': used,
                'repetitions': reps,
                'pieces': pieces,
            }
            if is_knife:
                pattern_dict['knife_count'] = sum(pieces.values())
            patterns_out[str(j)] = pattern_dict

        extra = {str(pid): (0 if is_closed else int(round(x_vals[pid]))) for pid in product_ids}

        prod_quantities = {}
        for pid in product_ids:
            for j in range(num_patterns):
                key = f'product_{pid}_pattern_{j}'
                prod_quantities[key] = int(round(a_vals[pid, j])) * int(round(r_vals[j]))

        return {
            'objective_value': float(obj_val),
            'patterns': patterns_out,
            'extra_pieces': extra,
            'production_quantities': prod_quantities,
        }

    def is_feasible(y_h, r_h, a_h, x_h):
        for pid in product_ids:
            prod = sum(a_h[pid, j] * r_h[j] for j in range(num_patterns))
            if prod + x_h[pid] < demands[pid]:
                return False
        for j in range(num_patterns):
            total_len = sum(lengths[pid] * a_h[pid, j] for pid in product_ids)
            if total_len > machine_length:
                return False
            if y_h[j] == 0 and r_h[j] > 0:
                return False
            if y_h[j] == 0:
                for pid in product_ids:
                    if a_h[pid, j] > 0:
                        return False
        if not is_closed:
            total_extra = sum(x_h.values())
            total_reps = sum(r_h.values())
            if total_extra > total_reps:
                return False
        for j in range(num_patterns - 1):
            if r_h[j] < r_h[j + 1]:
                return False
        return True

    def heuristic():
        nonzero_products = [pid for pid in product_ids if demands[pid] > 0]
        y_h = {j: 0 for j in range(num_patterns)}
        r_h = {j: 0 for j in range(num_patterns)}
        a_h = {(pid, j): 0 for pid in product_ids for j in range(num_patterns)}
        x_h = {pid: 0 for pid in product_ids}

        if not nonzero_products:
            return y_h, r_h, a_h, x_h, 0.0

        pattern_list = []
        for pid in nonzero_products:
            if N_i[pid] == 0:
                continue
            a_val = N_i[pid]
            reps = math.ceil(demands[pid] / a_val)
            pattern_list.append((pid, a_val, reps))

        pattern_list.sort(key=lambda x: -x[2])

        if len(pattern_list) <= num_patterns:
            for idx, (pid, a_val, reps) in enumerate(pattern_list):
                y_h[idx] = 1
                r_h[idx] = reps
                a_h[pid, idx] = a_val
            if not is_closed:
                total_reps = sum(r_h.values())
                for pid in nonzero_products:
                    if N_i[pid] == 0:
                        x_h[pid] = min(demands[pid], total_reps)
                        total_reps -= x_h[pid]
        else:
            remaining = {pid: demands[pid] for pid in nonzero_products if N_i[pid] > 0}
            for j in range(num_patterns):
                if all(v <= 0 for v in remaining.values()):
                    break
                used_length = 0
                for pid in sorted(remaining.keys(), key=lambda p: -lengths[p]):
                    if remaining[pid] <= 0:
                        continue
                    max_fit = min(N_i[pid], remaining[pid], (machine_length - used_length) // lengths[pid])
                    if max_fit > 0:
                        a_h[pid, j] = max_fit
                        used_length += max_fit * lengths[pid]
                if used_length > 0:
                    y_h[j] = 1
                    max_reps = 1
                    for pid in product_ids:
                        if a_h[pid, j] > 0 and remaining.get(pid, 0) > 0:
                            needed = math.ceil(remaining[pid] / a_h[pid, j])
                            max_reps = max(max_reps, needed)
                    r_h[j] = max_reps
                    for pid in product_ids:
                        if a_h[pid, j] > 0:
                            remaining[pid] -= a_h[pid, j] * max_reps
            if not is_closed:
                total_reps = sum(r_h.values())
                for pid in product_ids:
                    shortfall = max(0, demands[pid] - sum(a_h[pid, j] * r_h[j] for j in range(num_patterns)))
                    if shortfall > 0:
                        give = min(shortfall, total_reps)
                        x_h[pid] = give
                        total_reps -= give
            order = sorted(range(num_patterns), key=lambda j: -r_h[j])
            new_y = {j: 0 for j in range(num_patterns)}
            new_r = {j: 0 for j in range(num_patterns)}
            new_a = {(pid, j): 0 for pid in product_ids for j in range(num_patterns)}
            for new_idx, old_idx in enumerate(order):
                new_y[new_idx] = y_h[old_idx]
                new_r[new_idx] = r_h[old_idx]
                for pid in product_ids:
                    new_a[pid, new_idx] = a_h[pid, old_idx]
            y_h, r_h, a_h = new_y, new_r, new_a

        obj = cost_setup * sum(y_h.values()) + cost_rep * sum(r_h.values())
        if is_knife:
            obj += cost_knife * sum(a_h.values())
        return y_h, r_h, a_h, x_h, obj

    # Compute heuristic
    y_h, r_h, a_h, x_h, obj_h = None, None, None, None, float('inf')
    try:
        y_h, r_h, a_h, x_h, obj_h = heuristic()
        if is_feasible(y_h, r_h, a_h, x_h):
            if logger:
                sol_h = build_solution(y_h, r_h, a_h, x_h, obj_h)
                logger.log_solution(obj_h, sol_h)
        else:
            y_h = None
    except Exception:
        y_h = None

    # Callback definition
    last_obj = [obj_h if y_h is not None else float('inf')]
    best_sol_from_callback = [None]

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj_val < last_obj[0]:
                last_obj[0] = obj_val
                y_vals = {}
                r_vals = {}
                for j in range(num_patterns):
                    y_vals[j] = model.cbGetSolution(y[j])
                    r_vals[j] = model.cbGetSolution(r[j])
                x_vals = {}
                for pid in product_ids:
                    x_vals[pid] = model.cbGetSolution(x[pid])
                a_vals = {}
                for pid in product_ids:
                    for j in range(num_patterns):
                        a_vals[pid, j] = model.cbGetSolution(a[pid, j])
                sol = build_solution(y_vals, r_vals, a_vals, x_vals, obj_val)
                best_sol_from_callback[0] = sol
                if logger:
                    logger.log_solution(obj_val, sol)

    model = Model('cscp')
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('OutputFlag', 0)
    model.setParam('Presolve', 2)
    time_limit = max(args.time_limit - 3, 1)
    model.setParam('TimeLimit', time_limit)

    y = {}
    r = {}
    a = {}
    x = {}
    b = {}
    w = {}

    for j in range(num_patterns):
        y[j] = model.addVar(vtype=GRB.BINARY, name=f'y_{j}')
        r[j] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=M, name=f'r_{j}')

    for pid in product_ids:
        if is_closed:
            x[pid] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=0, name=f'x_{pid}')
        else:
            x[pid] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=demands[pid], name=f'x_{pid}')

    bits_needed = {}
    for pid in product_ids:
        bits_needed[pid] = max(1, N_i[pid].bit_length()) if N_i[pid] > 0 else 0

    for pid in product_ids:
        for j in range(num_patterns):
            a[pid, j] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=N_i[pid], name=f'a_{pid}_{j}')
            for k in range(bits_needed[pid]):
                b[pid, j, k] = model.addVar(vtype=GRB.BINARY, name=f'b_{pid}_{j}_{k}')
                w[pid, j, k] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=M, name=f'w_{pid}_{j}_{k}')

    model.update()

    for j in range(num_patterns):
        for pid in product_ids:
            model.addConstr(a[pid, j] <= N_i[pid] * y[j])
        model.addConstr(r[j] <= M * y[j])
        model.addConstr(quicksum(lengths[pid] * a[pid, j] for pid in product_ids) <= machine_length)
        for pid in product_ids:
            if bits_needed[pid] > 0:
                model.addConstr(a[pid, j] == quicksum((2**k) * b[pid, j, k] for k in range(bits_needed[pid])))
                for k in range(bits_needed[pid]):
                    model.addConstr(w[pid, j, k] <= M * b[pid, j, k])
                    model.addConstr(w[pid, j, k] <= r[j])
                    model.addConstr(w[pid, j, k] >= r[j] - M * (1 - b[pid, j, k]))

    for pid in product_ids:
        if bits_needed[pid] > 0:
            prod_expr = quicksum(
                quicksum((2**k) * w[pid, j, k] for k in range(bits_needed[pid]))
                for j in range(num_patterns)
            )
        else:
            prod_expr = 0
        model.addConstr(prod_expr + x[pid] >= demands[pid])

    if not is_closed:
        model.addConstr(quicksum(x[pid] for pid in product_ids) <= quicksum(r[j] for j in range(num_patterns)))

    for j in range(num_patterns - 1):
        model.addConstr(r[j] >= r[j + 1])

    obj_expr = cost_setup * quicksum(y[j] for j in range(num_patterns)) + cost_rep * quicksum(r[j] for j in range(num_patterns))
    if is_knife:
        obj_expr += cost_knife * quicksum(a[pid, j] for pid in product_ids for j in range(num_patterns))

    model.setObjective(obj_expr, GRB.MINIMIZE)

    # Warm start
    if y_h is not None:
        try:
            for j in range(num_patterns):
                y[j].Start = y_h[j]
                r[j].Start = r_h[j]
            for pid in product_ids:
                x[pid].Start = x_h[pid]
                for j in range(num_patterns):
                    a[pid, j].Start = a_h[pid, j]
                    val = a_h[pid, j]
                    for k in range(bits_needed[pid]):
                        b[pid, j, k].Start = (val >> k) & 1
                        w[pid, j, k].Start = r_h[j] * ((val >> k) & 1)
        except Exception:
            pass

    try:
        model.optimize(callback)
    except Exception:
        pass

    # Get best solution
    best_obj = obj_h if y_h is not None else float('inf')
    best_solution = None

    try:
        if model.SolCount > 0:
            mip_obj = model.ObjVal
            if mip_obj < best_obj:
                best_obj = mip_obj
                y_vals = {j: y[j].X for j in range(num_patterns)}
                r_vals = {j: r[j].X for j in range(num_patterns)}
                x_vals = {pid: x[pid].X for pid in product_ids}
                a_vals = {}
                for pid in product_ids:
                    for j in range(num_patterns):
                        a_vals[pid, j] = a[pid, j].X
                best_solution = build_solution(y_vals, r_vals, a_vals, x_vals, best_obj)
    except Exception:
        pass

    if best_solution is None and best_sol_from_callback[0] is not None:
        best_solution = best_sol_from_callback[0]

    if best_solution is None and y_h is not None:
        best_solution = build_solution(y_h, r_h, a_h, x_h, obj_h)

    if best_solution is None:
        best_solution = build_solution(
            {j: 0 for j in range(num_patterns)},
            {j: 0 for j in range(num_patterns)},
            {(pid, j): 0 for pid in product_ids for j in range(num_patterns)},
            {pid: 0 for pid in product_ids},
            0.0
        )

    with open(args.solution_path, 'w') as f:
        json.dump(best_solution, f, indent=2)

if __name__ == '__main__':
    main()