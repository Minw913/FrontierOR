import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    start_time = time.time()

    with open(args.instance_path) as f:
        instance = json.load(f)

    # Parse network
    legs = instance['network']['legs']
    num_legs = len(legs)
    alpha = float(instance.get('alpha', 1.0))
    leg_capacities = [float(leg['capacity']) * alpha for leg in legs]
    leg_id_to_idx = {leg['leg_id']: i for i, leg in enumerate(legs)}

    # Parse products
    products = instance['products']
    num_products = len(products)
    product_ids = [p['product_id'] for p in products]
    product_id_to_idx = {pid: i for i, pid in enumerate(product_ids)}
    fares = [float(p['fare']) for p in products]
    product_legs = [[leg_id_to_idx[lid] for lid in p['legs_used']] for p in products]

    # Parse segments
    segments = instance['segments']
    num_segments = len(segments)
    seg_lambda = [float(s['lambda_l']) for s in segments]
    seg_no_purchase = [float(s['no_purchase_preference']) for s in segments]

    seg_consideration = []
    seg_weight_dict = []
    for s in segments:
        cs = [product_id_to_idx[pid] for pid in s['consideration_set']]
        seg_consideration.append(cs)
        wd = {}
        for j, w in zip(cs, s['preference_vector']):
            wd[j] = float(w)
        seg_weight_dict.append(wd)

    T = instance['booking_horizon']['T']

    # Precompute leg usage sets for fast lookup
    product_leg_sets = [set(pl) for pl in product_legs]

    def compute_column(offer_set_indices):
        """Compute revenue coefficient and capacity consumption for an offer set."""
        offer_set = set(offer_set_indices)
        c_S = 0.0
        A = [0.0] * num_legs
        for m in range(num_segments):
            cs = seg_consideration[m]
            intersection = [j for j in cs if j in offer_set]
            if not intersection:
                continue
            D_m = sum(seg_weight_dict[m][j] for j in intersection) + seg_no_purchase[m]
            for j in intersection:
                p = seg_weight_dict[m][j] / D_m
                c_S += seg_lambda[m] * fares[j] * p
                for l in product_legs[j]:
                    A[l] += seg_lambda[m] * p
        return c_S, A

    # ---- Initial columns ----
    columns = []          # list of (offer_set_tuple, c_S, A)
    column_set = set()    # set of offer_set_tuple for dedup

    def add_initial_column(os_indices):
        os_tuple = tuple(sorted(os_indices))
        if os_tuple in column_set or not os_tuple:
            return
        c, A = compute_column(list(os_tuple))
        columns.append((os_tuple, c, A))
        column_set.add(os_tuple)

    # Single products
    for j in range(num_products):
        add_initial_column([j])
    # Full set
    add_initial_column(list(range(num_products)))
    # Each segment's consideration set
    for m in range(num_segments):
        add_initial_column(seg_consideration[m])

    if num_products == 0:
        solution = {"objective_value": 0.0, "active_columns": []}
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
        return

    # ---- Master LP ----
    master = gp.Model()
    master.setParam('Seed', 0)
    master.setParam('OutputFlag', 0)
    master.setParam('Threads', 1)
    master.setParam('NumericFocus', 0)

    # Create constraints first (empty for now)
    cap_constraints = []
    for l in range(num_legs):
        cap_constraints.append(
            master.addLConstr(gp.LinExpr(), GRB.LESS_EQUAL, leg_capacities[l], name=f'cap_{l}')
        )
    time_constraint = master.addLConstr(gp.LinExpr(), GRB.LESS_EQUAL, T, name='time')

    t_vars = []
    for i, (os, c, A) in enumerate(columns):
        col = gp.Column()
        for l in range(num_legs):
            if abs(A[l]) > 1e-12:
                col.addTerms(A[l], cap_constraints[l])
        col.addTerms(1.0, time_constraint)
        v = master.addVar(lb=0, obj=c, column=col, name=f't_{i}')
        t_vars.append(v)

    master.setAttr('ModelSense', GRB.MAXIMIZE)
    master.update()

    # ---- Subproblem MILP (reusable) ----
    sub = gp.Model()
    sub.setParam('Seed', 0)
    sub.setParam('OutputFlag', 0)
    sub.setParam('Threads', 1)
    sub.setParam('NumericFocus', 0)
    sub.setParam('MIPGap', 1e-4)

    x_vars = [sub.addVar(vtype=GRB.BINARY, name=f'x_{j}') for j in range(num_products)]

    u_vars = []
    z_vars = {}
    z_obj_info = []  # (var, m, j, weight) for objective updates

    for m in range(num_segments):
        W_max = sum(seg_weight_dict[m].values()) + seg_no_purchase[m]
        ub_u = 1.0 / seg_no_purchase[m]
        lb_u = 1.0 / W_max if W_max > 0 else ub_u
        u_m = sub.addVar(lb=lb_u, ub=ub_u, name=f'u_{m}')
        u_vars.append(u_m)

        for j in seg_consideration[m]:
            w = seg_weight_dict[m][j]
            z_mj = sub.addVar(lb=0, ub=ub_u, name=f'z_{m}_{j}')
            z_vars[(m, j)] = z_mj
            z_obj_info.append((z_mj, m, j, w))

            # Linearization: z = x * u
            sub.addConstr(z_mj <= u_m)
            sub.addConstr(z_mj <= ub_u * x_vars[j])
            sub.addConstr(z_mj >= u_m - ub_u * (1 - x_vars[j]))

    for m in range(num_segments):
        expr = gp.LinExpr()
        for j in seg_consideration[m]:
            w = seg_weight_dict[m][j]
            expr += w * z_vars[(m, j)]
        expr += seg_no_purchase[m] * u_vars[m]
        sub.addConstr(expr == 1)

    sub.addConstr(sum(x_vars) >= 1)
    sub.update()

    # ---- Helpers ----
    best_obj = -1e18
    best_solution = None

    def extract_and_log():
        nonlocal best_obj, best_solution
        if master.SolCount == 0:
            return
        obj_val = master.ObjVal
        if obj_val > best_obj + 1e-8:
            best_obj = obj_val
            active = []
            for i, (os, c, A) in enumerate(columns):
                if t_vars[i].X > 1e-8:
                    active.append({
                        "offer_set": [product_ids[j] for j in os],
                        "time_allocated": float(t_vars[i].X)
                    })
            best_solution = {
                "objective_value": float(obj_val),
                "active_columns": active
            }
            if logger:
                logger.log_solution(obj_val, best_solution)

    def add_column_to_master(os_tuple):
        c, A = compute_column(list(os_tuple))
        col = gp.Column()
        for l in range(num_legs):
            if abs(A[l]) > 1e-12:
                col.addTerms(A[l], cap_constraints[l])
        col.addTerms(1.0, time_constraint)
        new_var = master.addVar(lb=0, obj=c, column=col, name=f't_{len(columns)}')
        columns.append((os_tuple, c, A))
        t_vars.append(new_var)
        column_set.add(os_tuple)
        master.update()
        return c, A

    # ---- Column generation loop ----
    for iteration in range(10000):
        elapsed = time.time() - start_time
        if elapsed > args.time_limit - 2:
            break

        master.optimize()
        extract_and_log()

        if master.status != GRB.OPTIMAL:
            break

        # Duals
        pi = [cap_constraints[l].Pi for l in range(num_legs)]
        pi_0 = time_constraint.Pi

        # Modified revenues
        r = [fares[j] - sum(pi[l] for l in product_legs[j]) for j in range(num_products)]

        # --- Heuristic column generation (fast) ---
        heuristic_added = False

        # Heuristic 1: all products with positive r
        pos_r = [j for j in range(num_products) if r[j] > 0]
        if pos_r:
            pos_r_os = tuple(sorted(pos_r))
            if pos_r_os not in column_set:
                c, A = compute_column(list(pos_r_os))
                rc = c - sum(A[l] * pi[l] for l in range(num_legs)) - pi_0
                if rc > 1e-4:
                    add_column_to_master(pos_r_os)
                    heuristic_added = True

        # Heuristic 2: per-segment positive-r intersection
        if not heuristic_added:
            for m in range(num_segments):
                seg_pos = tuple(sorted(j for j in seg_consideration[m] if r[j] > 0))
                if seg_pos and seg_pos not in column_set:
                    c, A = compute_column(list(seg_pos))
                    rc = c - sum(A[l] * pi[l] for l in range(num_legs)) - pi_0
                    if rc > 1e-4:
                        add_column_to_master(seg_pos)
                        heuristic_added = True
                        break

        # Heuristic 3: top-k by r
        if not heuristic_added:
            sorted_products = sorted(range(num_products), key=lambda j: r[j], reverse=True)
            for k in range(1, min(num_products + 1, 6)):
                topk_os = tuple(sorted_products[:k])
                if topk_os not in column_set:
                    c, A = compute_column(list(topk_os))
                    rc = c - sum(A[l] * pi[l] for l in range(num_legs)) - pi_0
                    if rc > 1e-4:
                        add_column_to_master(topk_os)
                        heuristic_added = True
                        break

        if heuristic_added:
            continue  # Re-solve master with new columns

        # --- Exact subproblem (MILP) ---
        remaining = args.time_limit - (time.time() - start_time) - 1
        if remaining < 1:
            break

        sub.setParam('TimeLimit', min(remaining, 30))

        obj = gp.LinExpr()
        for z_var, m, j, w in z_obj_info:
            coeff = seg_lambda[m] * w * r[j]
            obj += coeff * z_var
        sub.setObjective(obj, GRB.MAXIMIZE)

        sub.optimize()

        if sub.SolCount == 0:
            break

        sub_obj = sub.ObjVal

        # Reduced cost = sub_obj - pi_0
        if sub_obj <= pi_0 + 1e-4:
            break  # No improving column found

        new_os = tuple(sorted(j for j in range(num_products) if x_vars[j].X > 0.5))
        if not new_os or new_os in column_set:
            break

        add_column_to_master(new_os)

    # ---- Output ----
    if best_solution is None:
        # Fallback: solve master one more time
        master.optimize()
        if master.SolCount > 0:
            active = []
            for i, (os, c, A) in enumerate(columns):
                if t_vars[i].X > 1e-8:
                    active.append({
                        "offer_set": [product_ids[j] for j in os],
                        "time_allocated": float(t_vars[i].X)
                    })
            best_solution = {
                "objective_value": float(master.ObjVal),
                "active_columns": active
            }
        else:
            best_solution = {"objective_value": 0.0, "active_columns": []}

    with open(args.solution_path, 'w') as f:
        json.dump(best_solution, f, indent=2)


if __name__ == '__main__':
    main()