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

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    num_sites = instance['num_sites']
    num_customers = instance['num_customers']
    T = instance['parameters']['T']
    delta = instance['parameters']['delta']
    epsilon = instance['parameters']['epsilon']
    L = instance['parameters']['L']
    c = instance['parameters']['c']
    q = instance['parameters']['q']
    weekly_attraction_cycle = instance['parameters']['weekly_attraction_cycle']
    avg_businesses_customers = instance['parameters']['avg_businesses_customers']
    avg_businesses_sites = instance['parameters']['avg_businesses_sites']

    sites = instance['sites']
    customers = instance['customers']
    site_to_site_distances = instance['site_to_site_distances']
    customer_to_site_distances = instance['customer_to_site_distances']

    # Dummy site 0
    S = list(range(0, num_sites + 1))  # 0 is dummy
    real_sites = list(range(1, num_sites + 1))
    periods = list(range(1, T + 1))
    customers_list = list(range(1, num_customers + 1))

    # Precompute distances
    dist = {}
    for s1 in S:
        for s2 in real_sites:
            if s1 == 0:
                dist[s1, s2] = 0.0
            else:
                key = f"{s1}_{s2}"
                dist[s1, s2] = site_to_site_distances.get(key, site_to_site_distances.get(f"{s2}_{s1}", 0.0))

    # Customer to site distances
    d_cs = {}
    for cc in customers_list:
        for ss in real_sites:
            key = f"{cc}_{ss}"
            d_cs[cc, ss] = customer_to_site_distances[key]

    # Site data
    capacity = {s['id']: s['capacity'] for s in sites}
    revenue = {s['id']: s['revenue_per_customer'] for s in sites}
    fixed_cost = {s['id']: s['fixed_cost'] for s in sites}
    max_events_site = {s['id']: s['max_events'] for s in sites}

    pop = {c['id']: c['population'] for c in customers}

    # Attraction values
    attr_site = {}
    for s in sites:
        sid = s['id']
        num_bus = s['num_businesses']
        attr_site[sid] = num_bus / avg_businesses_sites

    self_attr = {}
    for cc in customers:
        cid = cc['id']
        num_bus = cc['num_businesses']
        self_attr[cid] = num_bus / avg_businesses_customers

    # Weekly attraction cycle (1-indexed for periods)
    wac = [0.0] + weekly_attraction_cycle  # index 1 = Monday, etc.

    # Precompute attraction A_csp for all c,s,p (without x)
    A_base = {}
    for cid in customers_list:
        for sid in real_sites:
            dist_val = d_cs[cid, sid]
            if dist_val < 1e-6:
                dist_val = 1e-6
            decay = dist_val ** q
            base = attr_site[sid] / decay
            for p in periods:
                week_day = ((p - 1) % 7) + 1
                A_base[cid, sid, p] = base * wac[week_day]

    # Start model
    model = gp.Model("TouringEntertainment")
    model.Params.TimeLimit = args.time_limit - 5  # reserve some time
    model.Params.Threads = 1
    model.Params.MIPGap = 1e-4
    model.Params.LogToConsole = 0

    # Variables
    x = model.addVars(real_sites, periods, vtype=GRB.BINARY, name="x")  # x[s,p] = 1 if visit s in p
    y = model.addVars(real_sites, periods, vtype=GRB.BINARY, name="y")  # y[s,p] = 1 if event at s in p
    z = model.addVars(S, real_sites, periods, vtype=GRB.BINARY, name="z")  # z[from,to,p] move from->to in period p (p from 1 to T)

    # For linearization of attendance
    attend = model.addVars(real_sites, periods, lb=0, ub=GRB.INFINITY, name="attend")

    # Routing constraints
    # Period 1: from dummy
    model.addConstrs((z[0, s, 1] == x[s, 1] for s in real_sites), name="start")

    # Flow conservation
    for p in range(1, T):
        for s in real_sites:
            model.addConstr(
                gp.quicksum(z[s, t, p] for t in real_sites) == x[s, p],
                name=f"flow_out_{s}_{p}"
            )
            model.addConstr(
                gp.quicksum(z[t, s, p+1] for t in real_sites) == x[s, p+1],
                name=f"flow_in_{s}_{p+1}"
            )

    # At most one move per period
    model.addConstrs((gp.quicksum(z[i, j, p] for i in S for j in real_sites) <= 1 for p in periods), name="one_move")

    # Distance limit
    model.addConstrs((
        gp.quicksum(z[i, j, p] * dist[i, j] for i in S for j in real_sites) <= L
        for p in periods
    ), name="dist_limit")

    # Event only if visit
    model.addConstrs((y[s, p] <= x[s, p] for s in real_sites for p in periods), name="event_only_if_visit")

    # Max events per site
    model.addConstrs((
        gp.quicksum(y[s, p] for p in periods) <= max_events_site[s]
        for s in real_sites
    ), name="max_events_site")

    # Rest constraint: no more than delta events in any window of delta+2 periods
    window_size = delta + 2
    for t in range(1, T - window_size + 2):
        model.addConstr(
            gp.quicksum(y[s, p] for s in real_sites for p in range(t, t + window_size)) <= delta,
            name=f"rest_{t}"
        )

    # Linearization of attendance with big-M
    M = 1e7  # sufficiently large
    for s in real_sites:
        for p in periods:
            model.addConstr(attend[s, p] <= capacity[s], name=f"cap_{s}_{p}")
            model.addConstr(attend[s, p] <= M * y[s, p], name=f"attend_if_event_{s}_{p}")

    # Gravity model linearization is complex. We use a MIP formulation with auxiliary variables for denominators.
    # This is a known hard bilinear fractional model. We introduce variables for the denominator contribution.

    # For each customer c, period p, the denom is self + sum over all events in [p-eps, p+eps]
    # We create aux var D_cp = self_attr[c] + sum relevant attractions * y
    # But since we have product with y in num, it's bilinear.

    # To handle this, we use McCormick envelope or introduce new variables for each possible term.
    # Given single core and time, we will use a big-M based linearization for the proportion.

    # Let prop_csp = proportion from c to s in p
    # attend[s,p] = sum_c pop[c] * prop_csp   (approx, ignoring the min(1,..) which is loose)

    prop = model.addVars(customers_list, real_sites, periods, lb=0, ub=1, name="prop")

    # For each c,p: sum_s prop[c,s,p] <= 1
    for c in customers_list:
        for p in periods:
            model.addConstr(
                gp.quicksum(prop[c, s, p] for s in real_sites) <= 1.0,
                name=f"prop_sum_{c}_{p}"
            )

    # Relationship between prop, y and denom.
    # prop_csp * (self_c_p + sum_{k in [p-e..p+e], j} A_cjk * y_jk ) = A_csp * y_sp
    # This is bilinear. To linearize, we use additional variables and big-M constraints.

    # Let total_attr_cp = sum over relevant (j,k) of A_base[c,j,k] * y[j,k]
    total_attr = model.addVars(customers_list, periods, lb=0, name="total_attr")

    for c in customers_list:
        for p in periods:
            relevant = []
            for k in range(max(1, p - epsilon), min(T, p + epsilon) + 1):
                for j in real_sites:
                    relevant.append((j, k))
            model.addConstr(
                total_attr[c, p] == gp.quicksum(A_base[c, j, k] * y[j, k] for j, k in relevant),
                name=f"total_attr_{c}_{p}"
            )

    # denom_cp = self_attr[c] + total_attr[c,p]
    # Then prop_csp * denom_cp = A_base[c,s,p] * y[s,p]

    # Introduce aux var left_csp = prop_csp * denom_cp
    left = model.addVars(customers_list, real_sites, periods, lb=0, name="left")

    # For each c,s,p: left[c,s,p] == A_base[c,s,p] * y[s,p]   but only if we set it as equality with bounds
    # Since denom >= self > 0, we can bound it.

    for c in customers_list:
        for s in real_sites:
            for p in periods:
                Abase = A_base[c, s, p]
                model.addConstr(left[c, s, p] == Abase * y[s, p], name=f"define_left_{c}_{s}_{p}")

    # Now link left = prop * denom
    # We use big-M reformulation for product of two continuous vars (prop and denom).
    # But denom is also variable.

    denom = model.addVars(customers_list, periods, lb=0, name="denom")
    for c in customers_list:
        for p in periods:
            model.addConstr(denom[c, p] == self_attr[c] + total_attr[c, p], name=f"denom_def_{c}_{p}")

    # Now for product: left[c,s,p] = prop[c,s,p] * denom[c,p]
    # Since both are continuous, we introduce McCormick inequalities with bounds.

    # Get bounds for denom
    # Self is fixed, max total_attr is sum all possible A
    max_A_sum = {}
    for c in customers_list:
        max_possible = 0.0
        for s in real_sites:
            for p in periods:
                max_possible += A_base[c, s, p]
        max_A_sum[c] = max_possible

    for c in customers_list:
        for p in periods:
            d_min = self_attr[c]
            d_max = self_attr[c] + max_A_sum[c]
            for s in real_sites:
                # left is bounded [0, A_base[c,s,p]]
                l_min = 0.0
                l_max = A_base[c, s, p]
                p_min = 0.0
                p_max = 1.0

                # McCormick envelope
                model.addConstr(left[c, s, p] >= d_min * prop[c, s, p] + p_min * denom[c, p] - d_min * p_min)
                model.addConstr(left[c, s, p] >= d_max * prop[c, s, p] + p_max * denom[c, p] - d_max * p_max)
                model.addConstr(left[c, s, p] <= d_min * prop[c, s, p] + p_max * denom[c, p] - d_min * p_max)
                model.addConstr(left[c, s, p] <= d_max * prop[c, s, p] + p_min * denom[c, p] - d_max * p_min)

    # Link to attendance
    for s in real_sites:
        for p in periods:
            model.addConstr(
                attend[s, p] == gp.quicksum(pop[c] * prop[c, s, p] for c in customers_list),
                name=f"attend_def_{s}_{p}"
            )

    # Objective
    revenue_term = gp.quicksum(revenue[s] * attend[s, p] for s in real_sites for p in periods)
    fixed_term = gp.quicksum(fixed_cost[s] * y[s, p] for s in real_sites for p in periods)
    travel_term = c * gp.quicksum(z[i, j, p] * dist[i, j] for i in S for j in real_sites for p in periods)

    model.setObjective(revenue_term - fixed_term - travel_term, GRB.MAXIMIZE)

    # Callback for logging
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None
    best_obj = -float('inf')
    start_time = time.time()

    def callback(model, where):
        nonlocal best_obj
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if obj > best_obj + 1e-4:
                best_obj = obj
                if logger:
                    logger.log(obj)
        elif where == GRB.Callback.MIP:
            runtime = model.cbGet(GRB.Callback.RUNTIME)
            if runtime > args.time_limit - 2:
                model.terminate()

    # Optimize
    model.optimize(callback)

    # Extract solution
    if model.SolCount >= 1:
        obj_val = model.ObjVal
        tour_sol = {}
        events_sol = {}
        event_sites_sol = {}
        attendance_sol = {}

        x_sol = model.getAttr('X', x)
        y_sol = model.getAttr('X', y)
        attend_sol = model.getAttr('X', attend)

        for p in periods:
            for s in real_sites:
                if abs(x_sol[s, p] - 1.0) < 0.5:
                    tour_sol[str(p)] = s
                    break
            events_sol[str(p)] = int(round(y_sol[s, p])) if 's' in locals() else 0
            if events_sol[str(p)] == 1:
                event_sites_sol[str(p)] = tour_sol[str(p)]
            else:
                event_sites_sol[str(p)] = None

        for s in real_sites:
            for p_idx, p in enumerate(periods):
                key = f"{s}_{p}"
                attendance_sol[key] = float(attend_sol[s, p])

        # Fill any missing
        for p in periods:
            pstr = str(p)
            if pstr not in events_sol:
                events_sol[pstr] = 0
            if pstr not in event_sites_sol:
                event_sites_sol[pstr] = None
            if pstr not in tour_sol:
                tour_sol[pstr] = 0  # should not happen

        solution = {
            "objective_value": float(obj_val),
            "tour": tour_sol,
            "events": events_sol,
            "event_sites": event_sites_sol,
            "attendance": attendance_sol
        }

        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
    else:
        # No solution found, write empty feasible if possible (but should not happen)
        solution = {
            "objective_value": 0.0,
            "tour": {str(p): 1 for p in periods},
            "events": {str(p): 0 for p in periods},
            "event_sites": {str(p): None for p in periods},
            "attendance": {}
        }
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()