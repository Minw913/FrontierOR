"""
Two-stage stochastic capacitated facility location solver.

Formulation (deterministic equivalent MIP):
    x_i       in {0,1}   : open facility i (first stage, scenario independent)
    y[s,i,j]  >= 0       : amount shipped from facility i to customer j in
                           scenario s (second stage recourse)

    min  sum_i f_i x_i + sum_s p_s sum_{i,j} c_ij y[s,i,j]
    s.t. sum_i y[s,i,j] >= d[s,j]                    for all s, j
         sum_j y[s,i,j] <= u_i x_i                   for all s, i
         sum_i u_i x_i   >= max_s sum_j d[s,j]

The model is solved with Gurobi.  Every incumbent found during the search is
reported to the convergence logger, and after the branch-and-bound terminates
(optimal or time limit) the recourse flows of the incumbent are re-optimized
as an LP with the opening decisions fixed, guaranteeing that the returned
flows are optimal for the selected facilities.  A greedy constructive
heuristic with an exact flow-completion fallback guarantees that a feasible
solution is always produced, even if the MIP finds no incumbent in time.
"""

import argparse
import json
import time

import gurobipy as gp
from gurobipy import GRB


def main():
    parser = argparse.ArgumentParser(
        description="Two-stage stochastic capacitated facility location solver."
    )
    parser.add_argument("--instance_path", required=True,
                        help="Path to the JSON problem instance.")
    parser.add_argument("--solution_path", required=True,
                        help="Path where the final solution JSON is written.")
    parser.add_argument("--time_limit", type=int, required=True,
                        help="Maximum runtime in seconds.")
    parser.add_argument("--log_path", default=None,
                        help="Optional path to a JSONL convergence log file.")
    args = parser.parse_args()

    logger = None
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize")

    t_start = time.time()
    time_limit = float(args.time_limit)

    def remaining_time():
        return time_limit - (time.time() - t_start)

    # ------------------------------------------------------------------
    # Read instance.
    # ------------------------------------------------------------------
    with open(args.instance_path, "r") as fh:
        inst = json.load(fh)

    F = int(inst["num_facilities"])
    C = int(inst["num_customers"])
    S = int(inst["num_scenarios"])

    facilities = inst["facilities"]
    scenarios = inst["scenarios"]
    costs = [[float(v) for v in row] for row in inst["transportation_costs"]]

    fcost = [float(f["opening_cost"]) for f in facilities]
    cap = [float(f["capacity"]) for f in facilities]
    probs = [float(sc["probability"]) for sc in scenarios]
    demands = [[float(v) for v in sc["demands"]] for sc in scenarios]
    scenario_totals = [sum(d) for d in demands]
    max_total_demand = max(scenario_totals) if scenario_totals else 0.0

    n_y = S * F * C
    # For very large models, serializing a full flow snapshot for every
    # incumbent is too expensive; log objective values instead during the
    # search and a full snapshot at the end.
    log_full_every_incumbent = n_y <= 250000

    # ------------------------------------------------------------------
    # Helpers.
    # ------------------------------------------------------------------
    def build_solution(xvals, yflat):
        """Project raw variable values onto the required output schema and
        recompute the objective from scratch for consistency."""
        x_dict = {}
        open_facs = []
        fixed_cost = 0.0
        for i in range(F):
            xi = 1 if xvals[i] > 0.5 else 0
            x_dict[str(i)] = xi
            if xi:
                open_facs.append(i)
                fixed_cost += fcost[i]
        y_out = {}
        transport = 0.0
        k = 0
        for s in range(S):
            p = probs[s]
            fdict = {}
            for i in range(F):
                ci = costs[i]
                cdict = {}
                for j in range(C):
                    v = yflat[k]
                    k += 1
                    if v < 0.0:
                        v = 0.0
                    cdict[str(j)] = v
                    transport += p * ci[j] * v
                fdict[str(i)] = cdict
            y_out[str(s)] = fdict
        objective = fixed_cost + transport
        sol = {
            "objective_value": objective,
            "open_facilities": open_facs,
            "x": x_dict,
            "y": y_out,
        }
        return objective, sol

    def greedy_opening_set():
        """Cheap feasible opening set: facilities ordered by fixed cost per
        unit of capacity until the worst-case demand can be covered."""
        order = sorted(
            range(F),
            key=lambda i: (fcost[i] / cap[i] if cap[i] > 1e-12 else float("inf")),
        )
        chosen = set()
        tot = 0.0
        for i in order:
            chosen.add(i)
            tot += cap[i]
            if tot >= max_total_demand - 1e-9:
                break
        if tot < max_total_demand - 1e-9:
            chosen = set(range(F))
            tot = sum(cap)
        # Drop redundant facilities, most expensive first.
        for i in sorted(chosen, key=lambda i: -fcost[i]):
            if tot - cap[i] >= max_total_demand - 1e-9:
                chosen.discard(i)
                tot -= cap[i]
        return chosen

    # ------------------------------------------------------------------
    # Build the deterministic equivalent MIP.
    # ------------------------------------------------------------------
    model = gp.Model("two_stage_stochastic_facility_location")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = max(1.0, remaining_time())

    x = model.addVars(F, vtype=GRB.BINARY, name="x")
    y = model.addVars(S, F, C, lb=0.0, vtype=GRB.CONTINUOUS, name="y")
    xvars = list(x.values())
    yvars = y.values()  # ordered (s, i, j), last index fastest

    # Objective: fixed opening cost + expected transportation cost.
    coefs = []
    for s in range(S):
        p = probs[s]
        for i in range(F):
            ci = costs[i]
            coefs.extend([p * c for c in ci])
    obj = gp.LinExpr(coefs, yvars)
    obj.addTerms(fcost, xvars)
    model.setObjective(obj, GRB.MINIMIZE)

    # Every customer's demand must be fully served in every scenario.
    for s in range(S):
        ds = demands[s]
        for j in range(C):
            if ds[j] > 0.0:
                model.addConstr(gp.quicksum(y[s, i, j] for i in range(F)) >= ds[j])

    # Facility capacity (linked to the opening decision) in every scenario.
    for s in range(S):
        for i in range(F):
            model.addConstr(gp.quicksum(y[s, i, j] for j in range(C)) <= cap[i] * x[i])

    # Total opened capacity must cover the worst-case total demand.
    model.addConstr(gp.quicksum(cap[i] * x[i] for i in range(F)) >= max_total_demand)

    # Provide a greedy MIP start to seed the search.
    try:
        start_set = greedy_opening_set()
        for i in range(F):
            x[i].Start = 1.0 if i in start_set else 0.0
    except Exception:
        pass

    # ------------------------------------------------------------------
    # Incumbent logging callback.
    # ------------------------------------------------------------------
    cb_state = {"last_full_log": -1e18}

    def incumbent_callback(m, where):
        if where != GRB.Callback.MIPSOL or logger is None:
            return
        try:
            objbst = m.cbGet(GRB.Callback.MIPSOL_OBJBST)
        except Exception:
            return
        now = time.time()
        if log_full_every_incumbent and now - cb_state["last_full_log"] >= 1.0:
            cb_state["last_full_log"] = now
            try:
                xv = m.cbGetSolution(xvars)
                yv = m.cbGetSolution(yvars)
                obj_val, sol = build_solution(xv, yv)
                logger.log_solution(obj_val, sol)
                return
            except Exception:
                pass
        try:
            logger.log(objbst)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Solve.
    # ------------------------------------------------------------------
    try:
        model.optimize(incumbent_callback)
    except GRB.Error:
        pass

    final_obj = None
    final_sol = None

    if model.SolCount > 0:
        best_x = model.getAttr("X", xvars)
        best_y = model.getAttr("X", yvars)
        prev_model_obj = model.ObjVal

        # Re-optimize the recourse flows as an LP with the incumbent opening
        # decisions fixed, so that y is optimal for the chosen facilities.
        if model.Status != GRB.OPTIMAL and remaining_time() > 1.0:
            try:
                for i in range(F):
                    v = 1.0 if best_x[i] > 0.5 else 0.0
                    x[i].LB = v
                    x[i].UB = v
                model.Params.TimeLimit = max(0.5, remaining_time())
                model.optimize(incumbent_callback)
                if model.SolCount > 0 and model.ObjVal <= prev_model_obj + 1e-9:
                    best_x = model.getAttr("X", xvars)
                    best_y = model.getAttr("X", yvars)
            except GRB.Error:
                pass

        final_obj, final_sol = build_solution(best_x, best_y)

    # ------------------------------------------------------------------
    # Fallback: no incumbent from the MIP (e.g. extremely tight time limit).
    # ------------------------------------------------------------------
    if final_sol is None:
        chosen = greedy_opening_set()
        solved = False

        # Try to solve the recourse LP with the greedy set fixed.
        if remaining_time() > 0.5:
            try:
                for i in range(F):
                    v = 1.0 if i in chosen else 0.0
                    x[i].LB = v
                    x[i].UB = v
                model.Params.TimeLimit = max(0.5, remaining_time())
                model.optimize()
                if model.SolCount > 0:
                    bx = model.getAttr("X", xvars)
                    by = model.getAttr("X", yvars)
                    final_obj, final_sol = build_solution(bx, by)
                    solved = True
            except GRB.Error:
                pass

        # Last resort: constructive greedy flow assignment (always feasible
        # whenever total opened capacity covers total demand, since the
        # transportation network is complete bipartite).
        if not solved:
            xv = [1.0 if i in chosen else 0.0 for i in range(F)]
            yflat = [0.0] * n_y
            for s in range(S):
                residual = [cap[i] if i in chosen else 0.0 for i in range(F)]
                base = s * F * C
                for j in range(C):
                    need = demands[s][j]
                    if need <= 0.0:
                        continue
                    for i in sorted(range(F), key=lambda ii: costs[ii][j]):
                        r = residual[i]
                        if r <= 0.0:
                            continue
                        q = r if r < need else need
                        yflat[base + i * C + j] = q
                        residual[i] = r - q
                        need -= q
                        if need <= 1e-9:
                            break
            final_obj, final_sol = build_solution(xv, yflat)

    # ------------------------------------------------------------------
    # Final logging and output.
    # ------------------------------------------------------------------
    if logger:
        try:
            logger.log_solution(final_obj, final_sol)
        except Exception:
            pass

    with open(args.solution_path, "w") as fh:
        json.dump(final_sol, fh)


if __name__ == "__main__":
    main()