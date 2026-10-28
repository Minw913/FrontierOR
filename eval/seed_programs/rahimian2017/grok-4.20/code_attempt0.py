import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    num_days = data['num_days']
    num_nurses = data['num_nurses']
    num_shift_types = data['num_shift_types']
    num_weekends = data['num_weekends']

    shift_length = {}
    for st in data['shift_types']:
        shift_length[st['id']] = st['length_minutes']

    forbidden = {}
    for fsr in data['forbidden_shift_rotations']:
        s1 = fsr['shift_type']
        s2 = fsr['cannot_be_followed_by']
        if s1 not in forbidden:
            forbidden[s1] = set()
        forbidden[s1].add(s2)

    nurses = data['nurses']
    max_shifts = {}
    min_total_min = {}
    max_total_min = {}
    min_consec = {}
    max_consec = {}
    min_days_off = {}
    max_weekends_n = {}
    day_off_req = {}

    for n_info in nurses:
        nid = n_info['id']
        max_shifts[nid] = n_info['max_shifts_per_type']
        min_total_min[nid] = n_info['min_total_minutes']
        max_total_min[nid] = n_info['max_total_minutes']
        min_consec[nid] = n_info['min_consecutive_shifts']
        max_consec[nid] = n_info['max_consecutive_shifts']
        min_days_off[nid] = n_info['min_consecutive_days_off']
        max_weekends_n[nid] = n_info['max_weekends']
        day_off_req[nid] = set(n_info['day_off_requests'])

    shift_on_pen = {}
    for sor in data['shift_on_requests']:
        key = (sor['nurse_id'], sor['day'], sor['shift_type'])
        shift_on_pen[key] = sor['penalty']

    shift_off_pen = {}
    for sof in data['shift_off_requests']:
        key = (sof['nurse_id'], sof['day'], sof['shift_type'])
        shift_off_pen[key] = sof['penalty']

    preferred = {}
    under_w = {}
    over_w = {}
    for cov in data['coverage_requirements']:
        d = cov['day']
        s = cov['shift_type']
        key = (d, s)
        preferred[key] = cov['preferred']
        under_w[key] = cov['under_weight']
        over_w[key] = cov['over_weight']

    # Weekend mapping: day -> weekend_id (0-based)
    # Assuming days 0..num_days-1, weeks start Monday, weekend = Sat(5),Sun(6) if 0=Mon
    weekend_of = {}
    for w in range(num_weekends):
        sat = w * 7 + 5
        sun = w * 7 + 6
        if sat < num_days:
            weekend_of[sat] = w
            if sun < num_days:
                weekend_of[sun] = w

    model = gp.Model("NurseRoster")
    model.setParam('TimeLimit', args.time_limit - 5)  # safety margin
    model.setParam('Threads', 1)
    model.setParam('MIPGap', 0.0)
    model.setParam('LogToConsole', 0)

    # x[n,d,s] = 1 if nurse n works shift s on day d (s=0..num_shift_types-1, 0 means off)
    # Use s=0 for OFF
    S = list(range(num_shift_types))  # 0 is dummy OFF, real shifts 1.. but we map
    # Remap real shifts to 1..K, 0=off
    real_shifts = list(range(1, num_shift_types + 1))
    shift_id_map = {}  # original id -> model id 1..
    orig_to_model = {}
    model_to_orig = {}
    for idx, st in enumerate(data['shift_types'], 1):
        orig_id = st['id']
        orig_to_model[orig_id] = idx
        model_to_orig[idx] = orig_id
        shift_length[idx] = st['length_minutes']  # update

    # forbidden in model ids
    forbidden_model = {}
    for s1_orig, s2s in forbidden.items():
        s1m = orig_to_model[s1_orig]
        forbidden_model[s1m] = {orig_to_model[s2] for s2 in s2s}

    x = {}
    for n_info in nurses:
        n = n_info['id']
        for d in range(num_days):
            for s in range(num_shift_types + 1):  # 0=off, 1.. = shifts
                x[n, d, s] = model.addVar(vtype=GRB.BINARY, name=f"x_{n}_{d}_{s}")

    # Exactly one choice per nurse per day
    for n_info in nurses:
        n = n_info['id']
        for d in range(num_days):
            model.addConstr(gp.quicksum(x[n, d, s] for s in range(num_shift_types + 1)) == 1)

    # Day off requests: must be off (s=0)
    for n_info in nurses:
        n = n_info['id']
        for d in day_off_req[n]:
            if 0 <= d < num_days:
                model.addConstr(x[n, d, 0] == 1)

    # Forbidden successions (real shifts only)
    for n_info in nurses:
        n = n_info['id']
        for d in range(num_days - 1):
            for s1 in range(1, num_shift_types + 1):
                for s2 in forbidden_model.get(s1, []):
                    model.addConstr(x[n, d, s1] + x[n, d + 1, s2] <= 1)

    # Last of previous and first of next are off -> no constraint on first day from previous (implicit)
    # First day after assumed off, last before next assumed off: already handled by not having forbidden from 0, and no var for it.

    # Max shifts per type per nurse
    for n_info in nurses:
        n = n_info['id']
        for orig_sid, maxcnt in max_shifts[n].items():
            if orig_sid in orig_to_model:
                sm = orig_to_model[orig_sid]
                model.addConstr(gp.quicksum(x[n, d, sm] for d in range(num_days)) <= maxcnt)

    # Total minutes
    for n_info in nurses:
        n = n_info['id']
        total_min = gp.quicksum(shift_length[s] * x[n, d, s] for d in range(num_days) for s in range(1, num_shift_types + 1))
        model.addConstr(total_min >= min_total_min[n])
        model.addConstr(total_min <= max_total_min[n])

    # Consecutive shifts: max_consec
    M = max_consec[n_info['id']] + 2  # safe
    for n_info in nurses:
        n = n_info['id']
        mc = max_consec[n]
        for d in range(num_days - mc):
            model.addConstr(gp.quicksum(x[n, d + k, s] for k in range(mc + 1) for s in range(1, num_shift_types + 1)) <= mc)

    # Boundary for max consec at start and end
    # Day before is off, day after is off -> already covered by above if we add dummy off
    # For start: consider from d=-1 off
    for n_info in nurses:
        n = n_info['id']
        mc = max_consec[n]
        # first mc days
        model.addConstr(gp.quicksum(x[n, k, s] for k in range(min(mc, num_days)) for s in range(1, num_shift_types + 1)) <= mc)
        # last mc days
        if num_days > mc:
            model.addConstr(gp.quicksum(x[n, num_days - mc + k, s] for k in range(mc) for s in range(1, num_shift_types + 1)) <= mc)

    # Min consecutive shifts - use the "no short isolated work stretch" formulation
    for n_info in nurses:
        n = n_info['id']
        mcs = min_consec[n]
        if mcs <= 1:
            continue  # always satisfied
        # Inside
        for start in range(1, num_days - mcs):
            # if off at start-1 and off at start+mcs, then cannot have all working in between
            off_before = x[n, start - 1, 0]
            off_after = x[n, start + mcs, 0]
            sum_work = gp.quicksum(x[n, start + k, s] for k in range(mcs) for s in range(1, num_shift_types + 1))
            model.addConstr(sum_work <= mcs - 1 + (1 - off_before) + (1 - off_after))

        # Start of horizon: day -1 is "working infinite" wait - the description says:
        # "it is assumed that the nurse has been working an infinite stretch of consecutive shifts just before the first day and just after the last day"
        # For min_consec_shifts this affects off blocks? Let's read carefully.

# The description:
# "A nurse must never work an isolated block of consecutive shifts shorter than the nurse's minimum consecutive shifts requirement; 
# formally, if a nurse is off on some day and also off on a day that is only a short gap later [...], then the nurse cannot be working on all of the days in between without also working on at least one of those boundary days. 
# The same logic applies symmetrically at the boundaries of the planning horizon, where it is assumed that the nurse has been working an infinite stretch of consecutive shifts just before the first day and just after the last day."

        # So for min consecutive WORK: the boundary assumption is that before day 0 and after last day the nurse has been WORKING infinitely.
        # That means:
        # - If starts with off on day 0, then the "before" is working, so if then has a work stretch, it is bounded by working before.
        # But to forbid short work stretches at start: if the first work stretch is short and then off, it would be forbidden because before was "working" but we need the logic.

        # Standard way:
        # For a potential short work block from day a to b-1 (length l = b-a < min), it is forbidden if off on a-1 and off on b.

        # At start: a=0, a-1 is "working", so off(a-1)=0 always. Thus cannot have off on b and short work 0..b-1. i.e. if you work first l days with l<min and then off on l, this is forbidden.
        # Which is correct for "must not have short work block at the beginning".

        # Similarly at end.

        # So let's implement general:
        for n_info in nurses:
            n = n_info['id']
            minc = min_consec[n]
            if minc <= 1:
                continue
            # General windows
            for len_w in range(1, minc):  # short lengths
                for a in range(num_days - len_w + 1):
                    b = a + len_w
                    off_before = 1 if a == 0 else x[n, a - 1, 0]  # before 0 is NOT off (was working)
                    off_after = 1 if b == num_days else x[n, b, 0]  # after last is NOT off (working)
                    work_sum = gp.quicksum(x[n, a + k, s] for k in range(len_w) for s in range(1, num_shift_types + 1))
                    # if off_before and off_after then work_sum <= len_w - 1
                    if isinstance(off_before, int):  # 1 or 0
                        if off_before == 1:
                            if isinstance(off_after, int):
                                if off_after == 1:
                                    model.addConstr(work_sum <= len_w - 1)
                                # else no constraint
                            else:
                                model.addConstr(work_sum <= len_w - 1 + (1 - off_after))
                        else:
                            # no constraint
                            pass
                    else:
                        rhs = len_w - 1 + (1 - off_before)
                        if not isinstance(off_after, int):
                            rhs += (1 - off_after)
                        elif off_after == 0:
                            rhs += 1
                        model.addConstr(work_sum <= rhs)

    # Min consecutive days off
    # "A nurse must never have an isolated block of consecutive days off shorter than the nurse's minimum consecutive days off"
    # "at the horizon boundaries it is assumed the nurse has been off for an infinite stretch"
    # So boundary: before day 0 and after last = off = infinite off.
    # Thus short off block in middle forbidden if bounded by work on both sides.
    # At start: if starts with short off then work, since before was off, it would be continuation, not isolated.
    # The assumption matches symmetric logic but for off.

    for n_info in nurses:
        n = n_info['id']
        minoff = min_days_off[n]
        if minoff <= 1:
            continue
        for len_off in range(1, minoff):
            for a in range(num_days - len_off + 1):
                b = a + len_off
                # off block a .. b-1, to be isolated short: work before and work after
                work_before = 0 if a == 0 else gp.quicksum(x[n, a-1, s] for s in range(1, num_shift_types+1))  # before 0 is off -> work_before=0
                work_after = 0 if b == num_days else gp.quicksum(x[n, b, s] for s in range(1, num_shift_types+1))   # after end is off -> work_after=0
                off_sum = gp.quicksum(x[n, a+k, 0] for k in range(len_off))
                # if work_before==1 and work_after==1 then off_sum <= len_off-1
                rhs = len_off - 1
                if isinstance(work_before, int) and work_before == 0:
                    continue  # cannot be isolated
                if isinstance(work_after, int) and work_after == 0:
                    continue
                if not isinstance(work_before, int):
                    rhs += (1 - work_before)
                if not isinstance(work_after, int):
                    rhs += (1 - work_after)
                model.addConstr(off_sum <= rhs)

    # Max weekends
    for n_info in nurses:
        n = n_info['id']
        mw = max_weekends_n[n]
        worked_weekend = {}
        for w in range(num_weekends):
            worked = model.addVar(vtype=GRB.BINARY, name=f"wknd_{n}_{w}")
            days_in_w = []
            for d, wid in weekend_of.items():
                if wid == w:
                    days_in_w.append(d)
            if not days_in_w:
                continue
            model.addConstr(worked >= gp.quicksum(x[n, d, s] for d in days_in_w for s in range(1, num_shift_types + 1)) / len(days_in_w))
            # since at most one shift, this works as upper bound too but we use <= mw on sum
            worked_weekend[w] = worked
        model.addConstr(gp.quicksum(worked_weekend[w] for w in range(num_weekends) if w in worked_weekend) <= mw)

    # Objective components
    request_penalty = gp.LinExpr()
    for n_info in nurses:
        n = n_info['id']
        for d in range(num_days):
            for s_real, orig_s in model_to_orig.items():
                # shift on request: if not assigned but requested, penalty
                key_on = (n, d, orig_s)
                if key_on in shift_on_pen:
                    request_penalty.addTerms(shift_on_pen[key_on], (1 - x[n, d, s_real]))
                # shift off request: if assigned, penalty (s_real > 0)
                key_off = (n, d, orig_s)
                if key_off in shift_off_pen:
                    request_penalty.addTerms(shift_off_pen[key_off], x[n, d, s_real])

    under_vars = {}
    over_vars = {}
    coverage_penalty = gp.LinExpr()
    for cov in data['coverage_requirements']:
        d = cov['day']
        orig_s = cov['shift_type']
        if orig_s not in orig_to_model:
            continue
        s = orig_to_model[orig_s]
        pref = cov['preferred']
        uw = cov['under_weight']
        ow = cov['over_weight']
        assigned = gp.quicksum(x[n_info['id'], d, s] for n_info in nurses)
        under = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"under_{d}_{s}")
        over = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"over_{d}_{s}")
        model.addConstr(assigned + under - over == pref)
        coverage_penalty.add(uw * under + ow * over)
        under_vars[(d, orig_s)] = under
        over_vars[(d, orig_s)] = over

    model.setObjective(request_penalty + coverage_penalty, GRB.MINIMIZE)

    # Callback for logging incumbents
    best_obj = None
    start_time = time.time()

    def callback(model, where):
        nonlocal best_obj
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if best_obj is None or obj < best_obj:
                best_obj = obj
                if logger:
                    logger.log(float(obj))

    # Optimize
    model.optimize(callback)

    # Extract solution
    if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED):
        schedule = {}
        for n_info in nurses:
            nid = n_info['id']
            schedule[str(nid)] = {}
            for d in range(num_days):
                assigned = False
                for s in range(1, num_shift_types + 1):
                    if abs(x[nid, d, s].X) > 0.5:
                        schedule[str(nid)][str(d)] = model_to_orig[s]
                        assigned = True
                        break
                if not assigned:
                    schedule[str(nid)][str(d)] = None

        coverage_under_out = {}
        coverage_over_out = {}
        for key, var in under_vars.items():
            d, s_orig = key
            coverage_under_out[f"{d}_{s_orig}"] = max(0, int(round(var.X)))
        for key, var in over_vars.items():
            d, s_orig = key
            coverage_over_out[f"{d}_{s_orig}"] = max(0, int(round(var.X)))

        objective = model.objVal if model.objVal is not None else best_obj if best_obj is not None else 0.0

        solution = {
            "objective_value": float(objective),
            "solution": {
                "schedule": schedule,
                "coverage_under": coverage_under_out,
                "coverage_over": coverage_over_out
            }
        }

        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
    else:
        # Infeasible or other - write empty-ish
        solution = {
            "objective_value": 1e9,
            "solution": {
                "schedule": {},
                "coverage_under": {},
                "coverage_over": {}
            }
        }
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()