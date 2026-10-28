import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

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
        day_off_req[nid] = set(n_info.get('day_off_requests', []))

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

    weekend_of = {}
    for w in range(num_weekends):
        sat = w * 7 + 5
        sun = w * 7 + 6
        if sat < num_days:
            weekend_of[sat] = w
            if sun < num_days:
                weekend_of[sun] = w

    model = gp.Model("NurseRoster")
    model.Params.TimeLimit = args.time_limit - 5
    model.Params.Threads = 1
    model.Params.LogToConsole = 0
    model.Params.MIPGap = 1e-4

    # Mapping from original shift id to model index (1-based, 0=off)
    orig_to_model = {}
    model_to_orig = {}
    shift_len_model = {}
    for idx, st in enumerate(data['shift_types'], 1):
        orig_id = st['id']
        orig_to_model[orig_id] = idx
        model_to_orig[idx] = orig_id
        shift_len_model[idx] = st['length_minutes']

    # forbidden in model ids
    forbidden_model = {}
    for s1_orig, follows in forbidden.items():
        if s1_orig in orig_to_model:
            s1m = orig_to_model[s1_orig]
            forbidden_model[s1m] = {orig_to_model[s2] for s2 in follows if s2 in orig_to_model}

    x = {}
    for n_info in nurses:
        n = n_info['id']
        for d in range(num_days):
            for s in range(num_shift_types + 1):  # 0=off, 1..=shifts
                x[(n, d, s)] = model.addVar(vtype=GRB.BINARY, name=f"x_{n}_{d}_{s}")

    # Exactly one assignment per nurse-day
    for n_info in nurses:
        n = n_info['id']
        for d in range(num_days):
            model.addConstr(gp.quicksum(x[(n, d, s)] for s in range(num_shift_types + 1)) == 1, name=f"one_{n}_{d}")

    # Hard day off requests
    for n_info in nurses:
        n = n_info['id']
        for d in day_off_req.get(n, []):
            if 0 <= d < num_days:
                model.addConstr(x[(n, d, 0)] == 1, name=f"offreq_{n}_{d}")

    # Forbidden shift successions
    for n_info in nurses:
        n = n_info['id']
        for d in range(num_days - 1):
            for s1 in range(1, num_shift_types + 1):
                for s2 in forbidden_model.get(s1, []):
                    model.addConstr(x[(n, d, s1)] + x[(n, d + 1, s2)] <= 1, name=f"forbid_{n}_{d}_{s1}_{s2}")

    # Max shifts per type
    for n_info in nurses:
        n = n_info['id']
        for orig_sid_str, maxcnt in max_shifts.get(n, {}).items():
            orig_sid = int(orig_sid_str) if isinstance(orig_sid_str, str) else orig_sid_str
            if orig_sid in orig_to_model:
                sm = orig_to_model[orig_sid]
                model.addConstr(gp.quicksum(x[(n, d, sm)] for d in range(num_days)) <= maxcnt, name=f"maxshift_{n}_{sm}")

    # Total working minutes
    for n_info in nurses:
        n = n_info['id']
        total_min = gp.quicksum(shift_len_model.get(s, 0) * x[(n, d, s)] for d in range(num_days) for s in range(1, num_shift_types + 1))
        model.addConstr(total_min >= min_total_min[n], name=f"minmin_{n}")
        model.addConstr(total_min <= max_total_min[n], name=f"maxmin_{n}")

    # Max consecutive shifts
    for n_info in nurses:
        n = n_info['id']
        mc = max_consec[n]
        for d in range(num_days - mc):
            model.addConstr(gp.quicksum(x[(n, d + k, s)] for k in range(mc + 1) for s in range(1, num_shift_types + 1)) <= mc,
                            name=f"maxc_{n}_{d}")
        # start boundary
        model.addConstr(gp.quicksum(x[(n, k, s)] for k in range(min(mc, num_days)) for s in range(1, num_shift_types + 1)) <= mc,
                        name=f"maxcstart_{n}")
        # end boundary
        if num_days > mc:
            model.addConstr(gp.quicksum(x[(n, num_days - mc + k, s)] for k in range(mc) for s in range(1, num_shift_types + 1)) <= mc,
                            name=f"maxcend_{n}")

    # Min consecutive work shifts (no isolated short work blocks)
    for n_info in nurses:
        n = n_info['id']
        minc = min_consec[n]
        if minc <= 1:
            continue
        for length in range(1, minc):
            for a in range(num_days - length + 1):
                b = a + length
                # off before: before day 0 is working (not off)
                if a == 0:
                    off_before = 0
                else:
                    off_before = x[(n, a - 1, 0)]
                # off after: after last day is working (not off)
                if b == num_days:
                    off_after = 0
                else:
                    off_after = x[(n, b, 0)]
                work_in = gp.quicksum(x[(n, a + k, s)] for k in range(length) for s in range(1, num_shift_types + 1))
                rhs = length - 1
                if isinstance(off_before, gp.Var):
                    rhs += (1 - off_before)
                elif off_before == 0:
                    rhs += 1
                if isinstance(off_after, gp.Var):
                    rhs += (1 - off_after)
                elif off_after == 0:
                    rhs += 1
                model.addConstr(work_in <= rhs, name=f"minwork_{n}_{a}_{length}")

    # Min consecutive days off (no isolated short off blocks)
    for n_info in nurses:
        n = n_info['id']
        minoff = min_days_off[n]
        if minoff <= 1:
            continue
        for length in range(1, minoff):
            for a in range(num_days - length + 1):
                b = a + length
                # work before: before day 0 is off (work=0)
                if a == 0:
                    work_before = 0
                else:
                    work_before = gp.quicksum(x[(n, a - 1, s)] for s in range(1, num_shift_types + 1))
                # work after
                if b == num_days:
                    work_after = 0
                else:
                    work_after = gp.quicksum(x[(n, b, s)] for s in range(1, num_shift_types + 1))
                off_in = gp.quicksum(x[(n, a + k, 0)] for k in range(length))
                rhs = length - 1
                if isinstance(work_before, gp.LinExpr) or isinstance(work_before, gp.Var):
                    rhs += (1 - work_before)
                # if boundary off then work=0, no addition
                if isinstance(work_after, gp.LinExpr) or isinstance(work_after, gp.Var):
                    rhs += (1 - work_after)
                model.addConstr(off_in <= rhs, name=f"minoff_{n}_{a}_{length}")

    # Max weekends
    for n_info in nurses:
        n = n_info['id']
        mw = max_weekends_n[n]
        weekend_vars = []
        for w in range(num_weekends):
            days_w = [d for d, wid in weekend_of.items() if wid == w]
            if not days_w:
                continue
            worked = model.addVar(vtype=GRB.BINARY, name=f"wknd_{n}_{w}")
            any_work = gp.quicksum(x[(n, d, s)] for d in days_w for s in range(1, num_shift_types + 1))
            model.addConstr(any_work <= 7 * worked, name=f"wkndlink_{n}_{w}")  # loose but works since binary
            model.addConstr(worked >= any_work / len(days_w), name=f"wkndlink2_{n}_{w}")
            weekend_vars.append(worked)
        if weekend_vars:
            model.addConstr(gp.quicksum(weekend_vars) <= mw, name=f"maxwknd_{n}")

    # Objective: request penalties + coverage
    request_penalty = 0
    for n_info in nurses:
        n = n_info['id']
        for d in range(num_days):
            for s_model in range(1, num_shift_types + 1):
                s_orig = model_to_orig[s_model]
                # shift-on: penalty if not assigned
                key_on = (n, d, s_orig)
                if key_on in shift_on_pen:
                    request_penalty += shift_on_pen[key_on] * (1 - x[(n, d, s_model)])
                # shift-off: penalty if assigned
                key_off = (n, d, s_orig)
                if key_off in shift_off_pen:
                    request_penalty += shift_off_pen[key_off] * x[(n, d, s_model)]

    under_vars = {}
    over_vars = {}
    coverage_penalty = 0
    for cov in data['coverage_requirements']:
        d = cov['day']
        s_orig = cov['shift_type']
        if s_orig not in orig_to_model:
            continue
        s_model = orig_to_model[s_orig]
        pref = cov['preferred']
        uw = cov['under_weight']
        ow = cov['over_weight']
        assigned = gp.quicksum(x[(n_info['id'], d, s_model)] for n_info in nurses)
        under = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"under_{d}_{s_orig}")
        over = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"over_{d}_{s_orig}")
        model.addConstr(assigned + under - over == pref, name=f"cover_{d}_{s_orig}")
        coverage_penalty += uw * under + ow * over
        under_vars[(d, s_orig)] = under
        over_vars[(d, s_orig)] = over

    model.setObjective(request_penalty + coverage_penalty, GRB.MINIMIZE)

    best_obj = None
    def callback(model, where):
        nonlocal best_obj
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if best_obj is None or obj < best_obj - 1e-6:
                best_obj = obj
                if logger:
                    logger.log(float(obj))

    model.optimize(callback)

    if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED, GRB.SUBOPTIMAL):
        schedule = {}
        for n_info in nurses:
            nid_str = str(n_info['id'])
            schedule[nid_str] = {}
            n = n_info['id']
            for d in range(num_days):
                found = False
                for s in range(1, num_shift_types + 1):
                    if x[(n, d, s)].X > 0.5:
                        schedule[nid_str][str(d)] = int(model_to_orig[s])
                        found = True
                        break
                if not found:
                    schedule[nid_str][str(d)] = None

        coverage_under_out = {}
        coverage_over_out = {}
        for (d, s_orig), var in under_vars.items():
            coverage_under_out[f"{d}_{s_orig}"] = int(max(0, round(var.X)))
        for (d, s_orig), var in over_vars.items():
            coverage_over_out[f"{d}_{s_orig}"] = int(max(0, round(var.X)))

        final_obj = model.objVal if model.objVal is not None else (best_obj if best_obj is not None else 0.0)

        solution_dict = {
            "objective_value": float(final_obj),
            "solution": {
                "schedule": schedule,
                "coverage_under": coverage_under_out,
                "coverage_over": coverage_over_out
            }
        }

        with open(args.solution_path, 'w') as f:
            json.dump(solution_dict, f, indent=2)
    else:
        # Failure case
        solution_dict = {
            "objective_value": 1e9,
            "solution": {
                "schedule": {},
                "coverage_under": {},
                "coverage_over": {}
            }
        }
        with open(args.solution_path, 'w') as f:
            json.dump(solution_dict, f, indent=2)


if __name__ == "__main__":
    main()