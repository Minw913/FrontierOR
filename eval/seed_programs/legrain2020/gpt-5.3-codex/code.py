import argparse
import json
from collections import defaultdict
import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def build_and_solve(instance, time_limit, logger=None):
    # -------------------------
    # Basic data
    # -------------------------
    N = instance["num_nurses"]
    D = instance["num_days"]
    S = instance["num_shifts"]
    K = instance["num_skills"]

    nurses = instance["nurses"]
    shifts_data = instance["shifts"]
    demand_data = instance["demand"]
    penalties = instance["penalty_weights"]

    shift_names = [sh["name"] for sh in shifts_data]
    shift_to_idx = {name: i for i, name in enumerate(shift_names)}
    skill_names = instance["skills"]
    skill_to_idx = {name: i for i, name in enumerate(skill_names)}

    num_weeks = instance["num_weeks"]

    # Nurse skills matrix
    qual = [[0] * K for _ in range(N)]
    for i, n in enumerate(nurses):
        for sk in n["skills"]:
            if sk in skill_to_idx:
                qual[i][skill_to_idx[sk]] = 1

    # Shift consecutive bounds
    CS_minus = [sh["CS_minus"] for sh in shifts_data]
    CS_plus = [sh["CS_plus"] for sh in shifts_data]

    # Forbidden successions (indices)
    forbidden_pairs = []
    for p in instance.get("forbidden_shift_successions", []):
        if len(p) == 2 and p[0] in shift_to_idx and p[1] in shift_to_idx:
            forbidden_pairs.append((shift_to_idx[p[0]], shift_to_idx[p[1]]))

    # Demand arrays
    min_dem = [[[0 for _ in range(K)] for _ in range(S)] for _ in range(D)]
    opt_dem = [[[0 for _ in range(K)] for _ in range(S)] for _ in range(D)]
    for d in range(D):
        day_obj = demand_data[d] if d < len(demand_data) else {}
        for sh_name, sk_obj in day_obj.items():
            if sh_name not in shift_to_idx:
                continue
            s = shift_to_idx[sh_name]
            for sk_name, vals in sk_obj.items():
                if sk_name not in skill_to_idx:
                    continue
                k = skill_to_idx[sk_name]
                min_dem[d][s][k] = int(vals.get("minimum", 0))
                opt_dem[d][s][k] = int(vals.get("optimal", 0))

    # -------------------------
    # Model
    # -------------------------
    m = gp.Model("nurse_rostering")
    m.Params.TimeLimit = time_limit
    m.Params.Threads = 1
    m.Params.OutputFlag = 0

    # Decision vars
    # x[i,d,s,k] = nurse i works day d shift s with skill k
    x = m.addVars(N, D, S, K, vtype=GRB.BINARY, name="x")
    # y[i,d,s] = nurse i works shift s on day d
    y = m.addVars(N, D, S, vtype=GRB.BINARY, name="y")
    # w[i,d] = nurse i works any shift on day d
    w = m.addVars(N, D, vtype=GRB.BINARY, name="w")

    # Skill qualification hard constraints
    for i in range(N):
        for d in range(D):
            for s in range(S):
                for k in range(K):
                    if qual[i][k] == 0:
                        m.addConstr(x[i, d, s, k] == 0)

    # Link x, y, w and one shift/day
    for i in range(N):
        for d in range(D):
            m.addConstr(gp.quicksum(y[i, d, s] for s in range(S)) == w[i, d])
            for s in range(S):
                m.addConstr(gp.quicksum(x[i, d, s, k] for k in range(K)) == y[i, d, s])

    # Hard minimum demand and soft optimal shortfall
    u = m.addVars(D, S, K, lb=0.0, vtype=GRB.CONTINUOUS, name="short_opt")
    for d in range(D):
        for s in range(S):
            for k in range(K):
                cov = gp.quicksum(x[i, d, s, k] for i in range(N))
                m.addConstr(cov >= min_dem[d][s][k])
                m.addConstr(u[d, s, k] >= opt_dem[d][s][k] - cov)

    # Hard forbidden shift successions
    for i in range(N):
        for d in range(D - 1):
            for s1, s2 in forbidden_pairs:
                m.addConstr(y[i, d, s1] + y[i, d + 1, s2] <= 1)

    # Objective accumulator
    obj = gp.LinExpr()
    cS1 = penalties["c_S1"]
    cS2a = penalties["c_S2a"]
    cS2b = penalties["c_S2b"]
    cS3 = penalties["c_S3"]
    cS4 = penalties["c_S4"]
    cS5 = penalties["c_S5"]
    cS6 = penalties["c_S6"]
    cS7 = penalties["c_S7"]

    # Insufficient optimal staffing
    obj += cS1 * gp.quicksum(u[d, s, k] for d in range(D) for s in range(S) for k in range(K))

    # Preferences
    for i, nurse in enumerate(nurses):
        for pref in nurse.get("preferences", []):
            d = pref.get("day")
            sh = pref.get("shift")
            if isinstance(d, int) and 0 <= d < D and sh in shift_to_idx:
                obj += cS4 * y[i, d, shift_to_idx[sh]]

    # Per-nurse contract soft constraints
    weekend_work = {}
    incomplete_weekend = {}
    excess_weekends = {}
    under_total = {}
    over_total = {}

    for i, nurse in enumerate(nurses):
        contract = nurse["contract"]
        L_minus = contract["L_minus"]
        L_plus = contract["L_plus"]
        CD_minus = contract["CD_minus"]
        CD_plus = contract["CD_plus"]
        CR_minus = contract["CR_minus"]
        CR_plus = contract["CR_plus"]
        WE_plus = contract["WE_plus"]

        # Total assignments soft
        total_work = gp.quicksum(w[i, d] for d in range(D))
        under_total[i] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"under_total_{i}")
        over_total[i] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"over_total_{i}")
        m.addConstr(under_total[i] >= L_minus - total_work)
        m.addConstr(over_total[i] >= total_work - L_plus)
        obj += cS6 * (under_total[i] + over_total[i])

        # Weekend vars
        weekend_work[i] = {}
        incomplete_weekend[i] = {}
        for wk in range(num_weeks):
            sat = 7 * wk + 5
            sun = 7 * wk + 6
            if sun >= D:
                continue
            ww = m.addVar(vtype=GRB.BINARY, name=f"weekend_work_{i}_{wk}")
            iw = m.addVar(vtype=GRB.BINARY, name=f"incomplete_weekend_{i}_{wk}")
            weekend_work[i][wk] = ww
            incomplete_weekend[i][wk] = iw

            m.addConstr(ww >= w[i, sat])
            m.addConstr(ww >= w[i, sun])
            m.addConstr(ww <= w[i, sat] + w[i, sun])

            m.addConstr(iw >= w[i, sat] - w[i, sun])
            m.addConstr(iw >= w[i, sun] - w[i, sat])

            obj += cS5 * iw

        excess_weekends[i] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"excess_weekend_{i}")
        m.addConstr(excess_weekends[i] >= gp.quicksum(weekend_work[i][wk] for wk in weekend_work[i]) - WE_plus)
        obj += cS7 * excess_weekends[i]

        # Consecutive worked days (CD_minus / CD_plus)
        # Min: short work stretches
        if CD_minus > 1:
            for d0 in range(D):
                max_l = min(CD_minus - 1, D - d0)
                for l in range(1, max_l + 1):
                    v = m.addVar(vtype=GRB.BINARY, name=f"short_work_{i}_{d0}_{l}")
                    # upper bounds
                    for t in range(d0, d0 + l):
                        m.addConstr(v <= w[i, t])
                    if d0 > 0:
                        m.addConstr(v <= 1 - w[i, d0 - 1])
                    if d0 + l < D:
                        m.addConstr(v <= 1 - w[i, d0 + l])

                    # lower bound (activate if exact short stretch)
                    expr = gp.quicksum(w[i, t] for t in range(d0, d0 + l))
                    cnt = l
                    if d0 > 0:
                        expr += 1 - w[i, d0 - 1]
                    else:
                        expr += 1
                    cnt += 1
                    if d0 + l < D:
                        expr += 1 - w[i, d0 + l]
                    else:
                        expr += 1
                    cnt += 1
                    m.addConstr(v >= expr - (cnt - 1))

                    obj += cS2a * (CD_minus - l) * v

        # Max: every window of length CD_plus+1 fully worked => 1 excess day
        if CD_plus >= 0 and CD_plus + 1 <= D:
            Lw = CD_plus + 1
            for d0 in range(0, D - Lw + 1):
                z = m.addVar(vtype=GRB.BINARY, name=f"long_work_{i}_{d0}")
                for t in range(d0, d0 + Lw):
                    m.addConstr(z <= w[i, t])
                m.addConstr(z >= gp.quicksum(w[i, t] for t in range(d0, d0 + Lw)) - CD_plus)
                obj += cS2a * z

        # Consecutive days off (CR_minus / CR_plus)
        # Min: short off stretches
        if CR_minus > 1:
            for d0 in range(D):
                max_l = min(CR_minus - 1, D - d0)
                for l in range(1, max_l + 1):
                    v = m.addVar(vtype=GRB.BINARY, name=f"short_rest_{i}_{d0}_{l}")
                    for t in range(d0, d0 + l):
                        m.addConstr(v <= 1 - w[i, t])
                    if d0 > 0:
                        m.addConstr(v <= w[i, d0 - 1])
                    if d0 + l < D:
                        m.addConstr(v <= w[i, d0 + l])

                    expr = gp.quicksum(1 - w[i, t] for t in range(d0, d0 + l))
                    cnt = l
                    if d0 > 0:
                        expr += w[i, d0 - 1]
                    else:
                        expr += 1
                    cnt += 1
                    if d0 + l < D:
                        expr += w[i, d0 + l]
                    else:
                        expr += 1
                    cnt += 1
                    m.addConstr(v >= expr - (cnt - 1))

                    obj += cS3 * (CR_minus - l) * v

        # Max: every window of length CR_plus+1 fully off => 1 excess off day
        if CR_plus >= 0 and CR_plus + 1 <= D:
            Lr = CR_plus + 1
            for d0 in range(0, D - Lr + 1):
                z = m.addVar(vtype=GRB.BINARY, name=f"long_rest_{i}_{d0}")
                for t in range(d0, d0 + Lr):
                    m.addConstr(z <= 1 - w[i, t])
                m.addConstr(z >= gp.quicksum(1 - w[i, t] for t in range(d0, d0 + Lr)) - CR_plus)
                obj += cS3 * z

        # Consecutive same-shift constraints per shift
        for s in range(S):
            smin = CS_minus[s]
            smax = CS_plus[s]

            # Min same-shift stretches
            if smin > 1:
                for d0 in range(D):
                    max_l = min(smin - 1, D - d0)
                    for l in range(1, max_l + 1):
                        v = m.addVar(vtype=GRB.BINARY, name=f"short_shift_{i}_{s}_{d0}_{l}")
                        for t in range(d0, d0 + l):
                            m.addConstr(v <= y[i, t, s])
                        if d0 > 0:
                            m.addConstr(v <= 1 - y[i, d0 - 1, s])
                        if d0 + l < D:
                            m.addConstr(v <= 1 - y[i, d0 + l, s])

                        expr = gp.quicksum(y[i, t, s] for t in range(d0, d0 + l))
                        cnt = l
                        if d0 > 0:
                            expr += 1 - y[i, d0 - 1, s]
                        else:
                            expr += 1
                        cnt += 1
                        if d0 + l < D:
                            expr += 1 - y[i, d0 + l, s]
                        else:
                            expr += 1
                        cnt += 1
                        m.addConstr(v >= expr - (cnt - 1))

                        obj += cS2b * (smin - l) * v

            # Max same-shift stretches
            if smax >= 0 and smax + 1 <= D:
                Ls = smax + 1
                for d0 in range(0, D - Ls + 1):
                    z = m.addVar(vtype=GRB.BINARY, name=f"long_shift_{i}_{s}_{d0}")
                    for t in range(d0, d0 + Ls):
                        m.addConstr(z <= y[i, t, s])
                    m.addConstr(z >= gp.quicksum(y[i, t, s] for t in range(d0, d0 + Ls)) - smax)
                    obj += cS2b * z

    m.setObjective(obj, GRB.MINIMIZE)

    # -------------------------
    # Incumbent logging callback
    # -------------------------
    def cb(model, where):
        if where == GRB.Callback.MIPSOL and model._logger is not None:
            val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if model._best is None or val < model._best - 1e-9:
                model._best = val
                model._logger.log(float(val))

    m._logger = logger
    m._best = None

    m.optimize(cb)

    # -------------------------
    # Extract solution
    # -------------------------
    if m.SolCount == 0:
        # No feasible solution found
        return {
            "objective_value": 1e18,
            "schedule": [
                {"nurse_id": nurses[i]["id"], "nurse_name": nurses[i]["name"], "assignments": []}
                for i in range(N)
            ],
        }

    objective_value = float(m.ObjVal)
    schedule_out = []

    for i, nurse in enumerate(nurses):
        assigns = []
        for d in range(D):
            if w[i, d].X > 0.5:
                # find shift
                chosen_s = None
                for s in range(S):
                    if y[i, d, s].X > 0.5:
                        chosen_s = s
                        break
                if chosen_s is None:
                    continue

                # find skill
                chosen_k = None
                for k in range(K):
                    if x[i, d, chosen_s, k].X > 0.5:
                        chosen_k = k
                        break
                if chosen_k is None:
                    # fallback: first qualified skill
                    for k in range(K):
                        if qual[i][k] == 1:
                            chosen_k = k
                            break
                if chosen_k is None:
                    continue

                assigns.append(
                    {
                        "day": d,
                        "shift": shift_names[chosen_s],
                        "skill": skill_names[chosen_k],
                    }
                )

        schedule_out.append(
            {
                "nurse_id": nurse["id"],
                "nurse_name": nurse["name"],
                "assignments": assigns,
            }
        )

    return {"objective_value": objective_value, "schedule": schedule_out}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    sol = build_and_solve(instance, args.time_limit, logger=logger)

    with open(args.solution_path, "w") as f:
        json.dump(sol, f, indent=2)


if __name__ == "__main__":
    main()