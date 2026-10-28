import argparse
import json
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--instance_path', required=True)
    ap.add_argument('--solution_path', required=True)
    ap.add_argument('--time_limit', type=int, default=300)
    ap.add_argument('--log_path', default=None)
    args = ap.parse_args()

    start_time = time.time()
    deadline = start_time + max(5, args.time_limit)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path) as f:
        data = json.load(f)

    nf = data['num_facilities']
    nc = data['num_customers']
    d = data['demands']
    cap = data['capacities']
    f1 = data['fixed_costs_obj1']
    f2 = data['fixed_costs_obj2']
    c1 = data['assignment_costs_obj1']
    c2 = data['assignment_costs_obj2']

    def sol_costs(assign):
        opens = sorted(set(assign.values()))
        z1v = sum(f1[i] for i in opens) + sum(c1[assign[j]][j] for j in range(nc))
        z2v = sum(f2[i] for i in opens) + sum(c2[assign[j]][j] for j in range(nc))
        return z1v, z2v, opens

    def make_entry(assign):
        z1v, z2v, opens = sol_costs(assign)
        entry = {
            "z1": int(z1v),
            "z2": int(z2v),
            "open_facilities": [int(i) for i in opens],
            "assignments": {str(j): int(assign[j]) for j in range(nc)},
        }
        return (int(z1v), int(z2v), entry)

    def pareto_filter(points):
        pts = sorted(points, key=lambda p: (p[0], p[1]))
        res = []
        best2 = None
        for p in pts:
            if best2 is None or p[1] < best2:
                res.append(p)
                best2 = p[1]
        return res

    def build_output(points):
        front = pareto_filter(points)
        if not front:
            return {"objective_value": float('inf'), "pareto_front": [], "solutions": []}
        obj = min(0.5 * (p[0] + p[1]) for p in front)
        return {
            "objective_value": float(obj),
            "pareto_front": [[p[0], p[1]] for p in front],
            "solutions": [p[2] for p in front],
        }

    points = []
    best_combined = [float('inf')]

    def maybe_log(z1v, z2v):
        comb = 0.5 * (z1v + z2v)
        if comb < best_combined[0] - 1e-9:
            best_combined[0] = comb
            if logger:
                logger.log_solution(comb, build_output(points))

    # ---------- Greedy fallback / warm start ----------
    def greedy_solution():
        order = sorted(range(nf), key=lambda i: -cap[i])
        total_d = sum(d)
        for k in range(1, nf + 1):
            opened = order[:k]
            if sum(cap[i] for i in opened) < total_d:
                continue
            rem = {i: cap[i] for i in opened}
            assign = {}
            ok = True
            for j in sorted(range(nc), key=lambda jj: -d[jj]):
                cands = [i for i in opened if rem[i] >= d[j]]
                if not cands:
                    ok = False
                    break
                i = min(cands, key=lambda ii: c1[ii][j] + c2[ii][j])
                assign[j] = i
                rem[i] -= d[j]
            if ok:
                return assign
        return None

    greedy_assign = greedy_solution()
    if greedy_assign is not None:
        p = make_entry(greedy_assign)
        points.append(p)
        maybe_log(p[0], p[1])

    # ---------- Build MIP ----------
    m = gp.Model()
    m.Params.OutputFlag = 0
    m.Params.Seed = 0
    m.Params.MIPGap = 1e-4
    m.Params.NumericFocus = 0
    m.Params.Threads = 1

    y = m.addVars(nf, vtype=GRB.BINARY, name="y")
    x = m.addVars(nf, nc, vtype=GRB.BINARY, name="x")
    Z1 = m.addVar(lb=-GRB.INFINITY, name="Z1")
    Z2 = m.addVar(lb=-GRB.INFINITY, name="Z2")

    for j in range(nc):
        m.addConstr(gp.quicksum(x[i, j] for i in range(nf)) == 1)
    for i in range(nf):
        m.addConstr(gp.quicksum(d[j] * x[i, j] for j in range(nc)) <= cap[i] * y[i])
        for j in range(nc):
            m.addConstr(x[i, j] <= y[i])

    m.addConstr(Z1 == gp.quicksum(f1[i] * y[i] for i in range(nf)) +
                gp.quicksum(c1[i][j] * x[i, j] for i in range(nf) for j in range(nc)))
    m.addConstr(Z2 == gp.quicksum(f2[i] * y[i] for i in range(nf)) +
                gp.quicksum(c2[i][j] * x[i, j] for i in range(nf) for j in range(nc)))

    UB2 = sum(f2) + sum(max(c2[i][j] for i in range(nf)) for j in range(nc)) + 1
    eps_con = m.addConstr(Z2 <= UB2)

    # Warm start from greedy
    if greedy_assign is not None:
        opens = set(greedy_assign.values())
        for i in range(nf):
            y[i].Start = 1.0 if i in opens else 0.0
            for j in range(nc):
                x[i, j].Start = 1.0 if greedy_assign.get(j) == i else 0.0

    xlist = [x[i, j] for i in range(nf) for j in range(nc)]

    def extract_from_model():
        assign = {}
        xv = m.getAttr('X', xlist)
        for j in range(nc):
            for i in range(nf):
                if xv[i * nc + j] > 0.5:
                    assign[j] = i
                    break
        return make_entry(assign)

    def wcb(model, where):
        if where == GRB.Callback.MIPSOL:
            try:
                xv = model.cbGetSolution(xlist)
            except gp.GurobiError:
                return
            assign = {}
            for j in range(nc):
                for i in range(nf):
                    if xv[i * nc + j] > 0.5:
                        assign[j] = i
                        break
            if len(assign) == nc:
                z1v, z2v, opens = sol_costs(assign)
                comb = 0.5 * (z1v + z2v)
                if comb < best_combined[0] - 1e-9:
                    entry = {
                        "z1": int(z1v), "z2": int(z2v),
                        "open_facilities": [int(i) for i in sorted(opens)],
                        "assignments": {str(j): int(assign[j]) for j in range(nc)},
                    }
                    tmp = points + [(int(z1v), int(z2v), entry)]
                    best_combined[0] = comb
                    if logger:
                        logger.log_solution(comb, build_output(tmp))

    # ---------- Weighted-sum solve (0.5, 0.5) ----------
    rem = deadline - time.time() - 1.0
    if rem > 1:
        m.setObjective(0.5 * Z1 + 0.5 * Z2, GRB.MINIMIZE)
        m.Params.TimeLimit = max(1.0, min(rem, 0.4 * args.time_limit))
        m.optimize(wcb)
        if m.SolCount > 0:
            p = extract_from_model()
            points.append(p)
            maybe_log(p[0], p[1])

    # ---------- Epsilon-constraint enumeration of Pareto front ----------
    eps = UB2
    while True:
        rem = deadline - time.time() - 0.5
        if rem < 1.0:
            break
        eps_con.RHS = eps + 0.5
        m.setObjective(Z1, GRB.MINIMIZE)
        m.Params.TimeLimit = rem
        m.optimize()
        if m.Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
            break
        if m.SolCount == 0:
            break
        z1v, z2v, entry = extract_from_model()
        timedout = (m.Status == GRB.TIME_LIMIT)

        if not timedout:
            rem = deadline - time.time() - 0.5
            if rem >= 1.0:
                tcon = m.addConstr(Z1 <= z1v + 0.4)
                m.setObjective(Z2, GRB.MINIMIZE)
                m.Params.TimeLimit = rem
                m.optimize()
                if m.SolCount > 0:
                    z1v, z2v, entry = extract_from_model()
                    if m.Status == GRB.TIME_LIMIT:
                        timedout = True
                m.remove(tcon)

        points.append((z1v, z2v, entry))
        maybe_log(z1v, z2v)

        if timedout:
            break
        eps = z2v - 1

    # ---------- Output ----------
    out = build_output(points)
    with open(args.solution_path, 'w') as f:
        json.dump(out, f, indent=2)

    if logger and out["solutions"] and out["objective_value"] < best_combined[0] - 1e-9:
        logger.log_solution(out["objective_value"], out)


if __name__ == '__main__':
    main()