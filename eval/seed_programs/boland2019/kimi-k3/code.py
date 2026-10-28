import argparse
import json
import time

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def main():
    parser = argparse.ArgumentParser(
        description="Bi-objective single-assignment capacitated facility location"
    )
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, default=300)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    t0 = time.time()
    limit = float(args.time_limit)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as fh:
        inst = json.load(fh)

    nF = int(inst["num_facilities"])
    nC = int(inst["num_customers"])
    demands = [int(v) for v in inst["demands"]]
    capacities = [int(v) for v in inst["capacities"]]
    f1 = [int(v) for v in inst["fixed_costs_obj1"]]
    f2 = [int(v) for v in inst["fixed_costs_obj2"]]
    a1 = [[int(v) for v in row] for row in inst["assignment_costs_obj1"]]
    a2 = [[int(v) for v in row] for row in inst["assignment_costs_obj2"]]

    points = {}        # (z1, z2) -> (open_facilities, assignments)
    fallback = [None]  # best unproven feasible incumbent (z1, z2, open, assign)
    best_obj = [float("inf")]

    def build_result():
        items = sorted(points.items(), key=lambda kv: (kv[0][0], kv[0][1]))
        front = [[int(k[0]), int(k[1])] for k, _ in items]
        sols = [
            {
                "z1": int(k[0]),
                "z2": int(k[1]),
                "open_facilities": [int(i) for i in v[0]],
                "assignments": {str(c): int(fac) for c, fac in v[1].items()},
            }
            for k, v in items
        ]
        if items:
            bk = min(items, key=lambda kv: 0.5 * (kv[0][0] + kv[0][1]))[0]
            obj = 0.5 * (bk[0] + bk[1])
        else:
            obj = 0.0
        return {"objective_value": float(obj), "pareto_front": front, "solutions": sols}

    def register(zv1, zv2, open_f, assign):
        key = (int(zv1), int(zv2))
        points[key] = (list(open_f), dict(assign))
        result = build_result()
        if result["objective_value"] < best_obj[0] - 1e-9:
            best_obj[0] = result["objective_value"]
            if logger is not None:
                logger.log_solution(result["objective_value"], result)

    try:
        model = gp.Model("bi_cfl")
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1
        model.Params.OutputFlag = 0

        y = model.addVars(nF, vtype=GRB.BINARY, name="y")
        x = model.addVars(nF, nC, vtype=GRB.BINARY, name="x")

        # A customer can never fit into a facility whose capacity is smaller
        # than the customer's demand: fix those assignment variables to zero.
        for i in range(nF):
            for j in range(nC):
                if capacities[i] < demands[j]:
                    x[i, j].UB = 0.0

        model.addConstrs((x.sum("*", j) == 1 for j in range(nC)), name="assign")
        model.addConstrs(
            (
                gp.quicksum(demands[j] * x[i, j] for j in range(nC))
                <= capacities[i] * y[i]
                for i in range(nF)
            ),
            name="capacity",
        )
        model.addConstrs(
            (x[i, j] <= y[i] for i in range(nF) for j in range(nC)), name="link"
        )
        model.addConstr(
            gp.quicksum(capacities[i] * y[i] for i in range(nF)) >= sum(demands),
            name="total_capacity",
        )

        z1 = gp.quicksum(f1[i] * y[i] for i in range(nF)) + gp.quicksum(
            a1[i][j] * x[i, j] for i in range(nF) for j in range(nC)
        )
        z2 = gp.quicksum(f2[i] * y[i] for i in range(nF)) + gp.quicksum(
            a2[i][j] * x[i, j] for i in range(nF) for j in range(nC)
        )

        eps_con = model.addConstr(z2 <= GRB.INFINITY, name="eps_z2")

        def remaining():
            return limit - (time.time() - t0)

        def extract_solution():
            open_f = [i for i in range(nF) if y[i].X > 0.5]
            zz1 = sum(f1[i] for i in open_f)
            zz2 = sum(f2[i] for i in open_f)
            assign = {}
            for j in range(nC):
                chosen = -1
                for i in range(nF):
                    if x[i, j].X > 0.5:
                        chosen = i
                        break
                if chosen < 0:
                    chosen = open_f[0] if open_f else 0
                assign[str(j)] = chosen
                zz1 += a1[chosen][j]
                zz2 += a2[chosen][j]
            return zz1, zz2, open_f, assign

        def greedy_start():
            rem_cap = list(capacities)
            opened = [False] * nF
            asg = [-1] * nC
            order = sorted(range(nC), key=lambda j: -demands[j])
            for j in order:
                bi, bc = -1, None
                for i in range(nF):
                    if rem_cap[i] >= demands[j]:
                        c = a1[i][j] + a2[i][j]
                        if not opened[i]:
                            c += f1[i] + f2[i]
                        if bc is None or c < bc:
                            bc, bi = c, i
                if bi < 0:
                    return False
                asg[j] = bi
                rem_cap[bi] -= demands[j]
                opened[bi] = True
            for i in range(nF):
                y[i].Start = 1.0 if opened[i] else 0.0
            for j in range(nC):
                if asg[j] >= 0:
                    x[asg[j], j].Start = 1.0
            return True

        proven_infeasible = False

        # Phase 0: blended equal-weight objective for a fast, strong incumbent
        # (its minimizer is a supported non-dominated point).
        if remaining() > 1.0:
            try:
                greedy_start()
            except Exception:
                pass
            model.setObjectiveN(z1, index=0, priority=1, weight=1.0, name="z1")
            model.setObjectiveN(z2, index=1, priority=1, weight=1.0, name="z2")
            model.Params.TimeLimit = max(0.05, remaining() - 0.5)
            model.optimize()
            if model.Status == GRB.INFEASIBLE:
                proven_infeasible = True
            elif model.SolCount > 0:
                zv1, zv2, of, asg = extract_solution()
                if model.Status == GRB.OPTIMAL:
                    register(zv1, zv2, of, asg)
                else:
                    fallback[0] = (zv1, zv2, of, asg)

        # Phase 1: epsilon-constraint enumeration of the complete Pareto front.
        # Each iteration lexicographically minimizes z1 then z2 subject to
        # z2 <= (previous z2) - 1, which yields every non-dominated point.
        if not proven_infeasible:
            model.setObjectiveN(z1, index=0, priority=2, weight=1.0, name="z1")
            model.setObjectiveN(z2, index=1, priority=1, weight=1.0, name="z2")
            eps2 = None
            while remaining() > 1.0:
                if eps2 is not None:
                    eps_con.RHS = eps2
                model.Params.TimeLimit = max(0.05, remaining() - 0.5)
                model.optimize()
                st = model.Status
                if st == GRB.INFEASIBLE:
                    break
                if st != GRB.OPTIMAL:
                    if model.SolCount > 0:
                        zv1, zv2, of, asg = extract_solution()
                        fallback[0] = (zv1, zv2, of, asg)
                    break
                zv1, zv2, of, asg = extract_solution()
                register(zv1, zv2, of, asg)
                eps2 = zv2 - 1

    except Exception:
        pass

    if not points and fallback[0] is not None:
        zv1, zv2, of, asg = fallback[0]
        register(zv1, zv2, of, asg)

    result = build_result()
    with open(args.solution_path, "w") as fh:
        json.dump(result, fh)


if __name__ == "__main__":
    main()