import argparse
import json
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def greedy_solve(ops, job_ops, num_machines):
    if not any(job_ops.values()):
        return 0, {}, {}
    machine_avail = [1] * num_machines
    job_avail = {jid: 1 for jid in job_ops}
    remaining = {jid: list(job_ops[jid]) for jid in job_ops}
    schedule = {}
    assignment = {}
    while any(remaining[jid] for jid in remaining):
        candidates = sorted((job_avail[jid], jid) for jid in remaining if remaining[jid])
        jid = candidates[0][1]
        op_list = remaining[jid]
        oid, idx, em, pts = op_list[0]
        best = None
        for m in em:
            start = max(job_avail[jid], machine_avail[m])
            end = start + pts[m]
            if best is None or end < best[0]:
                best = (end, m, start)
        end, m, start = best
        machine_avail[m] = end
        job_avail[jid] = end
        assignment[(jid, oid)] = m
        schedule[(jid, oid)] = list(range(start, end))
        op_list.pop(0)
    makespan = max(job_avail[jid] for jid in job_ops if job_ops[jid])
    return makespan, assignment, schedule


def format_solution(obj, assign, sched):
    return {
        'objective_value': float(obj),
        'assignment': {f'({jid},{oid})': m for (jid, oid), m in assign.items()},
        'schedule': {f'({jid},{oid})': p for (jid, oid), p in sched.items()},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    with open(args.instance_path) as f:
        data = json.load(f)

    num_machines = data['num_machines']
    jobs_data = data['jobs']

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    ops = []
    job_ops = {}
    for job in jobs_data:
        jid = job['job_id']
        job_ops[jid] = []
        for op in job['operations']:
            oid = op['operation_id']
            em = op['eligible_machines']
            pt = op['processing_times']
            pts = {int(k): v for k, v in pt.items()}
            idx = len(ops)
            ops.append((jid, oid, em, pts))
            job_ops[jid].append((oid, idx, em, pts))

    wall_start = time.time()

    ub, greedy_assign, greedy_sched = greedy_solve(ops, job_ops, num_machines)

    if logger:
        logger.log_solution(float(ub), format_solution(ub, greedy_assign, greedy_sched))

    T = ub
    if T == 0:
        with open(args.solution_path, 'w') as f:
            json.dump(format_solution(0.0, {}, {}), f, indent=2)
        return

    best_obj = ub
    best_assign = dict(greedy_assign)
    best_sched = dict(greedy_sched)

    total_x = sum(len(em) * T for (jid, oid, em, pts) in ops)

    if total_x <= 250000 and args.time_limit > 5:
        try:
            model = gp.Model()
            model.setParam('Seed', 0)
            model.setParam('MIPGap', 1e-4)
            model.setParam('NumericFocus', 0)
            model.setParam('Threads', 1)
            model.setParam('OutputFlag', 0)

            C = model.addVar(lb=0, ub=T, vtype=GRB.INTEGER, name='C')

            eligible_pairs = [(idx, m) for idx, (jid, oid, em, pts) in enumerate(ops) for m in em]
            y = model.addVars(eligible_pairs, vtype=GRB.BINARY, name='y')
            x_keys = [(idx, m, t) for (idx, m) in eligible_pairs for t in range(1, T + 1)]
            x = model.addVars(x_keys, vtype=GRB.BINARY, name='x')

            # Assignment: exactly one machine per operation
            for idx, (jid, oid, em, pts) in enumerate(ops):
                model.addConstr(gp.quicksum(y[idx, m] for m in em) == 1)

            # Processing time (equality for tightness; x<=y is implied by sum=0 when y=0)
            model.addConstrs(
                (gp.quicksum(x[idx, m, t] for t in range(1, T + 1)) == pts[m] * y[idx, m]
                 for idx, (jid, oid, em, pts) in enumerate(ops) for m in em),
                name='proc',
            )

            # Machine capacity
            ops_on_machine = {m: [] for m in range(num_machines)}
            for idx, (jid, oid, em, pts) in enumerate(ops):
                for m in em:
                    ops_on_machine[m].append(idx)
            model.addConstrs(
                (gp.quicksum(x[idx, m, t] for idx in ops_on_machine[m]) <= 1
                 for m in range(num_machines) if ops_on_machine[m] for t in range(1, T + 1)),
                name='mach',
            )

            # Precedence: successor at t => predecessor has no processing at t' >= t
            def prec_gen():
                for jid, op_list in job_ops.items():
                    for i in range(len(op_list) - 1):
                        oid_p, idx_p, em_p, pts_p = op_list[i]
                        oid_s, idx_s, em_s, pts_s = op_list[i + 1]
                        M = max(pts_p[m] for m in em_p)
                        for t in range(1, T + 1):
                            yield (M * gp.quicksum(x[idx_s, m_s, t] for m_s in em_s)
                                   + gp.quicksum(x[idx_p, m_p, tp] for m_p in em_p for tp in range(t, T + 1)) <= M)
            model.addConstrs(prec_gen(), name='prec')

            # Makespan
            def makespan_gen():
                for jid, op_list in job_ops.items():
                    if not op_list:
                        continue
                    oid_last, idx_last, em_last, pts_last = op_list[-1]
                    for t in range(1, T + 1):
                        yield C >= (t + 1) * gp.quicksum(x[idx_last, m, t] for m in em_last)
            model.addConstrs(makespan_gen(), name='ms')

            # Warm start with greedy
            for idx, (jid, oid, em, pts) in enumerate(ops):
                if (jid, oid) in greedy_assign:
                    gm = greedy_assign[(jid, oid)]
                    for m in em:
                        y[idx, m].Start = 1 if m == gm else 0
                    sset = set(greedy_sched[(jid, oid)])
                    for m in em:
                        for t in range(1, T + 1):
                            x[idx, m, t].Start = 1 if (m == gm and t in sset) else 0

            model.setObjective(C, GRB.MINIMIZE)

            elapsed = time.time() - wall_start
            remaining = max(1, args.time_limit - elapsed - 2)
            model.setParam('TimeLimit', remaining)

            model.optimize()

            if model.SolCount > 0:
                obj_val = model.ObjVal
                if obj_val < best_obj - 1e-6:
                    if abs(obj_val - round(obj_val)) < 1e-6:
                        obj_val = round(obj_val)
                    assignment = {}
                    schedule = {}
                    for idx, (jid, oid, em, pts) in enumerate(ops):
                        for m in em:
                            if y[idx, m].X > 0.5:
                                assignment[(jid, oid)] = m
                                schedule[(jid, oid)] = [t for t in range(1, T + 1) if x[idx, m, t].X > 0.5]
                                break
                    best_obj = obj_val
                    best_assign = assignment
                    best_sched = schedule
                    if logger:
                        logger.log_solution(float(obj_val), format_solution(obj_val, assignment, schedule))
        except Exception:
            pass

    solution = format_solution(best_obj, best_assign, best_sched)
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == '__main__':
    main()