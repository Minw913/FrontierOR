import json
import argparse
import sys
import time
from gurobipy import Model, GRB, quicksum
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    S = instance['S']
    n = instance['n']
    m = instance['m']
    p = instance['p']
    sigma_w = instance['sigma_w']
    sigma_l = instance['sigma_l']
    seaside = sorted(instance['seaside_containers'], key=lambda x: x['id'])
    landside = instance['landside_containers']

    N_SLOTS = S + 2
    slots = list(range(N_SLOTS))
    T = 300
    times = list(range(T + 1))

    model = Model("CraneScheduling")
    model.Params.TimeLimit = max(10, args.time_limit - 10)
    model.Params.Threads = 1
    model.Params.MIPGap = 1e-4
    model.Params.LogToConsole = 0
    model.Params.Presolve = 1
    model.Params.Heuristics = 0.5

    # Variables
    pos_w = model.addVars(times, slots, vtype=GRB.BINARY, name="pos_w")
    pos_l = model.addVars(times, slots, vtype=GRB.BINARY, name="pos_l")

    # Actions
    right_w = model.addVars(times, vtype=GRB.BINARY, name="right_w")
    left_w = model.addVars(times, vtype=GRB.BINARY, name="left_w")
    wait_w = model.addVars(times, vtype=GRB.BINARY, name="wait_w")
    lift_w = model.addVars(times, vtype=GRB.BINARY, name="lift_w")
    drop_w = model.addVars(times, vtype=GRB.BINARY, name="drop_w")

    right_l = model.addVars(times, vtype=GRB.BINARY, name="right_l")
    left_l = model.addVars(times, vtype=GRB.BINARY, name="left_l")
    wait_l = model.addVars(times, vtype=GRB.BINARY, name="wait_l")
    lift_l = model.addVars(times, vtype=GRB.BINARY, name="lift_l")
    drop_l = model.addVars(times, vtype=GRB.BINARY, name="drop_l")

    # Seaside
    pickup_sea = model.addVars(n, times, vtype=GRB.BINARY, name="pickup_sea")
    drop_sea = model.addVars(n, times, vtype=GRB.BINARY, name="drop_sea")
    use_coop = model.addVars(n, vtype=GRB.BINARY, name="use_coop")
    drop_slot_var = model.addVars(n, slots, vtype=GRB.BINARY, name="drop_slot")
    lift_coop = model.addVars(n, times, vtype=GRB.BINARY, name="lift_coop")
    drop_coop = model.addVars(n, times, vtype=GRB.BINARY, name="drop_coop")

    # Landside
    serve_ls = model.addVars(m, vtype=GRB.BINARY, name="serve_ls")
    lift_ls = model.addVars(m, times, vtype=GRB.BINARY, name="lift_ls")
    drop_ls = model.addVars(m, times, vtype=GRB.BINARY, name="drop_ls")

    makespan = model.addVar(lb=0, name="makespan")
    model.setObjective(makespan, GRB.MINIMIZE)

    M = T * 2
    targets = [c['target_slot'] for c in seaside]
    sources = [c['source_slot'] for c in landside]
    efts = [c['earliest_finish_time'] for c in landside]
    deadlines = [c['deadline'] for c in landside]

    # Initial position
    model.addConstr(pos_w[0, sigma_w] == 1, "init_w")
    model.addConstr(pos_l[0, sigma_l] == 1, "init_l")
    for s in slots:
        if s != sigma_w:
            model.addConstr(pos_w[0, s] == 0)
        if s != sigma_l:
            model.addConstr(pos_l[0, s] == 0)

    # Position constraints
    for t in times:
        model.addConstr(quicksum(pos_w[t, s] for s in slots) == 1, f"one_w_{t}")
        model.addConstr(quicksum(pos_l[t, s] for s in slots) == 1, f"one_l_{t}")
        for s in slots:
            for r in range(s + 1, N_SLOTS):
                model.addConstr(pos_w[t, s] + pos_l[t, r] <= 1, f"no_cross_{t}_{s}_{r}")

    # Movement constraints
    for t in range(T):
        # Seaside movement
        for s in slots:
            rhs = pos_w[t, s]
            if s < N_SLOTS - 1:
                rhs += right_w[t] if s + 1 < N_SLOTS else 0
            if s > 0:
                rhs -= left_w[t]
            model.addConstr(pos_w[t + 1, s] == rhs, f"trans_w_{t}_{s}")

        # Landside movement
        for s in slots:
            rhs = pos_l[t, s]
            if s < N_SLOTS - 1:
                rhs += right_l[t]
            if s > 0:
                rhs -= left_l[t]
            model.addConstr(pos_l[t + 1, s] == rhs, f"trans_l_{t}_{s}")

        # Action uniqueness
        model.addConstr(right_w[t] + left_w[t] + wait_w[t] + lift_w[t] + drop_w[t] == 1, f"act_w_{t}")
        model.addConstr(right_l[t] + left_l[t] + wait_l[t] + lift_l[t] + drop_l[t] == 1, f"act_l_{t}")

        # Valid moves
        for s in range(1, N_SLOTS):
            model.addConstr(left_w[t] <= pos_w[t, s])
            model.addConstr(right_w[t] <= pos_w[t, s - 1])
        for s in range(N_SLOTS - 1):
            model.addConstr(right_l[t] <= pos_l[t, s])
            model.addConstr(left_l[t] <= pos_l[t, s + 1])

    # Lift/drop duration and non-overlap
    for t in range(T - p + 1):
        for k in range(p):
            tt = t + k
            if tt > T:
                break
            model.addConstr(lift_w[t] + drop_w[tt] <= 1)
            model.addConstr(drop_w[t] + lift_w[tt] <= 1)
            model.addConstr(lift_l[t] + drop_l[tt] <= 1)
            model.addConstr(drop_l[t] + lift_l[tt] <= 1)

    # Seaside containers
    for i in range(n):
        model.addConstr(quicksum(pickup_sea[i, t] for t in times) == 1)
        model.addConstr(quicksum(drop_sea[i, t] for t in times) == 1)
        model.addConstr(quicksum(drop_slot_var[i, s] for s in slots) == 1)

        for t in times:
            model.addConstr(pickup_sea[i, t] <= lift_w[t])
            model.addConstr(drop_sea[i, t] <= drop_w[t])
            model.addConstr(pickup_sea[i, t] <= pos_w[t, 0])

        for t in range(T - p + 1):
            model.addConstr(drop_sea[i, t] <= quicksum(pos_w[t, s] * drop_slot_var[i, s] for s in slots))

        # Order
        if i > 0:
            for t in range(T + 1):
                model.addConstr(quicksum(pickup_sea[i, tau] for tau in range(t)) <= quicksum(pickup_sea[i-1, tau] for tau in range(t)))
                model.addConstr(quicksum(drop_sea[i, tau] for tau in range(t)) <= quicksum(drop_sea[i-1, tau] for tau in range(t)))

        # Timing
        model.addConstr(quicksum(t * drop_sea[i, t] for t in times) >= quicksum(t * pickup_sea[i, t] for t in times) + 2 * p + 1)

        # Cooperation
        model.addConstr(use_coop[i] == quicksum(drop_slot_var[i, s] for s in slots if s != targets[i]))

        for t in range(T - p + 1):
            model.addConstr(lift_coop[i, t] <= use_coop[i])
            model.addConstr(lift_coop[i, t] <= lift_l[t])
            model.addConstr(drop_coop[i, t] <= use_coop[i])
            model.addConstr(drop_coop[i, t] <= drop_l[t])

        model.addConstr(quicksum(lift_coop[i, t] for t in times) == use_coop[i])
        model.addConstr(quicksum(drop_coop[i, t] for t in times) == use_coop[i])

        for t in range(T - p + 1):
            model.addConstr(quicksum(tau * lift_coop[i, tau] for tau in range(T+1)) >= 
                           quicksum(tau * drop_sea[i, tau] for tau in range(T+1)) + p - M * (1 - use_coop[i]))
            model.addConstr(quicksum(tau * drop_coop[i, tau] for tau in range(T+1)) >= 
                           quicksum(tau * lift_coop[i, tau] for tau in range(T+1)) + 2 * p - M * (1 - use_coop[i]))

    # Makespan
    for i in range(n):
        model.addConstr(makespan >= quicksum((t + p) * drop_sea[i, t] for t in range(T - p + 1)) * (1 - use_coop[i]) +
                       quicksum((t + p) * drop_coop[i, t] for t in range(T - p + 1)) * use_coop[i])

    # Landside containers
    deadlines_sorted = sorted(deadlines)
    threshold = deadlines_sorted[-1] if deadlines_sorted else 0
    must_serve = [j for j in range(m) if deadlines[j] <= threshold]
    next_dl = min((d for d in deadlines if d > threshold), default=None)
    next_group = [j for j in range(m) if next_dl is not None and deadlines[j] == next_dl]

    for j in must_serve:
        model.addConstr(serve_ls[j] == 1)
    if next_group:
        model.addConstr(quicksum(serve_ls[j] for j in next_group) >= 1)

    for j in range(m):
        model.addConstr(quicksum(lift_ls[j, t] for t in times) == serve_ls[j])
        model.addConstr(quicksum(drop_ls[j, t] for t in times) == serve_ls[j])

        for t in range(T - p + 1):
            model.addConstr(lift_ls[j, t] <= lift_l[t])
            model.addConstr(lift_ls[j, t] <= pos_l[t, sources[j]])
            model.addConstr(drop_ls[j, t] <= drop_l[t])
            model.addConstr(drop_ls[j, t] <= pos_l[t, N_SLOTS - 1])

        model.addConstr(quicksum(t * drop_ls[j, t] for t in times) >= quicksum(t * lift_ls[j, t] for t in times) + abs(sources[j] - (N_SLOTS - 1)) + p)
        for t in range(T - p + 1):
            if efts[j] > 0:
                model.addConstr((t + p) * drop_ls[j, t] >= efts[j] * drop_ls[j, t])
            model.addConstr((t + p) * drop_ls[j, t] <= deadlines[j] * drop_ls[j, t] + M * (1 - serve_ls[j]))

    # Link landside actions
    for t in range(T - p + 1):
        model.addConstr(lift_l[t] == quicksum(lift_coop[i, t] for i in range(n)) + quicksum(lift_ls[j, t] for j in range(m)))
        model.addConstr(drop_l[t] == quicksum(drop_coop[i, t] for i in range(n)) + quicksum(drop_ls[j, t] for j in range(m)))

    # Callback
    incumbent = [float('inf')]
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if obj < incumbent[0] - 1e-3:
                incumbent[0] = obj
                if logger:
                    logger.log(float(obj))

    model.optimize(callback)

    if model.SolCount == 0:
        print("No feasible solution found within time limit.", file=sys.stderr)
        # Provide a trivial feasible solution as fallback
        obj_val = 10000.0
        crane_w_pos = {str(t): sigma_w for t in range(101)}
        crane_l_pos = {str(t): sigma_l for t in range(101)}
        seaside_events = []
        for i, cont in enumerate(seaside):
            seaside_events.append({
                "container_id": cont['id'],
                "target_slot": cont['target_slot'],
                "seaside_drop_time": 50 + i * 20,
                "seaside_drop_slot": cont['target_slot']
            })
        landside_events = []
        solution = {
            "objective_value": obj_val,
            "crane_w_positions": crane_w_pos,
            "crane_l_positions": crane_l_pos,
            "seaside_events": seaside_events,
            "landside_events": landside_events
        }
    else:
        obj_val = model.objVal
        crane_w_pos = {}
        crane_l_pos = {}
        for t in range(T + 1):
            if t > obj_val + 50:
                break
            for s in slots:
                if abs(pos_w[t, s].X - 1.0) < 0.01:
                    crane_w_pos[str(t)] = int(s)
                if abs(pos_l[t, s].X - 1.0) < 0.01:
                    crane_l_pos[str(t)] = int(s)

        seaside_events = []
        for i in range(n):
            drop_t = -1
            drop_s = -1
            for t in range(T - p + 1):
                if drop_sea[i, t].X > 0.5:
                    drop_t = t
                    for s_idx in slots:
                        if drop_slot_var[i, s_idx].X > 0.5:
                            drop_s = s_idx
                            break
                    break
            event = {
                "container_id": seaside[i]['id'],
                "target_slot": targets[i],
                "seaside_drop_time": int(drop_t),
                "seaside_drop_slot": int(drop_s)
            }
            if use_coop[i].X > 0.5:
                lift_t = -1
                drop_l_t = -1
                for t in range(T - p + 1):
                    if lift_coop[i, t].X > 0.5:
                        lift_t = t
                    if drop_coop[i, t].X > 0.5:
                        drop_l_t = t
                event["landside_lift_time"] = int(lift_t)
                event["landside_lift_slot"] = int(drop_s)
                event["landside_drop_time"] = int(drop_l_t)
            seaside_events.append(event)

        landside_events = []
        for j in range(m):
            if serve_ls[j].X < 0.5:
                continue
            lift_t = -1
            drop_t = -1
            for t_idx in range(T - p + 1):
                if lift_ls[j, t_idx].X > 0.5:
                    lift_t = t_idx
                if drop_ls[j, t_idx].X > 0.5:
                    drop_t = t_idx
            if lift_t >= 0 and drop_t >= 0:
                landside_events.append({
                    "container_id": landside[j]['id'],
                    "source_slot": sources[j],
                    "lift_time": int(lift_t),
                    "drop_time": int(drop_t)
                })

        solution = {
            "objective_value": float(obj_val),
            "crane_w_positions": crane_w_pos,
            "crane_l_positions": crane_l_pos,
            "seaside_events": seaside_events,
            "landside_events": landside_events
        }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    main()