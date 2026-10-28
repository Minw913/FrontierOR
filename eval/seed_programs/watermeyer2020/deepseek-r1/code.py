import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        data = json.load(f)
    
    n_real = data['n']
    num_activities = data['num_activities']
    d_bar = data['d_bar']
    activities_data = data['activities']
    resources_data = data['resources']
    num_resources = data['num_resources']
    raw_temporal_constraints = data['temporal_constraints']

    temporal_constraints = []
    for constr in raw_temporal_constraints:
        i_str = str(constr['i'])
        j_str = str(constr['j'])
        temporal_constraints.append({'i': i_str, 'j': j_str, 'delta': constr['delta']})

    activity_ids = [str(i) for i in range(num_activities)]
    resource_ids = list(resources_data.keys())

    prefix = {}
    for k in resource_ids:
        periods = resources_data[k]['periods']
        arr = [0] * (d_bar + 1)
        for p in periods:
            if 1 <= p <= d_bar:
                arr[p] = 1
        pre = [0] * (d_bar + 1)
        for i in range(1, d_bar + 1):
            pre[i] = pre[i - 1] + arr[i]
        prefix[k] = pre

    S0 = {aid: -10**9 for aid in activity_ids}
    S0['0'] = 0

    for _ in range(num_activities - 1):
        updated = False
        for constr in temporal_constraints:
            i = constr['i']
            j = constr['j']
            delta = constr['delta']
            if S0[i] != -10**9:
                new_val = S0[i] + delta
                if new_val > S0[j]:
                    S0[j] = new_val
                    updated = True
        if not updated:
            break

    for aid in activity_ids:
        if S0[aid] < 0:
            S0[aid] = 0

    temporal_feasible = True
    for constr in temporal_constraints:
        i = constr['i']
        j = constr['j']
        delta = constr['delta']
        if S0[j] - S0[i] < delta:
            temporal_feasible = False
            break

    resource_feasible = True
    if temporal_feasible:
        for k in resource_ids:
            capacity = resources_data[k]['capacity']
            total_consumption = 0
            for aid in activity_ids:
                p_i = activities_data[aid]['processing_time']
                demands = activities_data[aid].get('resource_demands', {})
                demand = demands.get(k, 0)
                if demand == 0:
                    continue
                t_start = S0[aid]
                low = max(t_start + 1, 1)
                high = min(t_start + p_i, d_bar)
                if low > high:
                    count = 0
                else:
                    pre_arr = prefix[k]
                    count = pre_arr[high] - pre_arr[low - 1]
                total_consumption += demand * count
                if total_consumption > capacity:
                    resource_feasible = False
                    break
            if not resource_feasible:
                break
    else:
        resource_feasible = False

    initial_resource_feasible = temporal_feasible and resource_feasible

    m = gp.Model()
    m.setParam('OutputFlag', 0)
    m.setParam('TimeLimit', args.time_limit)
    m.setParam('Threads', 1)

    x = {}
    for aid in activity_ids:
        for t in range(0, d_bar + 1):
            x[(aid, t)] = m.addVar(vtype=GRB.BINARY, name=f'x_{aid}_{t}')

    m.addConstr(x[('0', 0)] == 1, "StartAtZero")

    for aid in activity_ids:
        m.addConstr(gp.quicksum(x[(aid, t)] for t in range(0, d_bar + 1)) == 1, f"Act_{aid}")

    for constr in temporal_constraints:
        i = constr['i']
        j = constr['j']
        delta = constr['delta']
        S_i = gp.quicksum(t * x[(i, t)] for t in range(0, d_bar + 1))
        S_j = gp.quicksum(t * x[(j, t)] for t in range(0, d_bar + 1))
        m.addConstr(S_j - S_i >= delta, f"Temp_{i}_{j}")

    for k in resource_ids:
        expr = gp.LinExpr()
        capacity = resources_data[k]['capacity']
        for aid in activity_ids:
            p_i = activities_data[aid]['processing_time']
            demands = activities_data[aid].get('resource_demands', {})
            demand = demands.get(k, 0)
            if demand == 0:
                continue
            for t in range(0, d_bar + 1):
                low = max(t + 1, 1)
                high = min(t + p_i, d_bar)
                if low > high:
                    c_val = 0
                else:
                    pre_arr = prefix[k]
                    c_val = demand * (pre_arr[high] - pre_arr[low - 1])
                if c_val != 0:
                    expr.addTerms(c_val, x[(aid, t)])
        m.addConstr(expr <= capacity, f"Res_{k}")

    project_end_id = str(n_real + 1)
    obj_expr = gp.quicksum(t * x[(project_end_id, t)] for t in range(0, d_bar + 1))
    m.setObjective(obj_expr, GRB.MINIMIZE)

    if initial_resource_feasible:
        for aid in activity_ids:
            t_val = min(max(S0[aid], 0), d_bar)
            for t in range(0, d_bar + 1):
                if t == t_val:
                    x[(aid, t)].Start = 1
                else:
                    x[(aid, t)].Start = 0

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(int(obj_val))

    m.optimize(callback if logger else None)

    if m.SolCount > 0:
        start_times = {}
        for aid in activity_ids:
            for t in range(0, d_bar + 1):
                if x[(aid, t)].X > 0.5:
                    start_times[aid] = t
                    break
        objective_value = int(round(m.ObjVal))
    else:
        start_times = S0
        objective_value = start_times[project_end_id]

    solution = {
        "objective_value": objective_value,
        "start_times": start_times
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == "__main__":
    main()