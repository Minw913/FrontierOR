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

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    num_nodes = instance['num_nodes']
    deadline = instance['deadline']
    processing_times = instance['processing_times']
    temporal_constraints = instance['temporal_constraints']
    resources = instance['resources']

    n = num_nodes
    ES = [-10**9] * n
    LS = [10**9] * n
    ES[0] = 0
    for _ in range(n-1):
        updated = False
        for con in temporal_constraints:
            u = con['from']
            v = con['to']
            delta = con['delta']
            if ES[u] + delta > ES[v]:
                ES[v] = ES[u] + delta
                updated = True
        if not updated:
            break

    LS[n-1] = deadline
    for _ in range(n-1):
        updated = False
        for con in temporal_constraints:
            u = con['from']
            v = con['to']
            delta = con['delta']
            if LS[v] - delta < LS[u]:
                LS[u] = LS[v] - delta
                updated = True
        if not updated:
            break

    infeasible_temporal = False
    for i in range(n):
        if ES[i] > LS[i]:
            infeasible_temporal = True
            break

    if infeasible_temporal:
        solution = {
            "objective_value": ES[n-1],
            "start_times": ES
        }
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f)
        if args.log_path:
            logger = SolutionLogger(args.log_path, sense="minimize")
            logger.log(ES[n-1])
        return

    fixed = [False] * n
    start_times_fixed = [0] * n
    for i in range(n):
        if ES[i] == LS[i]:
            fixed[i] = True
            start_times_fixed[i] = ES[i]

    F_dict = {}
    for resource in resources:
        rid = resource['resource_id']
        time_periods = resource['time_periods']
        F_r = [0] * (deadline + 1)
        in_period = [False] * (deadline + 1)
        for t in time_periods:
            if 1 <= t <= deadline:
                in_period[t] = True
        for x in range(1, deadline + 1):
            F_r[x] = F_r[x-1] + (1 if in_period[x] else 0)
        F_dict[rid] = F_r

    model = gp.Model()
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', args.time_limit)
    y = {}
    for i in range(n):
        if not fixed[i]:
            for s in range(ES[i], LS[i] + 1):
                y[(i, s)] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{s}")

    for i in range(n):
        if not fixed[i]:
            model.addConstr(gp.quicksum(y[(i, s)] for s in range(ES[i], LS[i] + 1)) == 1)

    for con in temporal_constraints:
        u = con['from']
        v = con['to']
        delta = con['delta']
        if fixed[u] and fixed[v]:
            if start_times_fixed[v] - start_times_fixed[u] < delta:
                model.addConstr(0 >= 1, name=f"infeasible_{u}_{v}")
        elif fixed[u]:
            expr_v = gp.quicksum(s * y[(v, s)] for s in range(ES[v], LS[v] + 1))
            model.addConstr(expr_v >= start_times_fixed[u] + delta)
        elif fixed[v]:
            expr_u = gp.quicksum(s * y[(u, s)] for s in range(ES[u], LS[u] + 1))
            model.addConstr(expr_u <= start_times_fixed[v] - delta)
        else:
            expr_v = gp.quicksum(s * y[(v, s)] for s in range(ES[v], LS[v] + 1))
            expr_u = gp.quicksum(s * y[(u, s)] for s in range(ES[u], LS[u] + 1))
            model.addConstr(expr_v - expr_u >= delta)

    for resource in resources:
        rid = resource['resource_id']
        capacity = resource['capacity']
        demands = resource['demands']
        F_r = F_dict[rid]
        constant = 0
        for i in range(n):
            if fixed[i]:
                s0 = start_times_fixed[i]
                p_i = processing_times[i]
                if s0 + p_i <= deadline:
                    usage = F_r[s0 + p_i] - F_r[s0]
                else:
                    usage = F_r[deadline] - F_r[s0]
                constant += demands[i] * usage
        expr = gp.LinExpr()
        for i in range(n):
            if not fixed[i]:
                d_i = demands[i]
                if d_i == 0:
                    continue
                p_i = processing_times[i]
                for s in range(ES[i], LS[i] + 1):
                    if s + p_i <= deadline:
                        usage = F_r[s + p_i] - F_r[s]
                    else:
                        usage = F_r[deadline] - F_r[s]
                    expr.add(d_i * usage * y[(i, s)])
        model.addConstr(constant + expr <= capacity, name=f"res_{rid}")

    if fixed[n-1]:
        dummy = model.addVar(vtype=GRB.CONTINUOUS, name="dummy")
        model.addConstr(dummy == start_times_fixed[n-1])
        model.setObjective(dummy, GRB.MINIMIZE)
    else:
        obj_expr = gp.quicksum(s * y[(n-1, s)] for s in range(ES[n-1], LS[n-1] + 1))
        model.setObjective(obj_expr, GRB.MINIMIZE)

    logger = None
    if args.log_path:
        logger = SolutionLogger(args.log_path, sense="minimize")

    def callback(model_param, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model_param.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(int(obj_val))

    model.optimize(callback)

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT and model.SolCount > 0:
        start_times = [0] * n
        for i in range(n):
            if fixed[i]:
                start_times[i] = start_times_fixed[i]
            else:
                for s in range(ES[i], LS[i] + 1):
                    if model.cbGetSolution(y.get((i, s), 0)) > 0.5:
                        start_times[i] = s
                        break
        objective_value = start_times[n-1]
    else:
        start_times = ES
        objective_value = ES[n-1]

    solution = {
        "objective_value": objective_value,
        "start_times": start_times
    }
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == "__main__":
    main()