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
        data = json.load(f)

    num_items = data['num_items']
    num_periods = data['num_periods']
    demand = data['demand']           # [num_items][num_periods]
    setup_cost = data['setup_cost']   # [num_items]
    holding_cost = data['holding_cost']  # [num_items]
    variable_cost = data['variable_cost']  # [num_items]
    setup_time = data['setup_time']   # [num_items]
    variable_time = data['variable_time']  # [num_items]
    capacity = data['capacity']       # [num_periods]
    initial_inventory_cost = data['initial_inventory_cost']  # [num_items]

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    # Remaining demand from period t to end
    remaining_demand = [[0.0] * num_periods for _ in range(num_items)]
    for i in range(num_items):
        s = 0
        for t in range(num_periods - 1, -1, -1):
            s += demand[i][t]
            remaining_demand[i][t] = float(s)

    model = gp.Model('production_planning')
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', float(args.time_limit))

    # Decision variables
    x = {}   # production quantity
    y = {}   # setup binary
    I = {}   # inventory (I[i,0] = initial, I[i,t+1] = end of period t)

    for i in range(num_items):
        I[i, 0] = model.addVar(lb=0, ub=remaining_demand[i][0] if remaining_demand[i][0] > 0 else 0,
                               obj=initial_inventory_cost[i], name=f'I_{i}_0')
        for t in range(num_periods):
            x[i, t] = model.addVar(lb=0, obj=variable_cost[i], name=f'x_{i}_{t}')
            y[i, t] = model.addVar(vtype=GRB.BINARY, obj=setup_cost[i], name=f'y_{i}_{t}')
            I[i, t + 1] = model.addVar(lb=0, obj=holding_cost[i], name=f'I_{i}_{t+1}')

    model.update()

    # Inventory balance constraints
    for i in range(num_items):
        for t in range(num_periods):
            model.addConstr(I[i, t] + x[i, t] == demand[i][t] + I[i, t + 1],
                            name=f'bal_{i}_{t}')
        # Final inventory must be zero
        model.addConstr(I[i, num_periods] == 0, name=f'final_{i}')

    # Capacity constraints
    for t in range(num_periods):
        model.addConstr(
            gp.quicksum(setup_time[i] * y[i, t] + variable_time[i] * x[i, t]
                        for i in range(num_items)) <= capacity[t],
            name=f'cap_{t}'
        )

    # Setup linking and production upper bound
    for i in range(num_items):
        for t in range(num_periods):
            if remaining_demand[i][t] <= 0:
                ub = 0.0
            else:
                vt = variable_time[i]
                if vt <= 0:
                    ub = remaining_demand[i][t]
                else:
                    time_ub = (capacity[t] - setup_time[i]) / vt
                    ub = min(time_ub, remaining_demand[i][t])
            model.addConstr(x[i, t] <= ub * y[i, t], name=f'link_{i}_{t}')

    # Callback for logging incumbents
    var_list_x = x
    var_list_y = y
    var_list_I = I

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            prod = [[0.0] * num_periods for _ in range(num_items)]
            stp = [[0] * num_periods for _ in range(num_items)]
            inv = [[0.0] * (num_periods + 1) for _ in range(num_items)]
            xsol = model.cbGetSolution([var_list_x[i, t] for i in range(num_items) for t in range(num_periods)])
            ysol = model.cbGetSolution([var_list_y[i, t] for i in range(num_items) for t in range(num_periods)])
            isol = model.cbGetSolution([var_list_I[i, k] for i in range(num_items) for k in range(num_periods + 1)])
            idx = 0
            for i in range(num_items):
                for t in range(num_periods):
                    v = xsol[idx]
                    prod[i][t] = float(v) if v > 1e-6 else 0.0
                    idx += 1
            idx = 0
            for i in range(num_items):
                for t in range(num_periods):
                    v = ysol[idx]
                    stp[i][t] = 1 if v > 0.5 else 0
                    idx += 1
            idx = 0
            for i in range(num_items):
                for k in range(num_periods + 1):
                    v = isol[idx]
                    inv[i][k] = float(v) if v > 1e-6 else 0.0
                    idx += 1
            solution = {
                'objective_value': float(obj_val),
                'production': prod,
                'setup': stp,
                'inventory': inv,
            }
            if logger:
                logger.log_solution(float(obj_val), solution)

    model.optimize(callback)

    status = model.Status
    if status == GRB.INFEASIBLE or status == GRB.INF_OR_UNBD:
        # No feasible solution; write a trivial solution if possible
        # Try to construct a feasible solution manually (produce everything with setups)
        # Fall back to writing zeros (may be infeasible but we must output something)
        solution = {
            'objective_value': 0.0,
            'production': [[0.0] * num_periods for _ in range(num_items)],
            'setup': [[0] * num_periods for _ in range(num_items)],
            'inventory': [[0.0] * (num_periods + 1) for _ in range(num_items)],
        }
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
        return

    # Extract best solution
    if model.SolCount == 0:
        solution = {
            'objective_value': 0.0,
            'production': [[0.0] * num_periods for _ in range(num_items)],
            'setup': [[0] * num_periods for _ in range(num_items)],
            'inventory': [[0.0] * (num_periods + 1) for _ in range(num_items)],
        }
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
        return

    obj_val = model.ObjVal

    production = [[0.0] * num_periods for _ in range(num_items)]
    setup = [[0] * num_periods for _ in range(num_items)]
    inventory = [[0.0] * (num_periods + 1) for _ in range(num_items)]

    for i in range(num_items):
        v = I[i, 0].X
        inventory[i][0] = float(v) if v is not None and v > 1e-6 else 0.0
        for t in range(num_periods):
            vx = x[i, t].X
            vy = y[i, t].X
            vi = I[i, t + 1].X
            production[i][t] = float(vx) if vx is not None and vx > 1e-6 else 0.0
            setup[i][t] = 1 if vy is not None and vy > 0.5 else 0
            inventory[i][t + 1] = float(vi) if vi is not None and vi > 1e-6 else 0.0

    solution = {
        'objective_value': float(obj_val),
        'production': production,
        'setup': setup,
        'inventory': inventory,
    }

    if logger:
        logger.log_solution(float(obj_val), solution)

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == '__main__':
    main()