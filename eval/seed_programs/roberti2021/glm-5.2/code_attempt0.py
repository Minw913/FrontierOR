import json
import argparse
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
    
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    with open(args.instance_path) as f:
        instance = json.load(f)
    
    N = instance['num_customers']
    num_nodes = instance['num_nodes']  # N+1 (depot + customers)
    
    truck_tt = instance['truck_travel_times']
    drone_tt = instance['drone_travel_times']
    
    # Nodes: 0 (depot origin), 1..N (customers), N+1 (depot destination)
    D = N + 1  # depot destination index
    
    # Build arc sets
    # Truck arcs: (0,j) for j in 1..N, (i,j) for i,j in 1..N, i!=j, (i,D) for i in 1..N
    # Same for drone arcs
    # No arc (0, D)
    
    all_nodes = list(range(0, N+2))  # 0..N+1
    customers = list(range(1, N+1))
    
    # Arcs
    arcs = []
    for j in customers:
        arcs.append((0, j))
    for i in customers:
        for j in customers:
            if i != j:
                arcs.append((i, j))
    for i in customers:
        arcs.append((i, D))
    
    # Get travel time for an arc
    def get_truck_time(i, j):
        if j == D:
            return truck_tt[i][0]
        if i == 0 and j == 0:
            return 0
        return truck_tt[i][j]
    
    def get_drone_time(i, j):
        if j == D:
            return drone_tt[i][0]
        return drone_tt[i][j]
    
    # Big M
    max_tt = max(max(row) for row in truck_tt)
    max_dt = max(max(row) for row in drone_tt)
    M = (N + 1) * max(max_tt, max_dt) + 1
    
    model = gp.Model()
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', args.time_limit)
    
    # Variables
    x = {}  # truck arcs
    y = {}  # drone arcs
    for (i, j) in arcs:
        x[i, j] = model.addVar(vtype=GRB.BINARY, name=f'x_{i}_{j}')
        y[i, j] = model.addVar(vtype=GRB.BINARY, name=f'y_{i}_{j}')
    
    t = {}
    for n in all_nodes:
        t[n] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f't_{n}')
    
    # Combined status
    c = {}
    for i in customers:
        c[i] = model.addVar(vtype=GRB.BINARY, name=f'c_{i}')
    
    model.setObjective(t[D], GRB.MINIMIZE)
    
    # Truck flow
    # Out of depot
    model.addConstr(gp.quicksum(x[0, j] for j in customers) == 1)
    # Into depot destination
    model.addConstr(gp.quicksum(x[i, D] for i in customers) == 1)
    
    for i in customers:
        # Flow conservation
        in_arcs = [j for j in customers if j != i] + [0]  # arcs into i: from other customers and from depot
        out_arcs = [j for j in customers if j != i] + [D]  # arcs out of i: to other customers and to depot dest
        model.addConstr(gp.quicksum(x[j, i] for j in in_arcs) == gp.quicksum(x[i, j] for j in out_arcs))
    
    # Drone flow - same
    model.addConstr(gp.quicksum(y[0, j] for j in customers) == 1)
    model.addConstr(gp.quicksum(y[i, D] for i in customers) == 1)
    
    for i in customers:
        in_arcs = [j for j in customers if j != i] + [0]
        out_arcs = [j for j in customers if j != i] + [D]
        model.addConstr(gp.quicksum(y[j, i] for j in in_arcs) == gp.quicksum(y[i, j] for j in out_arcs))
    
    # Customer service: each customer visited by at least one vehicle
    for i in customers:
        truck_visits_i = gp.quicksum(x[i, j] for j in customers if j != i) + x[i, D]
        drone_visits_i = gp.quicksum(y[i, j] for j in customers if j != i) + y[i, D]
        # out-degree = 1 means visited
        # Each customer served exactly once
        model.addConstr(truck_visits_i + drone_visits_i - c[i] == 1)
        model.addConstr(c[i] <= truck_visits_i)
        model.addConstr(c[i] <= drone_visits_i)
        model.addConstr(c[i] >= truck_visits_i + drone_visits_i - 1)
    
    # Timing constraints
    for (i, j) in arcs:
        model.addConstr(t[j] >= t[i] + get_truck_time(i, j) - M * (1 - x[i, j]))
        model.addConstr(t[j] >= t[i] + get_drone_time(i, j) - M * (1 - y[i, j]))
    
    t[0] = 0  # Fix via constraint
    model.addConstr(t[0] == 0)
    
    # Drone pairing constraint
    for i in range(1, N+1):
        for j in range(i+1, N+1):
            # y[i,j] + y[j,i] <= c[i] + c[j]
            if (i, j) in y and (j, i) in y:
                model.addConstr(y[i, j] + y[j, i] <= c[i] + c[j])
            elif (i, j) in y:
                model.addConstr(y[i, j] <= c[i] + c[j])
            elif (j, i) in y:
                model.addConstr(y[j, i] <= c[i] + c[j])
    
    # Drone cannot do single-customer sortie
    for i in customers:
        model.addConstr(y[0, i] + y[i, D] <= 1)
    
    model.optimize()
    
    # Extract solution
    # ...