import json
import argparse
import time
import sys
import random
import math
from itertools import permutations

def solve_mip(instance, time_limit):
    """Solve using MIP formulation with Gurobi."""
    import gurobipy as gp
    from gurobipy import GRB
    
    params = instance['parameters']
    n = params['num_requests_n']
    s = params['num_stacks_s']
    l = params['stack_height_l']
    
    pickup_dist = instance['pickup_region']['distance_matrix']
    delivery_dist = instance['delivery_region']['distance_matrix']
    pickup_depot = instance['pickup_region']['depot_index']
    delivery_depot = instance['delivery_region']['depot_index']
    
    num_pv = params['num_pickup_vertices']
    num_dv = params['num_delivery_vertices']
    
    # Map requests to pickup/delivery vertices
    requests = instance['requests']
    pickup_vertices = [r['pickup_vertex'] for r in requests]
    delivery_vertices = [r['delivery_vertex'] for r in requests]
    
    model = gp.Model("pickup_delivery")
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', time_limit)
    
    # Decision variables
    # x[i,j] = 1 if arc (i,j) is in pickup tour
    x = {}
    for i in range(num_pv):
        for j in range(num_pv):
            if i != j:
                x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
    
    # y[i,j] = 1 if arc (i,j) is in delivery tour
    y = {}
    for i in range(num_dv):
        for j in range(num_dv):
            if i != j:
                y[i, j] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{j}")
    
    # z[i,k] = 1 if item i is assigned to stack k
    z = {}
    for i in range(n):
        for k in range(s):
            z[i, k] = model.addVar(vtype=GRB.BINARY, name=f"z_{i}_{k}")
    
    # u[i] = position of pickup vertex i in pickup tour (MTZ)
    u = {}
    for i in range(num_pv):
        if i != pickup_depot:
            u[i] = model.addVar(lb=1, ub=n, vtype=GRB.CONTINUOUS, name=f"u_{i}")
    
    # v[i] = position of delivery vertex i in delivery tour (MTZ)
    v = {}
    for i in range(num_dv):
        if i != delivery_depot:
            v[i] = model.addVar(lb=1, ub=n, vtype=GRB.CONTINUOUS, name=f"v_{i}")
    
    # a[i,j] = 1 if item i is picked up before item j
    a = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                a[i, j] = model.addVar(vtype=GRB.BINARY, name=f"a_{i}_{j}")
    
    model.update()
    
    # Objective
    obj = gp.LinExpr()
    for i in range(num_pv):
        for j in range(num_pv):
            if i != j:
                obj += pickup_dist[i][j] * x[i, j]
    for i in range(num_dv):
        for j in range(num_dv):
            if i != j:
                obj += delivery_dist[i][j] * y[i, j]
    model.setObjective(obj, GRB.MINIMIZE)
    
    # Degree constraints for pickup tour
    for i in range(num_pv):
        model.addConstr(gp.quicksum(x[i, j] for j in range(num_pv) if j != i) == 1, name=f"pickup_out_{i}")
        model.addConstr(gp.quicksum(x[j, i] for j in range(num_pv) if j != i) == 1, name=f"pickup_in_{i}")
    
    # Degree constraints for delivery tour
    for i in range(num_dv):
        model.addConstr(gp.quicksum(y[i, j] for j in range(num_dv) if j != i) == 1, name=f"delivery_out_{i}")
        model.addConstr(gp.quicksum(y[j, i] for j in range(num_dv) if j != i) == 1, name=f"delivery_in_{i}")
    
    # MTZ subtour elimination for pickup
    for i in range(num_pv):
        if i == pickup_depot:
            continue
        for j in range(num_pv):
            if j == pickup_depot or j == i:
                continue
            model.addConstr(u[i] - u[j] + n * x[i, j] <= n - 1, name=f"mtz_p_{i}_{j}")
    
    # MTZ subtour elimination for delivery
    for i in range(num_dv):
        if i == delivery_depot:
            continue
        for j in range(num_dv):
            if j == delivery_depot or j == i:
                continue
            model.addConstr(v[i] - v[j] + n * y[i, j] <= n - 1, name=f"mtz_d_{i}_{j}")
    
    # Stack assignment constraints
    for i in range(n):
        model.addConstr(gp.quicksum(z[i, k] for k in range(s)) == 1, name=f"assign_{i}")
    
    for k in range(s):
        model.addConstr(gp.quicksum(z[i, k] for i in range(n)) <= l, name=f"capacity_{k}")
    
    # Link a_ij to pickup positions
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            pi = pickup_vertices[i]
            pj = pickup_vertices[j]
            # a[i,j] = 1 => u[pi] < u[pj]
            model.addConstr(u[pi] - u[pj] + 1 <= n * (1 - a[i, j]), name=f"a_link1_{i}_{j}")
            # a[i,j] = 0 => u[pj] < u[pi]
            model.addConstr(u[pj] - u[pi] + 1 <= n * a[i, j], name=f"a_link2_{i}_{j}")
    
    # LIFO constraints
    for i in range(n):
        for j in range(i+1, n):
            di = delivery_vertices[i]
            dj = delivery_vertices[j]
            for k in range(s):
                # If a[i,j]=1 (i before j in pickup) and both in stack k:
                # then j before i in delivery: v[dj] < v[di]
                # v[di] - v[dj] >= 1
                # v[dj] - v[di] + 1 <= 0
                model.addConstr(
                    v[dj] - v[di] + 1 <= n * (1 - a[i, j]) + n * (1 - z[i, k]) + n * (1 - z[j, k]),
                    name=f"lifo1_{i}_{j}_{k}"
                )
                # If a[i,j]=0 (j before i in pickup) and both in stack k:
                # then i before j in delivery: v[di] < v[dj]
                # v[di] - v[dj] + 1 <= 0
                model.addConstr(
                    v[di] - v[dj] + 1 <= n * a[i, j] + n * (1 - z[i, k]) + n * (1 - z[j, k]),
                    name=f"lifo2_{i}_{j}_{k}"
                )
    
    model.optimize()
    
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        if model.SolCount > 0:
            # Extract solution
            pickup_tour = extract_pickup_tour(model, x, num_pv, pickup_depot)
            delivery_tour = extract_delivery_tour(model, y, num_dv, delivery_depot)
            
            obj_val = compute_objective(pickup_tour, delivery_tour, pickup_dist, delivery_dist)
            return pickup_tour, delivery_tour, obj_val
    
    return None, None, float('inf')


def extract_pickup_tour(model, x, num_pv, depot):
    """Extract the pickup tour from the MIP solution."""
    # Build adjacency from x variables
    next_vertex = {}
    for i in range(num_pv):
        for j in range(num_pv):
            if i != j and x[i, j].X > 0.5:
                next_vertex[i] = j
                break
    
    tour = [depot]
    current = depot
    for _ in range(num_pv - 1):
        current = next_vertex[current]
        tour.append(current)
    tour.append(depot)
    return tour


def extract_delivery_tour(model, y, num_dv, depot):
    """Extract the delivery tour from the MIP solution."""
    next_vertex = {}
    for i in range(num_dv):
        for j in range(num_dv):
            if i != j and y[i, j].X > 0.5:
                next_vertex[i] = j
                break
    
    tour = [depot]
    current = depot
    for _ in range(num_dv - 1):
        current = next_vertex[current]
        tour.append(current)
    tour.append(depot)
    return tour


def compute_objective(pickup_tour, delivery_tour, pickup_dist, delivery_dist):
    """Compute total routing cost."""
    cost = 0
    for i in range(len(pickup_tour) - 1):
        cost += pickup_dist[pickup_tour[i]][pickup_tour[i+1]]
    for i in range(len(delivery_tour) - 1):
        cost += delivery_dist[delivery_tour[i]][delivery_tour[i+1]]
    return cost