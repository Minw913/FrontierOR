import os
os.environ['GRB_USE_SYSTEMD'] = '0'
os.environ['GRB_LOGFILE'] = ''
os.environ['GRB_IS_SERVER'] = '1'

import gurobipy as grb
import json
import time
import argparse
from solution_logger import SolutionLogger

def solve_follower_MILP(instance, x0, time_limit=None):
    n1 = instance['n1']
    n2 = instance['n2']
    m2 = instance['m2']
    integer_follower_vars = set(instance['integer_follower_vars'])
    d2 = instance['d2']
    A2 = instance['A2']
    B2 = instance['B2']
    b2 = instance['b2']
    
    model = grb.Model()
    model.setParam('OutputFlag', False)
    model.setParam('Seed', 0)
    model.setParam('Threads', 1)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    if time_limit is not None:
        model.setParam('TimeLimit', time_limit)
    
    y = model.addVars(n2, lb=0.0, name='y')
    for j in range(n2):
        if (j+1) in integer_follower_vars:
            y[j].vtype = grb.GRB.INTEGER
    
    for i in range(m2):
        expr = grb.quicksum(B2[i][j] * y[j] for j in range(n2))
        constant = sum(A2[i][k] * x0[k] for k in range(n1))
        model.addConstr(expr <= b2[i] - constant)
    
    model.setObjective(grb.quicksum(d2[j] * y[j] for j in range(n2)), grb.GRB.MAXIMIZE)
    model.optimize()
    
    if model.status == grb.GRB.OPTIMAL:
        return [y[j].X for j in range(n2)], model.ObjVal, model.status
    return None, None, model.status

def solve_feasibility_problem(instance, x0, follower_opt, time_limit=None):
    n1 = instance['n1']
    n2 = instance['n2']
    m1 = instance['m1']
    m2 = instance['m2']
    integer_follower_vars = set(instance['integer_follower_vars'])
    d2 = instance['d2']
    A1 = instance['A1']
    B1 = instance['B1']
    b1 = instance['b1']
    A2 = instance['A2']
    B2 = instance['B2']
    b2 = instance['b2']
    
    model = grb.Model()
    model.setParam('OutputFlag', False)
    model.setParam('Seed', 0)
    model.setParam('Threads', 1)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    if time_limit is not None:
        model.setParam('TimeLimit', time_limit)
    
    y = model.addVars(n2, lb=0.0, name='y')
    for j in range(n2):
        if (j+1) in integer_follower_vars:
            y[j].vtype = grb.GRB.INTEGER
    
    for i in range(m2):
        expr = grb.quicksum(B2[i][j] * y[j] for j in range(n2))
        constant = sum(A2[i][k] * x0[k] for k in range(n1))
        model.addConstr(expr <= b2[i] - constant)
    
    for i in range(m1):
        expr = grb.quicksum(B1[i][j] * y[j] for j in range(n2))
        constant = sum(A1[i][k] * x0[k] for k in range(n1))
        model.addConstr(expr <= b1[i] - constant)
    
    model.addConstr(grb.quicksum(d2[j] * y[j] for j in range(n2)) == follower_opt)
    model.optimize()
    
    if model.status == grb.GRB.OPTIMAL:
        return [y[j].X for j in range(n2)], model.status
    return None, model.status

def solve_HPR(instance, time_limit=None):
    n1 = instance['n1']
    n2 = instance['n2']
    m1 = instance['m1']
    m2 = instance['m2']
    integer_follower_vars = set(instance['integer_follower_vars'])
    c = instance['c']
    A1 = instance['A1']
    B1 = instance['B1']
    b1 = instance['b1']
    A2 = instance['A2']
    B2 = instance['B2']
    b2 = instance['b2']
    
    model = grb.Model()
    model.setParam('OutputFlag', False)
    model.setParam('Seed', 0)
    model.setParam('Threads', 1)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    if time_limit is not None:
        model.setParam('TimeLimit', time_limit)
    
    x = model.addVars(n1, vtype=grb.GRB.INTEGER, lb=0.0, name='x')
    y = model.addVars(n2, lb=0.0, name='y')
    for j in range(n2):
        if (j+1) in integer_follower_vars:
            y[j].vtype = grb.GRB.INTEGER
    
    for i in range(m1):
        expr_x = grb.quicksum(A1[i][k] * x[k] for k in range(n1))
        expr_y = grb.quicksum(B1[i][j] * y[j] for j in range(n2))
        model.addConstr(expr_x + expr_y <= b1[i])
    
    for i in range(m2):
        expr_x = grb.quicksum(A2[i][k] * x[k] for k in range(n1))
        expr_y = grb.quicksum(B2[i][j] * y[j] for j in range(n2))
        model.addConstr(expr_x + expr_y <= b2[i])
    
    model.setObjective(grb.quicksum(c[k] * x[k] for k in range(n1)), grb.GRB.MAXIMIZE)
    model.optimize()
    
    if model.status == grb.GRB.OPTIMAL:
        return [x[k].X for k in range(n1)], [y[j].X for j in range(n2)], model.status
    return None, None, model.status

def check_upper_level(instance, x, y):
    n1 = instance['n1']
    m1 = instance['m1']
    A1 = instance['A1']
    B1 = instance['B1']
    b1 = instance['b1']
    
    for i in range(m1):
        lhs = sum(A1[i][k] * x[k] for k in range(n1)) + sum(B1[i][j] * y[j] for j in range(len(y)))
        if lhs > b1[i] + 1e-5:
            return False
    return True

def compute_leader_obj(instance, x):
    c = instance['c']
    return sum(c[k] * x[k] for k in range(len(x)))

def output_solution(solution_dict, solution_path):
    with open(solution_path, 'w') as f:
        json.dump(solution_dict, f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, required=False, default=None)
    args = parser.parse_args()
    
    try:
        logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None
        
        with open(args.instance_path) as f:
            instance = json.load(f)
        
        start_time = time.time()
        best_solution = None
        best_obj = -grb.GRB.INFINITY
        
        # Strategy 1: Zero leader solution
        time_left = max(0, args.time_limit - (time.time() - start_time))
        if time_left > 0:
            x0_zero = [0] * instance['n1']
            y1, v1, status = solve_follower_MILP(instance, x0_zero, time_left)
            if status == grb.GRB.OPTIMAL:
                if check_upper_level(instance, x0_zero, y1):
                    leader_obj = compute_leader_obj(instance, x0_zero)
                    if leader_obj > best_obj:
                        best_obj = leader_obj
                        best_solution = {
                            'x': x0_zero, 
                            'y': y1,
                            'follower_objective_value': v1,
                            'objective_value': leader_obj
                        }
                        if logger:
                            logger.log_solution(leader_obj, best_solution)
                else:
                    time_left2 = max(0, args.time_limit - (time.time() - start_time))
                    if time_left2 > 0:
                        y2, status2 = solve_feasibility_problem(instance, x0_zero, v1, time_left2)
                        if status2 == grb.GRB.OPTIMAL:
                            leader_obj = compute_leader_obj(instance, x0_zero)
                            if leader_obj > best_obj:
                                best_obj = leader_obj
                                best_solution = {
                                    'x': x0_zero, 
                                    'y': y2,
                                    'follower_objective_value': v1,
                                    'objective_value': leader_obj
                                }
                                if logger:
                                    logger.log_solution(leader_obj, best_solution)
        
        # Strategy 2: High Point Relaxation (HPR)
        time_left = max(0, args.time_limit - (time.time() - start_time))
        if time_left > 0:
            x0, y0, status_hpr = solve_HPR(instance, time_left)
            if status_hpr == grb.GRB.OPTIMAL:
                time_left2 = max(0, args.time_limit - (time.time() - start_time))
                if time_left2 > 0:
                    y1, v1, status_f = solve_follower_MILP(instance, x0, time_left2)
                    if status_f == grb.GRB.OPTIMAL:
                        if check_upper_level(instance, x0, y1):
                            leader_obj = compute_leader_obj(instance, x0)
                            if leader_obj > best_obj:
                                best_obj = leader_obj
                                best_solution = {
                                    'x': x0, 
                                    'y': y1,
                                    'follower_objective_value': v1,
                                    'objective_value': leader_obj
                                }
                                if logger:
                                    logger.log_solution(leader_obj, best_solution)
                        else:
                            time_left3 = max(0, args.time_limit - (time.time() - start_time))
                            if time_left3 > 0:
                                y2, status_f2 = solve_feasibility_problem(instance, x0, v1, time_left3)
                                if status_f2 == grb.GRB.OPTIMAL:
                                    leader_obj = compute_leader_obj(instance, x0)
                                    if leader_obj > best_obj:
                                        best_obj = leader_obj
                                        best_solution = {
                                            'x': x0, 
                                            'y': y2,
                                            'follower_objective_value': v1,
                                            'objective_value': leader_obj
                                        }
                                        if logger:
                                            logger.log_solution(leader_obj, best_solution)
        
        # Output best found solution or fallback
        if best_solution is not None:
            output_solution(best_solution, args.solution_path)
        else:
            fallback_solution = {
                'x': [0] * instance['n1'],
                'y': [0.0] * instance['n2'],
                'follower_objective_value': 0.0,
                'objective_value': 0.0
            }
            output_solution(fallback_solution, args.solution_path)
            if logger:
                logger.log_solution(0.0, fallback_solution)
                
    except Exception as e:
        # Fallback solution in case of any exception
        n1 = instance.get('n1', 0)
        n2 = instance.get('n2', 0)
        fallback_solution = {
            'x': [0] * n1,
            'y': [0.0] * n2,
            'follower_objective_value': 0.0,
            'objective_value': 0.0
        }
        output_solution(fallback_solution, args.solution_path)
        if args.log_path:
            logger = SolutionLogger(args.log_path, sense="maximize")
            logger.log_solution(0.0, fallback_solution)

if __name__ == '__main__':
    main()