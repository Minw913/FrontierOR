import argparse
import json
import numpy as np
from gurobipy import *
import math
import time

def solve_defense_interception(instance_data, time_limit):
    # Extract parameters
    N_D = instance_data["N_D"]
    N_A = instance_data["N_A"]
    epsilon = instance_data["epsilon"]
    M_u = instance_data["M_u"]
    M_I = instance_data["M_I"]
    M_dz = instance_data["M_dz"]
    N_u = instance_data["N_u"]
    N_a = instance_data["N_a"]
    R_dz = instance_data["R_dz"]
    R_I = instance_data["R_I"]
    T_a = instance_data["T_a"]
    T_u = instance_data["T_u"]
    
    # Defender initial states
    def_initial_pos_vel = []
    for d in instance_data["defenders"]:
        def_initial_pos_vel.append([d["x_s"], d["y_s"], d["xdot_s"], d["ydot_s"]])
        
    # Attacker initial states
    att_initial_pos_vel = []
    for a in instance_data["attackers"]:
        att_initial_pos_vel.append([a["p_s"], a["q_s"], a["v_p"], a["v_q"]])

    # Compute rotation angles for polygon constraints
    theta_u = [2 * math.pi * i / M_u for i in range(M_u)]
    A_u = [[math.cos(t), math.sin(t)] for t in theta_u]
    b_u = [math.cos(math.pi / M_u)] * M_u  # Halfspace RHS for control polygon
    
    theta_dz = [2 * math.pi * i / M_dz for i in range(M_dz)]
    A_dz = [[math.cos(t), math.sin(t)] for t in theta_dz]
    b_dz = [R_dz] * M_dz  # Halfspace RHS for defense zone
    
    theta_I = [2 * math.pi * i / M_I for i in range(M_I)]
    A_I = [[math.cos(t), math.sin(t)] for t in theta_I]
    b_I = [R_I] * M_I  # Halfspace RHS for intercept zone

    # Create model
    model = Model("DefenseInterception")
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', time_limit)

    # Variables for defender positions, velocities, controls
    x = {}  # x positions at control steps
    y = {}  # y positions at control steps
    xdot = {}  # x velocities at control steps
    ydot = {}  # y velocities at control steps
    ux = {}  # x control inputs
    uy = {}  # y control inputs
    
    for d in range(N_D):
        for k in range(N_u + 1):
            x[d, k] = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"x_{d}_{k}")
            y[d, k] = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"y_{d}_{k}")
            xdot[d, k] = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"xdot_{d}_{k}")
            ydot[d, k] = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"ydot_{d}_{k}")
        
        for k in range(N_u):
            ux[d, k] = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"ux_{d}_{k}")
            uy[d, k] = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"uy_{d}_{k}")

    # Variables for attacker positions, status flags
    p = {}  # x positions at sample steps
    q = {}  # y positions at sample steps
    a = {}  # attack status (1 = active, 0 = inactive)
    gamma = {}  # inside defense zone flag (1 = inside, 0 = outside)
    delta = {}  # intercept flag (1 = intercepted by defender, 0 = not)
    
    for a_idx in range(N_A):
        for k in range(N_a + 1):
            p[a_idx, k] = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"p_{a_idx}_{k}")
            q[a_idx, k] = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"q_{a_idx}_{k}")
            a[a_idx, k] = model.addVar(vtype=GRB.BINARY, name=f"a_{a_idx}_{k}")
            gamma[a_idx, k] = model.addVar(vtype=GRB.BINARY, name=f"gamma_{a_idx}_{k}")
            
            for d in range(N_D):
                delta[a_idx, d, k] = model.addVar(vtype=GRB.BINARY, name=f"delta_{a_idx}_{d}_{k}")

    # Control effort variables for L1 norm
    abs_ux = {}
    abs_uy = {}
    for d in range(N_D):
        for k in range(N_u):
            abs_ux[d, k] = model.addVar(lb=0, ub=GRB.INFINITY, name=f"abs_ux_{d}_{k}")
            abs_uy[d, k] = model.addVar(lb=0, ub=GRB.INFINITY, name=f"abs_uy_{d}_{k}")

    model.update()

    # Dynamics constraints for defenders
    alpha = math.exp(-T_u)
    beta = 1 - math.exp(-T_u)
    gamma_dyn = T_u - 1 + math.exp(-T_u)
    
    for d in range(N_D):
        # Initial conditions
        model.addConstr(x[d, 0] == def_initial_pos_vel[d][0])
        model.addConstr(y[d, 0] == def_initial_pos_vel[d][1])
        model.addConstr(xdot[d, 0] == def_initial_pos_vel[d][2])
        model.addConstr(ydot[d, 0] == def_initial_pos_vel[d][3])
        
        # Dynamics
        for k in range(N_u):
            model.addConstr(xdot[d, k+1] == alpha * xdot[d, k] + beta * ux[d, k])
            model.addConstr(ydot[d, k+1] == alpha * ydot[d, k] + beta * uy[d, k])
            model.addConstr(x[d, k+1] == x[d, k] + gamma_dyn * xdot[d, k] + beta * ux[d, k])
            model.addConstr(y[d, k+1] == y[d, k] + gamma_dyn * ydot[d, k] + beta * uy[d, k])

    # Dynamics for attackers (constant velocity while active)
    for a_idx in range(N_A):
        model.addConstr(p[a_idx, 0] == att_initial_pos_vel[a_idx][0])
        model.addConstr(q[a_idx, 0] == att_initial_pos_vel[a_idx][1])
        
        for k in range(N_a):
            # If attacker is still active, move with constant velocity
            model.addConstr(p[a_idx, k+1] == p[a_idx, k] + T_a * att_initial_pos_vel[a_idx][2])
            model.addConstr(q[a_idx, k+1] == q[a_idx, k] + T_a * att_initial_pos_vel[a_idx][3])
            
            # Update attack status: becomes inactive if intercepted or in defense zone
            # Use big-M constraints to model logic
            M_big = 999999  # Large enough value
            
            # If gamma[a_idx, k+1] = 1, then attacker is in defense zone at k+1
            # If delta[a_idx, d, k+1] = 1 for any d, then attacker is intercepted at k+1
            # So a[a_idx, k+1] = 0 if gamma[a_idx, k+1] = 1 OR any delta[a_idx, d, k+1] = 1
            intercept_sum = quicksum(delta[a_idx, d, k+1] for d in range(N_D))
            model.addConstr(a[a_idx, k+1] >= a[a_idx, k] - gamma[a_idx, k+1])
            model.addConstr(a[a_idx, k+1] >= a[a_idx, k] - intercept_sum)
            # Ensure a doesn't become 1 again after being 0
            model.addConstr(a[a_idx, k+1] <= a[a_idx, k])

    # Defense zone membership constraints
    for a_idx in range(N_A):
        for k in range(N_a + 1):
            # gamma[a_idx, k] = 1 if attacker is in defense zone at time k
            for i in range(M_dz):
                # If gamma[a_idx, k] = 1, then A_dz[i][0]*p[a_idx,k] + A_dz[i][1]*q[a_idx,k] <= b_dz[i]
                # This is equivalent to: A_dz[i][0]*p[a_idx,k] + A_dz[i][1]*q[a_idx,k] <= b_dz[i] + M*(1-gamma[a_idx,k])
                model.addConstr(
                    A_dz[i][0] * p[a_idx, k] + A_dz[i][1] * q[a_idx, k] <= b_dz[i] + M_big * (1 - gamma[a_idx, k])
                )
    
    # Intercept constraints using linear interpolation of defender state
    for a_idx in range(N_A):
        for k in range(N_a + 1):
            # For each defender, check if attacker is within intercept range
            for d in range(N_D):
                # We need to interpolate defender position at attacker sample time
                # Since N_a might not be a multiple of N_u, we find the relevant control periods
                # The relationship between sample steps and control steps:
                # Sample step k corresponds to time k*T_a
                # Control step j corresponds to time j*T_u
                # Find which control periods the sample falls into
                
                # To simplify, we'll approximate defender position at sample time k using closest control steps
                # For now, we'll use the control step just before the sample time
                # More precisely: if k*T_a falls between j*T_u and (j+1)*T_u, 
                # we interpolate between states at j and j+1
                
                # Calculate corresponding control step index
                # t_sample = k * T_a
                # j_control = floor(t_sample / T_u)
                # frac = (t_sample % T_u) / T_u
                # Then interpolate between j and j+1
                
                for i in range(M_I):
                    # Add constraints to model intercept condition
                    # For intercept: |A_I[i] * (attacker_pos - defender_pos)| <= b_I[i]
                    
                    # We'll use big-M to model this: if delta[a_idx, d, k] = 1, then attacker is in intercept zone of defender
                    
                    # First compute interpolated defender position at time k*T_a
                    # Find which control period this sample falls into
                    t_sample = k * T_a
                    j_control = min(int(t_sample // T_u), N_u - 1)  # Clamp to valid range
                    if j_control < 0:
                        j_control = 0
                    elif j_control >= N_u:
                        j_control = N_u - 1
                        
                    if j_control == N_u - 1:
                        # If at last control step, use N_u and N_u+1 is not defined, so use just the end state
                        lambda_t = (t_sample - j_control * T_u) / T_u
                        if lambda_t > 1.0:
                            lambda_t = 1.0  # Clamp to prevent numerical issues
                            
                        # Interpolated defender position
                        def_x_interp = x[d, j_control] + lambda_t * (x[d, j_control + 1] - x[d, j_control])
                        def_y_interp = y[d, j_control] + lambda_t * (y[d, j_control + 1] - y[d, j_control])
                        
                        # For simplicity in this implementation, we'll check intercept at control points
                        # This is an approximation - a more precise version would check at all sample points
                    else:
                        # Use the current control step position as an approximation
                        def_x_interp = x[d, j_control]
                        def_y_interp = y[d, j_control]
                    
                    # If delta[a_idx, d, k] = 1, then attacker is within intercept zone of defender
                    # A_I[i][0]*(p[a_idx, k] - def_x_interp) + A_I[i][1]*(q[a_idx, k] - def_y_interp) <= b_I[i]
                    # Which is: A_I[i][0]*p[a_idx, k] + A_I[i][1]*q[a_idx, k] - A_I[i][0]*def_x_interp - A_I[i][1]*def_y_interp <= b_I[i]
                    # Using big-M: A_I[i][0]*p[a_idx, k] + A_I[i][1]*q[a_idx, k] - A_I[i][0]*def_x_interp - A_I[i][1]*def_y_interp <= b_I[i] + M*(1-delta[a_idx, d, k])
                    
                    # Linearize: we need terms like A_I[i][0]*def_x_interp which involves products of variables
                    # For simplicity, we'll use the position at the control step closest to the sample time
                    # We'll create helper variables to linearize this
                    
                    # For now, let's implement a simplified version checking at control steps
                    # This is an approximation but makes the model linear
                    # Check intercept at control steps by creating helper variables
                    for ctrl_step in range(N_u + 1):
                        # For each control step, check if sample time is close to it
                        t_ctrl = ctrl_step * T_u
                        if abs(k * T_a - t_ctrl) < 1e-6 or (ctrl_step < N_u and k * T_a >= ctrl_step * T_u and k * T_a < (ctrl_step + 1) * T_u):
                            # If the sample time k*T_a falls in the period of control step ctrl_step,
                            # we'll check intercept using defender state at ctrl_step
                            # This is a simplification - in reality we'd interpolate
                            
                            # Add constraints for intercept using defender state at control step
                            model.addConstr(
                                A_I[i][0] * (p[a_idx, k] - x[d, ctrl_step]) + A_I[i][1] * (q[a_idx, k] - y[d, ctrl_step]) <= 
                                b_I[i] + M_big * (1 - delta[a_idx, d, k])
                            )

    # Control input constraints
    for d in range(N_D):
        for k in range(N_u):
            for i in range(M_u):
                model.addConstr(A_u[i][0] * ux[d, k] + A_u[i][1] * uy[d, k] <= b_u[i])

    # Absolute value constraints for control effort
    for d in range(N_D):
        for k in range(N_u):
            model.addConstr(abs_ux[d, k] >= ux[d, k])
            model.addConstr(abs_ux[d, k] >= -ux[d, k])
            model.addConstr(abs_uy[d, k] >= uy[d, k])
            model.addConstr(abs_uy[d, k] >= -uy[d, k])

    # Obstacle avoidance (defenders must stay outside at least one side of obstacle)
    # Obstacle is a square centered at origin with apothem R_dz
    # Sides of square: x <= -R_dz, x >= R_dz, y <= -R_dz, y >= R_dz
    # So: x >= -R_dz-eps OR x <= R_dz-eps OR y >= -R_dz-eps OR y <= R_dz-eps
    # Use binary variables to model disjunction
    
    z_obst = {}  # Binary variables for obstacle avoidance
    for d in range(N_D):
        for k in range(1, N_u + 1):  # Only at positive control-step boundaries
            for side in range(4):  # 4 sides of square
                z_obst[d, k, side] = model.addVar(vtype=GRB.BINARY, name=f"z_obst_{d}_{k}_{side}")
    
    model.update()
    
    for d in range(N_D):
        for k in range(1, N_u + 1):
            # At least one side must be satisfied (outside of it)
            model.addConstr(quicksum(z_obst[d, k, side] for side in range(4)) >= 1)
            
            # If z_obst[d, k, 0] = 1 -> x[d, k] >= -R_dz + eps (not in left side)
            # If z_obst[d, k, 1] = 1 -> x[d, k] <= R_dz - eps (not in right side)  
            # If z_obst[d, k, 2] = 1 -> y[d, k] >= -R_dz + eps (not in bottom side)
            # If z_obst[d, k, 3] = 1 -> y[d, k] <= R_dz - eps (not in top side)
            eps_safe = 0.01  # Small margin
            model.addConstr(x[d, k] >= -R_dz + eps_safe - M_big * (1 - z_obst[d, k, 0]))
            model.addConstr(x[d, k] <= R_dz - eps_safe + M_big * (1 - z_obst[d, k, 1]))
            model.addConstr(y[d, k] >= -R_dz + eps_safe - M_big * (1 - z_obst[d, k, 2]))
            model.addConstr(y[d, k] <= R_dz - eps_safe + M_big * (1 - z_obst[d, k, 3]))

    # Objective function
    obj_terms = []
    # Sum of gamma (time steps attackers are in defense zone)
    for a_idx in range(N_A):
        for k in range(N_a + 1):
            obj_terms.append(gamma[a_idx, k])
    
    # Plus epsilon times control effort
    if epsilon > 0:
        for d in range(N_D):
            for k in range(N_u):
                obj_terms.append(epsilon * abs_ux[d, k])
                obj_terms.append(epsilon * abs_uy[d, k])
    
    model.setObjective(quicksum(obj_terms), GRB.MINIMIZE)

    # Optimize
    start_time = time.time()
    model.optimize()
    end_time = time.time()
    
    print(f"Optimization completed in {end_time - start_time:.2f}s")
    print(f"Status: {model.status}")
    print(f"Objective value: {model.objVal if model.solCount > 0 else 'No solution'}")

    # Prepare solution
    if model.solCount > 0:
        solution = {
            "objective_value": model.objVal,
            "defenders": [],
            "attackers": []
        }

        # Fill defender trajectories
        for d in range(N_D):
            def_traj = {
                "x": [x[d, k].X for k in range(N_u + 1)],
                "y": [y[d, k].X for k in range(N_u + 1)],
                "xdot": [xdot[d, k].X for k in range(N_u + 1)],
                "ydot": [ydot[d, k].X for k in range(N_u + 1)],
                "ux": [ux[d, k].X for k in range(N_u)],
                "uy": [uy[d, k].X for k in range(N_u)]
            }
            solution["defenders"].append(def_traj)

        # Fill attacker trajectories
        for a_idx in range(N_A):
            att_traj = {
                "p": [p[a_idx, k].X for k in range(N_a + 1)],
                "q": [q[a_idx, k].X for k in range(N_a + 1)],
                "a": [int(round(a[a_idx, k].X)) for k in range(N_a + 1)],
                "gamma": [int(round(gamma[a_idx, k].X)) for k in range(N_a + 1)],
                "delta": [
                    [int(round(delta[a_idx, d, k].X)) for k in range(N_a + 1)]
                    for d in range(N_D)
                ]
            }
            solution["attackers"].append(att_traj)

        return solution
    else:
        # Return empty solution if no solution found
        return {
            "objective_value": float('inf'),
            "defenders": [{"x": [], "y": [], "xdot": [], "ydot": [], "ux": [], "uy": []} for _ in range(N_D)],
            "attackers": [{"p": [], "q": [], "a": [], "gamma": [], "delta": [[] for _ in range(N_D)]} for _ in range(N_A)]
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    # Initialize logger
    logger = None
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize")

    # Load instance
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)

    # Solve
    solution = solve_defense_interception(instance_data, args.time_limit)

    # Log final solution
    if logger and solution.get("objective_value", float('inf')) != float('inf'):
        logger.log_solution(solution["objective_value"], solution)

    # Write solution
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()