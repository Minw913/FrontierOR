import argparse
import json
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    return instance

def solve_instance(instance, time_limit, logger):
    N_D = instance['N_D']
    N_A = instance['N_A']
    epsilon = instance['epsilon']
    M_u = instance['M_u']
    M_I = instance['M_I']
    M_dz = instance['M_dz']
    N_u = instance['N_u']
    N_a = instance['N_a']
    R_dz = instance['R_dz']
    R_I = instance['R_I']
    T_a = instance['T_a']
    T_u = instance['T_u']
    defenders = instance['defenders']
    attackers = instance['attackers']

    try:
        model = gp.Model("defender_control")
        model.Params.Seed = 0
        model.Params.MIPGap = 1e-4
        model.Params.NumericFocus = 0
        model.Params.Threads = 1
        model.Params.TimeLimit = time_limit

        # Defender variables
        x = model.addVars(N_D, N_u+1, lb=-GRB.INFINITY, name="x")
        y = model.addVars(N_D, N_u+1, lb=-GRB.INFINITY, name="y")
        xdot = model.addVars(N_D, N_u+1, lb=-GRB.INFINITY, name="xdot")
        ydot = model.addVars(N_D, N_u+1, lb=-GRB.INFINITY, name="ydot")
        ux = model.addVars(N_D, N_u, lb=-1, ub=1, name="ux")
        uy = model.addVars(N_D, N_u, lb=-1, ub=1, name="uy")

        # Attacker variables
        p = model.addVars(N_A, N_a+1, lb=-GRB.INFINITY, name="p")
        q = model.addVars(N_A, N_a+1, lb=-GRB.INFINITY, name="q")
        a = model.addVars(N_A, N_a+1, vtype=GRB.BINARY, name="a")
        gamma = model.addVars(N_A, N_a+1, vtype=GRB.BINARY, name="gamma")
        delta = model.addVars(N_D, N_A, N_a+1, vtype=GRB.BINARY, name="delta")

        # Control input constraints
        for d in range(N_D):
            for k in range(N_u):
                model.addConstr(gp.quicksum([ux[d, k] * np.cos(2 * np.pi * i / M_u) + uy[d, k] * np.sin(2 * np.pi * i / M_u) for i in range(M_u)]) <= 1, name=f"ctrl_input_{d}_{k}")

        # Defender dynamics
        exp_Tu = np.exp(-T_u)
        for d in range(N_D):
            x[d, 0].setAttr(GRB.Attr.LB, defenders[d]['x_s'])
            x[d, 0].setAttr(GRB.Attr.UB, defenders[d]['x_s'])
            y[d, 0].setAttr(GRB.Attr.LB, defenders[d]['y_s'])
            y[d, 0].setAttr(GRB.Attr.UB, defenders[d]['y_s'])
            xdot[d, 0].setAttr(GRB.Attr.LB, defenders[d]['xdot_s'])
            xdot[d, 0].setAttr(GRB.Attr.UB, defenders[d]['xdot_s'])
            ydot[d, 0].setAttr(GRB.Attr.LB, defenders[d]['ydot_s'])
            ydot[d, 0].setAttr(GRB.Attr.UB, defenders[d]['ydot_s'])
            for k in range(N_u):
                model.addConstr(xdot[d, k+1] == xdot[d, k] * exp_Tu + ux[d, k] * (1 - exp_Tu), name=f"xdot_dyn_{d}_{k}")
                model.addConstr(ydot[d, k+1] == ydot[d, k] * exp_Tu + uy[d, k] * (1 - exp_Tu), name=f"ydot_dyn_{d}_{k}")
                model.addConstr(x[d, k+1] == x[d, k] + xdot[d, k] * (1 - exp_Tu) + ux[d, k] * (T_u - 1 + exp_Tu), name=f"x_dyn_{d}_{k}")
                model.addConstr(y[d, k+1] == y[d, k] + ydot[d, k] * (1 - exp_Tu) + uy[d, k] * (T_u - 1 + exp_Tu), name=f"y_dyn_{d}_{k}")

        # Attacker dynamics
        for a_idx in range(N_A):
            p[a_idx, 0].setAttr(GRB.Attr.LB, attackers[a_idx]['p_s'])
            p[a_idx, 0].setAttr(GRB.Attr.UB, attackers[a_idx]['p_s'])
            q[a_idx, 0].setAttr(GRB.Attr.LB, attackers[a_idx]['q_s'])
            q[a_idx, 0].setAttr(GRB.Attr.UB, attackers[a_idx]['q_s'])
            for k in range(N_a):
                model.addConstr(p[a_idx, k+1] == p[a_idx, k] + attackers[a_idx]['v_p'] * T_a, name=f"p_dyn_{a_idx}_{k}")
                model.addConstr(q[a_idx, k+1] == q[a_idx, k] + attackers[a_idx]['v_q'] * T_a, name=f"q_dyn_{a_idx}_{k}")

        # Attacker status
        for a_idx in range(N_A):
            a[a_idx, 0].setAttr(GRB.Attr.LB, 1)
            a[a_idx, 0].setAttr(GRB.Attr.UB, 1)
            for k in range(1, N_a+1):
                model.addConstr(a[a_idx, k] <= a[a_idx, k-1], name=f"a_deactivate_{a_idx}_{k}")
                model.addConstr(gamma[a_idx, k] <= a[a_idx, k], name=f"gamma_active_{a_idx}_{k}")
                for d in range(N_D):
                    model.addConstr(delta[d, a_idx, k] <= a[a_idx, k], name=f"delta_active_{d}_{a_idx}_{k}")

        # Defense Zone and Interception
        for a_idx in range(N_A):
            for k in range(1, N_a+1):
                for i in range(M_dz):
                    angle = 2 * np.pi * i / M_dz
                    model.addConstr(np.cos(angle) * p[a_idx, k] + np.sin(angle) * q[a_idx, k] <= R_dz + (1 - gamma[a_idx, k]) * 1e6, name=f"dz_{a_idx}_{k}_{i}")
                for d in range(N_D):
                    t = k * T_a / T_u
                    k_low = int(np.floor(t))
                    k_high = min(k_low + 1, N_u)
                    alpha = t - k_low
                    x_interp = (1 - alpha) * x[d, k_low] + alpha * x[d, k_high]
                    y_interp = (1 - alpha) * y[d, k_low] + alpha * y[d, k_high]
                    for i in range(M_I):
                        angle = 2 * np.pi * i / M_I
                        model.addConstr(np.cos(angle) * (p[a_idx, k] - x_interp) + np.sin(angle) * (q[a_idx, k] - y_interp) <= R_I + (1 - delta[d, a_idx, k]) * 1e6, name=f"intercept_{d}_{a_idx}_{k}_{i}")
                model.addConstr(gp.quicksum(delta[d, a_idx, k] for d in range(N_D)) + gamma[a_idx, k] >= 1 - (1 - a[a_idx, k]) * 1e6, name=f"intercept_or_dz_{a_idx}_{k}")

        # Defender avoidance
        for d1 in range(N_D):
            for d2 in range(d1 + 1, N_D):
                for k in range(1, N_u+1):
                    for i in range(4):
                        angle = 2 * np.pi * i / 4 + np.pi / 4
                        model.addConstr(np.cos(angle) * (x[d1, k] - x[d2, k]) + np.sin(angle) * (y[d1, k] - y[d2, k]) >= 0, name=f"avoidance_{d1}_{d2}_{k}_{i}")

        # Objective
        objective = gp.quicksum(gamma[a_idx, k] for a_idx in range(N_A) for k in range(1, N_a+1)) + epsilon * gp.quicksum(gp.abs_(ux[d, k]) + gp.abs_(uy[d, k]) for d in range(N_D) for k in range(N_u))
        model.setObjective(objective, GRB.MINIMIZE)

        model.optimize()
        if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
            solution = {
                "objective_value": model.getObjective().getValue(),
                "defenders": [],
                "attackers": []
            }
            for d in range(N_D):
                defender = {
                    "x": [x[d, k].X for k in range(N_u+1)],
                    "y": [y[d, k].X for k in range(N_u+1)],
                    "xdot": [xdot[d, k].X for k in range(N_u+1)],
                    "ydot": [ydot[d, k].X for k in range(N_u+1)],
                    "ux": [ux[d, k].X for k in range(N_u)],
                    "uy": [uy[d, k].X for k in range(N_u)]
                }
                solution["defenders"].append(defender)
            for a_idx in range(N_A):
                attacker = {
                    "p": [p[a_idx, k].X for k in range(N_a+1)],
                    "q": [q[a_idx, k].X for k in range(N_a+1)],
                    "a": [a[a_idx, k].X for k in range(N_a+1)],
                    "gamma": [gamma[a_idx, k].X for k in range(N_a+1)],
                    "delta": [[delta[d, a_idx, k].X for k in range(N_a+1)] for d in range(N_D)]
                }
                solution["attackers"].append(attacker)
            if logger:
                logger.log_solution(model.getObjective().getValue(), solution)
            return solution
        else:
            return None
    except gp.GurobiError as e:
        print(f'Gurobi error: {e}')
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True, help='Path to the JSON file containing the problem instance.')
    parser.add_argument('--solution_path', required=True, help='Path where the final solution JSON file must be written.')
    parser.add_argument('--time_limit', type=int, required=True, help='Maximum runtime in seconds.')
    parser.add_argument('--log_path', help='Path to a JSONL file for logging intermediate solutions.')
    args = parser.parse_args()

    instance = read_instance(args.instance_path)
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    solution = solve_instance(instance, args.time_limit, logger)
    if solution is not None:
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=4)

if __name__ == "__main__":
    main()