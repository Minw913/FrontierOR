import argparse
import json
import math
import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None

def main():
    parser = argparse.ArgumentParser(description="Solve Defending and Attacking Robots Problem")
    parser.add_argument('--instance_path', type=str, required=True, help="Path to the instance JSON file.")
    parser.add_argument('--solution_path', type=str, required=True, help="Path to the output solution JSON file.")
    parser.add_argument('--time_limit', type=int, default=300, help="Time limit in seconds.")
    parser.add_argument('--log_path', type=str, default='', help="Path to the log file.")
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if (SolutionLogger and args.log_path) else None

    # 1. Read the instance
    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    N_D = data['N_D']
    N_A = data['N_A']
    epsilon = data['epsilon']
    M_u = data['M_u']
    M_I = data['M_I']
    M_dz = data['M_dz']
    N_u = data['N_u']
    N_a = data['N_a']
    R_dz = data['R_dz']
    R_I = data['R_I']
    v_a = data['v_a']
    T_a = data['T_a']
    T_u = data['T_u']

    attackers = data['attackers']
    defenders = data['defenders']

    # 2. Precompute attackers' nominal positions
    P_nom = []
    for j in range(N_A):
        p_s = attackers[j]['p_s']
        q_s = attackers[j]['q_s']
        v_p = attackers[j]['v_p']
        v_q = attackers[j]['v_q']
        path = []
        for k in range(N_a + 1):
            path.append((p_s + k * T_a * v_p, q_s + k * T_a * v_q))
        P_nom.append(path)

    # 3. Precompute step K_j at which attacker j enters Defense Zone
    def check_in_dz(px, py):
        for m_idx in range(M_dz):
            angle = 2 * math.pi * m_idx / M_dz
            if px * math.cos(angle) + py * math.sin(angle) > R_dz + 1e-7:
                return False
        return True

    K = []
    for j in range(N_A):
        k_in = N_a + 1
        for k in range(N_a + 1):
            if check_in_dz(P_nom[j][k][0], P_nom[j][k][1]):
                k_in = k
                break
        K.append(k_in)

    # Calculate Big-M bound based on max possible defender coordinate
    V_max = 2.0
    for d in range(N_D):
        vx0 = abs(defenders[d]['xdot_s'])
        vy0 = abs(defenders[d]['ydot_s'])
        V_max = max(V_max, vx0, vy0)
    MaxPos = 0.0
    for d in range(N_D):
        MaxPos = max(MaxPos, abs(defenders[d]['x_s']), abs(defenders[d]['y_s']))
    MaxPos += N_u * T_u * V_max + 1.0

    # 4. Build MILP Model
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    m = gp.Model("Defense", env=env)
    m.setParam('Seed', 0)
    m.setParam('MIPGap', 1e-4)
    m.setParam('NumericFocus', 0)
    m.setParam('Threads', 1)
    if args.time_limit:
        m.setParam('TimeLimit', args.time_limit)

    # Defender state variables
    p_x = m.addVars(N_D, N_u + 1, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="p_x")
    p_y = m.addVars(N_D, N_u + 1, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="p_y")
    v_x = m.addVars(N_D, N_u + 1, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="v_x")
    v_y = m.addVars(N_D, N_u + 1, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="v_y")
    u_x = m.addVars(N_D, N_u, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="u_x")
    u_y = m.addVars(N_D, N_u, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="u_y")
    u_abs_x = m.addVars(N_D, N_u, lb=0, name="u_abs_x")
    u_abs_y = m.addVars(N_D, N_u, lb=0, name="u_abs_y")

    # Initial states
    for d in range(N_D):
        m.addConstr(p_x[d, 0] == defenders[d]['x_s'])
        m.addConstr(p_y[d, 0] == defenders[d]['y_s'])
        m.addConstr(v_x[d, 0] == defenders[d]['xdot_s'])
        m.addConstr(v_y[d, 0] == defenders[d]['ydot_s'])

    # Dynamics & Control Constraints
    exp_Tu = math.exp(-T_u)
    for d in range(N_D):
        for c in range(N_u):
            m.addConstr(v_x[d, c+1] == v_x[d, c] * exp_Tu + u_x[d, c] * (1 - exp_Tu))
            m.addConstr(v_y[d, c+1] == v_y[d, c] * exp_Tu + u_y[d, c] * (1 - exp_Tu))
            m.addConstr(p_x[d, c+1] == p_x[d, c] + v_x[d, c] * (1 - exp_Tu) + u_x[d, c] * (T_u - 1 + exp_Tu))
            m.addConstr(p_y[d, c+1] == p_y[d, c] + v_y[d, c] * (1 - exp_Tu) + u_y[d, c] * (T_u - 1 + exp_Tu))
            
            m.addConstr(u_abs_x[d, c] >= u_x[d, c])
            m.addConstr(u_abs_x[d, c] >= -u_x[d, c])
            m.addConstr(u_abs_y[d, c] >= u_y[d, c])
            m.addConstr(u_abs_y[d, c] >= -u_y[d, c])
            
            for m_idx in range(M_u):
                angle = 2 * math.pi * m_idx / M_u
                m.addConstr(u_x[d, c] * math.cos(angle) + u_y[d, c] * math.sin(angle) <= math.cos(math.pi / M_u))

    # Avoidance constraints (defenders must stay outside the DZ obstacle)
    O = m.addVars(N_D, N_u + 1, 4, vtype=GRB.BINARY, name="O")
    M_avoid = R_dz + MaxPos + 10.0
    for d in range(N_D):
        for c in range(1, N_u + 1):
            m.addConstr(O[d, c, 0] + O[d, c, 1] + O[d, c, 2] + O[d, c, 3] >= 1)
            m.addConstr(p_x[d, c] >= R_dz - M_avoid * (1 - O[d, c, 0]))
            m.addConstr(p_y[d, c] >= R_dz - M_avoid * (1 - O[d, c, 1]))
            m.addConstr(-p_x[d, c] >= R_dz - M_avoid * (1 - O[d, c, 2]))
            m.addConstr(-p_y[d, c] >= R_dz - M_avoid * (1 - O[d, c, 3]))

    # Interception variables and constraints
    I_var = {}
    for j in range(N_A):
        max_k = min(K[j], N_a)
        for k in range(max_k + 1):
            for d in range(N_D):
                I_var[d, j, k] = m.addVar(vtype=GRB.BINARY, name=f"I_{d}_{j}_{k}")

    for j in range(N_A):
        max_k = min(K[j], N_a)
        m.addConstr(gp.quicksum(I_var[d, j, k] for k in range(max_k + 1) for d in range(N_D)) <= 1)

    for j in range(N_A):
        max_k = min(K[j], N_a)
        for k in range(max_k + 1):
            t = k * T_a
            c = int(math.floor(t / T_u + 1e-7))
            if c >= N_u:
                c = N_u - 1
            alpha = (t - c * T_u) / T_u
            if alpha > 1.0: alpha = 1.0
            if alpha < 0.0: alpha = 0.0
            
            for d in range(N_D):
                pdx = p_x[d, c] + alpha * (p_x[d, c+1] - p_x[d, c])
                pdy = p_y[d, c] + alpha * (p_y[d, c+1] - p_y[d, c])
                
                P_nom_x = P_nom[j][k][0]
                P_nom_y = P_nom[j][k][1]
                
                for m_idx in range(M_I):
                    angle = 2 * math.pi * m_idx / M_I
                    Cm = math.cos(angle)
                    Sm = math.sin(angle)
                    
                    RHS_val = Cm * P_nom_x + Sm * P_nom_y - R_I
                    BigM = abs(RHS_val) + (abs(Cm) + abs(Sm)) * MaxPos + 10.0
                    
                    m.addConstr(Cm * pdx + Sm * pdy >= RHS_val - BigM * (1 - I_var[d, j, k]))

    # Objective Function
    obj = gp.LinExpr()
    for j in range(N_A):
        if K[j] <= N_a:
            y_j = gp.quicksum(I_var[d, j, k] for k in range(K[j] + 1) for d in range(N_D))
            obj += (N_a - K[j] + 1) * (1 - y_j)

    ctrl_effort = gp.quicksum(u_abs_x[d, c] + u_abs_y[d, c] for d in range(N_D) for c in range(N_u))
    obj += epsilon * ctrl_effort

    m.setObjective(obj, GRB.MINIMIZE)

    # 5. Build Solution helper function
    def build_solution(get_value):
        sol_defenders = []
        for d in range(N_D):
            sol_def = {
                "x": [float(get_value(p_x[d, c])) for c in range(N_u + 1)],
                "y": [float(get_value(p_y[d, c])) for c in range(N_u + 1)],
                "xdot": [float(get_value(v_x[d, c])) for c in range(N_u + 1)],
                "ydot": [float(get_value(v_y[d, c])) for c in range(N_u + 1)],
                "ux": [float(get_value(u_x[d, c])) for c in range(N_u)],
                "uy": [float(get_value(u_y[d, c])) for c in range(N_u)],
            }
            sol_defenders.append(sol_def)
        
        sol_attackers = []
        total_gamma = 0
        
        for j in range(N_A):
            p_hist = []
            q_hist = []
            a_hist = []
            gamma_hist = []
            delta_hist = [[0] * (N_a + 1) for _ in range(N_D)]
            
            state = "ACTIVE"
            stop_reason = None
            px, py = P_nom[j][0][0], P_nom[j][0][1]
            
            for k in range(N_a + 1):
                t = k * T_a
                c = int(math.floor(t / T_u + 1e-7))
                if c >= N_u:
                    c = N_u - 1
                alpha = (t - c * T_u) / T_u
                if alpha > 1.0: alpha = 1.0
                if alpha < 0.0: alpha = 0.0
                
                if state == "ACTIVE":
                    px = P_nom[j][k][0]
                    py = P_nom[j][k][1]
                    in_dz = (k >= K[j])
                    
                    intercepting_defs = []
                    for d in range(N_D):
                        milp_intercept = False
                        if k <= K[j] and k <= N_a:
                            if get_value(I_var[d, j, k]) > 0.5:
                                milp_intercept = True
                        
                        if not milp_intercept:
                            pdx = sol_defenders[d]["x"][c] + alpha * (sol_defenders[d]["x"][c+1] - sol_defenders[d]["x"][c])
                            pdy = sol_defenders[d]["y"][c] + alpha * (sol_defenders[d]["y"][c+1] - sol_defenders[d]["y"][c])
                            
                            intercept = True
                            for m_idx in range(M_I):
                                angle = 2 * math.pi * m_idx / M_I
                                if (px - pdx) * math.cos(angle) + (py - pdy) * math.sin(angle) > R_I + 1e-5:
                                    intercept = False
                                    break
                            if intercept:
                                milp_intercept = True
                        
                        if milp_intercept:
                            intercepting_defs.append(d)
                    
                    if in_dz or len(intercepting_defs) > 0:
                        state = "INACTIVE"
                        if len(intercepting_defs) > 0:
                            for d in intercepting_defs:
                                delta_hist[d][k] = 1
                            stop_reason = "INTERCEPTED"
                            is_in_dz_now = False
                        else:
                            stop_reason = "DZ"
                            is_in_dz_now = True
                    else:
                        is_in_dz_now = False
                        
                else:
                    if stop_reason == "DZ":
                        is_in_dz_now = True
                    else:
                        is_in_dz_now = False
                
                p_hist.append(float(px))
                q_hist.append(float(py))
                a_hist.append(1 if state == "ACTIVE" else 0)
                gamma_hist.append(1 if is_in_dz_now else 0)
                if is_in_dz_now:
                    total_gamma += 1
                    
            sol_attackers.append({
                "p": p_hist,
                "q": q_hist,
                "a": a_hist,
                "gamma": gamma_hist,
                "delta": delta_hist
            })
        
        ctrl = 0.0
        for d in range(N_D):
            for c in range(N_u):
                ctrl += abs(sol_defenders[d]["ux"][c]) + abs(sol_defenders[d]["uy"][c])
                
        obj_val = float(total_gamma + epsilon * ctrl)
        
        sol_dict = {
            "objective_value": obj_val,
            "defenders": sol_defenders,
            "attackers": sol_attackers
        }
        return obj_val, sol_dict

    # 6. Optimization and Callback
    def cb(model, where):
        if where == GRB.Callback.MIPSOL:
            def get_val(var):
                return model.cbGetSolution(var)
            obj_val, sol_dict = build_solution(get_val)
            if logger:
                logger.log_solution(obj_val, sol_dict)

    m.optimize(cb)

    # 7. Dump final solution
    if m.SolCount > 0:
        def get_val(var):
            return var.X
        obj_val, sol_dict = build_solution(get_val)
        with open(args.solution_path, 'w') as f:
            json.dump(sol_dict, f, indent=2)

if __name__ == "__main__":
    main()