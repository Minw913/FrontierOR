import argparse
import json
import numpy as np
import time
from itertools import combinations

def solve(instance_path, solution_path, time_limit, log_path=None):
    from solution_logger import SolutionLogger
    
    start_time = time.time()
    
    # Read instance data
    with open(instance_path, 'r') as f:
        data = json.load(f)
    
    N = data['N']
    S = data['S']
    k = data['k']
    beta = data['beta']
    gamma = data['gamma']
    mu_bar = data['mu_bar']
    mu = np.array(data['mu'])
    p_s = data['p_s']
    scenarios = np.array(data['scenarios'])  # S x N
    
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    # For small/medium instances, use Gurobi MIP directly
    # For large instances, we might need heuristics
    
    # Try Gurobi MIP approach
    try:
        import gurobipy as gp
        from gurobipy import GRB
        
        remaining_time = time_limit - (time.time() - start_time)
        
        if N <= 200 and S <= 5000:
            # Direct MIP formulation
            model = gp.Model("portfolio_cvar")
            model.setParam('TimeLimit', remaining_time - 2)
            model.setParam('Threads', 1)
            model.setParam('OutputFlag', 0)
            
            # Variables
            x = model.addVars(N, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="x")
            z = model.addVars(N, vtype=GRB.BINARY, name="z")
            a = model.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="a")
            v = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="v")
            # Auxiliary variables for CVaR: u_s = max(loss_s - a, 0)
            u = model.addVars(S, lb=0.0, vtype=GRB.CONTINUOUS, name="u")
            
            # Budget constraint: sum of x = 1
            model.addConstr(gp.quicksum(x[i] for i in range(N)) == 1.0, "budget")
            
            # Expected return constraint
            model.addConstr(gp.quicksum(mu[i] * x[i] for i in range(N)) >= mu_bar, "return")
            
            # Cardinality constraint
            model.addConstr(gp.quicksum(z[i] for i in range(N)) <= k, "cardinality")
            
            # Linking constraints: x[i] <= z[i]
            for i in range(N):
                model.addConstr(x[i] <= z[i], f"link_{i}")
            
            # CVaR constraints: u_s >= -sum(scenarios[s][i]*x[i]) - a
            for s in range(S):
                model.addConstr(
                    u[s] >= -gp.quicksum(scenarios[s, i] * x[i] for i in range(N)) - a,
                    f"cvar_{s}"
                )
            
            # v >= (1/(1-beta)) * sum(p_s * u_s)
            model.addConstr(
                v >= (1.0 / (1.0 - beta)) * gp.quicksum(p_s * u[s] for s in range(S)),
                "v_cvar"
            )
            
            # Objective: minimize (1/(2*gamma)) * sum(x_i^2) + a + v
            obj = (1.0 / (2.0 * gamma)) * gp.quicksum(x[i] * x[i] for i in range(N)) + a + v
            model.setObjective(obj, GRB.MINIMIZE)
            
            # Callback for logging
            best_obj = [float('inf')]
            best_sol = [None]
            
            def callback(model, where):
                if where == GRB.Callback.MIPSOL:
                    obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    if obj_val < best_obj[0] - 1e-9:
                        best_obj[0] = obj_val
                        x_vals = model.cbGetSolution([x[i] for i in range(N)])
                        z_vals = model.cbGetSolution([z[i] for i in range(N)])
                        a_val = model.cbGetSolution(a)
                        v_val = model.cbGetSolution(v)
                        best_sol[0] = {
                            'x': list(x_vals),
                            'z': [int(round(zv)) for zv in z_vals],
                            'a': a_val,
                            'v': v_val,
                            'objective_value': obj_val
                        }
                        if logger:
                            logger.log(obj_val)
            
            model.optimize(callback)
            
            if model.SolCount > 0:
                sol = {
                    'objective_value': model.ObjVal,
                    'x': [x[i].X for i in range(N)],
                    'z': [int(round(z[i].X)) for i in range(N)],
                    'a': a.X,
                    'v': v.X
                }
                if logger and model.ObjVal < best_obj[0] - 1e-9:
                    logger.log(model.ObjVal)
            elif best_sol[0] is not None:
                sol = best_sol[0]
            else:
                # Fallback: try heuristic
                sol = heuristic_solve(N, S, k, beta, gamma, mu_bar, mu, p_s, scenarios, start_time, time_limit, logger)
            
        else:
            # For larger instances, use a heuristic + Gurobi for subproblems
            sol = large_instance_solve(N, S, k, beta, gamma, mu_bar, mu, p_s, scenarios, start_time, time_limit, logger)
        
    except Exception as e:
        # Fallback to heuristic
        sol = heuristic_solve(N, S, k, beta, gamma, mu_bar, mu, p_s, scenarios, start_time, time_limit, logger)
    
    # Write solution
    with open(solution_path, 'w') as f:
        json.dump(sol, f)


def large_instance_solve(N, S, k, beta, gamma, mu_bar, mu, p_s, scenarios, start_time, time_limit, logger):
    """For larger instances, enumerate promising subsets or use greedy + local search."""
    import gurobipy as gp
    from gurobipy import GRB
    
    best_obj = float('inf')
    best_sol = None
    
    # Precompute scenario losses for each asset
    # loss_s = -scenarios[s] @ x
    
    def solve_continuous_subproblem(selected_assets, time_remaining):
        """Given a set of selected assets, solve the continuous QP."""
        n_sel = len(selected_assets)
        if n_sel == 0:
            return None
        
        try:
            model = gp.Model("sub")
            model.setParam('TimeLimit', max(time_remaining - 1, 1))
            model.setParam('Threads', 1)
            model.setParam('OutputFlag', 0)
            
            x = model.addVars(n_sel, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="x")
            a = model.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="a")
            v = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="v")
            u = model.addVars(S, lb=0.0, vtype=GRB.CONTINUOUS, name="u")
            
            sel = list(selected_assets)
            
            model.addConstr(gp.quicksum(x[i] for i in range(n_sel)) == 1.0)
            model.addConstr(gp.quicksum(mu[sel[i]] * x[i] for i in range(n_sel)) >= mu_bar)
            
            # Precompute sub-scenarios
            sub_scenarios = scenarios[:, sel]
            
            for s in range(S):
                model.addConstr(
                    u[s] >= -gp.quicksum(sub_scenarios[s, i] * x[i] for i in range(n_sel)) - a
                )
            
            model.addConstr(
                v >= (1.0 / (1.0 - beta)) * gp.quicksum(p_s * u[s] for s in range(S))
            )
            
            obj = (1.0 / (2.0 * gamma)) * gp.quicksum(x[i] * x[i] for i in range(n_sel)) + a + v
            model.setObjective(obj, GRB.MINIMIZE)
            model.optimize()
            
            if model.SolCount > 0:
                x_full = [0.0] * N
                z_full = [0] * N
                for i in range(n_sel):
                    x_full[sel[i]] = x[i].X
                    z_full[sel[i]] = 1
                return {
                    'objective_value': model.ObjVal,
                    'x': x_full,
                    'z': z_full,
                    'a': a.X,
                    'v': v.X
                }
        except:
            pass
        return None
    
    # Greedy selection: pick top-k assets by expected return (those >= mu_bar if possible)
    # Also try diversification-based selection
    
    # Strategy 1: Top-k by expected return
    sorted_by_return = np.argsort(-mu)
    
    candidates_sets = []
    
    # Top-k by return
    top_k_return = sorted_by_return[:k].tolist()
    candidates_sets.append(set(top_k_return))
    
    # Check if mu_bar is achievable with top-k
    # If not, we must include high-return assets
    
    # Strategy 2: Assets with return >= mu_bar, then fill with low-variance assets
    high_return = [i for i in range(N) if mu[i] >= mu_bar]
    if len(high_return) <= k:
        # Add more assets for diversification
        remaining = [i for i in range(N) if i not in high_return]
        # Sort remaining by variance (lower is better for risk)
        variances = np.var(scenarios, axis=0)
        remaining_sorted = sorted(remaining, key=lambda i: variances[i])
        candidate = set(high_return + remaining_sorted[:k - len(high_return)])
        candidates_sets.append(candidate)
    else:
        # Pick top-k from high_return by some criterion
        # Sharpe-like ratio
        stds = np.std(scenarios, axis=0)
        stds = np.maximum(stds, 1e-10)
        sharpe = mu / stds
        high_return_sorted = sorted(high_return, key=lambda i: -sharpe[i])
        candidates_sets.append(set(high_return_sorted[:k]))
    
    # Strategy 3: Sharpe ratio based
    stds = np.std(scenarios, axis=0)
    stds = np.maximum(stds, 1e-10)
    sharpe = mu / stds
    top_k_sharpe = np.argsort(-sharpe)[:k].tolist()
    candidates_sets.append(set(top_k_sharpe))
    
    # Strategy 4: Mix of return and low CVaR contribution
    # Use mean - some penalty * downside
    downside = np.mean(np.maximum(-scenarios, 0), axis=0)
    score = mu - 0.5 * downside
    top_k_score = np.argsort(-score)[:k].tolist()
    candidates_sets.append(set(top_k_score))
    
    for candidate_set in candidates_sets:
        elapsed = time.time() - start_time
        if elapsed > time_limit - 5:
            break
        
        remaining = time_limit - elapsed - 2
        result = solve_continuous_subproblem(candidate_set, remaining)
        if result is not None and result['objective_value'] < best_obj:
            best_obj = result['objective_value']
            best_sol = result
            if logger:
                logger.log(best_obj)
    
    # Local search: try swapping assets
    if best_sol is not None:
        current_set = set(i for i in range(N) if best_sol['z'][i] == 1)
        excluded = set(range(N)) - current_set
        
        improved = True
        while improved:
            improved = False
            elapsed = time.time() - start_time
            if elapsed > time_limit - 10:
                break
            
            for out_asset in list(current_set):
                for in_asset in list(excluded):
                    elapsed = time.time() - start_time
                    if elapsed > time_limit - 5:
                        break
                    
                    new_set = (current_set - {out_asset}) | {in_asset}
                    remaining = time_limit - (time.time() - start_time) - 2
                    result = solve_continuous_subproblem(new_set, min(remaining, 10))
                    if result is not None and result['objective_value'] < best_obj - 1e-8:
                        best_obj = result['objective_value']
                        best_sol = result
                        current_set = new_set
                        excluded = set(range(N)) - current_set
                        improved = True
                        if logger:
                            logger.log(best_obj)
                        break
                if improved:
                    break
                elapsed = time.time() - start_time
                if elapsed > time_limit - 5:
                    break
    
    if best_sol is None:
        best_sol = heuristic_solve(N, S, k, beta, gamma, mu_bar, mu, p_s, scenarios, start_time, time_limit, logger)
    
    return best_sol


def heuristic_solve(N, S, k, beta, gamma, mu_bar, mu, p_s, scenarios, start_time, time_limit, logger):
    """Simple heuristic fallback."""
    # Pick the asset with highest return and put all weight on it
    # Or pick top-k and equal weight
    
    sorted_idx = np.argsort(-mu)
    selected = sorted_idx[:k].tolist()
    
    # Equal weight among selected
    n_sel = len(selected)
    x_vals = np.zeros(N)
    
    # Check if equal weight satisfies return constraint
    eq_return = np.mean(mu[selected])
    
    if eq_return >= mu_bar:
        for i in selected:
            x_vals[i] = 1.0 / n_sel
    else:
        # Put all weight on highest return asset
        x_vals[selected[0]] = 1.0
    
    z_vals = [0] * N
    for i in selected:
        z_vals[i] = 1
    
    # Compute objective
    reg = (1.0 / (2.0 * gamma)) * np.sum(x_vals ** 2)
    
    # Compute optimal a and v given x
    losses = -scenarios @ x_vals  # S-vector
    
    # CVaR: minimize a + 1/(1-beta) * E[max(loss - a, 0)]
    # Optimal a is the beta-quantile of losses
    sorted_losses = np.sort(losses)
    a_val = np.percentile(losses, beta * 100)
    
    excess = np.maximum(losses - a_val, 0)
    v_val = (1.0 / (1.0 - beta)) * p_s * np.sum(excess)
    
    obj = reg + a_val + v_val
    
    if logger:
        logger.log(obj)
    
    return {
        'objective_value': float(obj),
        'x': x_vals.tolist(),
        'z': z_vals,
        'a': float(a_val),
        'v': float(v_val)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    solve(args.instance_path, args.solution_path, args.time_limit, args.log_path)