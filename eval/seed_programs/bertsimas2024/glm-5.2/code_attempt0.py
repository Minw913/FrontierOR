import json
import argparse
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
    
    with open(args.instance_path) as f:
        data = json.load(f)
    
    num_nodes = data['num_nodes']
    K = data['num_commodities']
    S = data['num_scenarios']
    E = data['num_edges']
    gamma = data['gamma']
    c_0 = data['c_0']
    edges = data['edges']
    construction_costs = data['construction_costs']
    flow_costs = data['flow_costs']
    capacities = data['capacities']
    demands = data['demands']
    
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    # Build adjacency lists
    out_edges = [[] for _ in range(num_nodes)]
    in_edges = [[] for _ in range(num_nodes)]
    for e_idx, (i, j) in enumerate(edges):
        out_edges[i].append(e_idx)
        in_edges[j].append(e_idx)
    
    # Create Gurobi environment
    env = gp.Env(empty=True)
    env.setParam('Seed', 0)
    env.setParam('Threads', 1)
    env.setParam('NumericFocus', 0)
    env.start()
    
    model = gp.Model(env=env)
    model.setParam('MIPGap', 1e-4)
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('OutputFlag', 0)
    
    if args.time_limit <= 120:
        model.setParam('MIPFocus', 1)
    
    # Variables
    z = model.addVars(E, vtype=GRB.BINARY, name='z')
    x = model.addVars(K, S, E, lb=0, name='x')
    tf = model.addVars(S, E, lb=0, name='tf')
    
    # Capacity linking: tf[r,e] <= capacities[e] * z[e]
    model.addConstrs(
        (tf[r, e] <= capacities[e] * z[e] for r in range(S) for e in range(E)),
        name='cap'
    )
    
    # Total flow definition: tf[r,e] = sum_k x[k,r,e]
    model.addConstrs(
        (tf[r, e] == gp.quicksum(x[k, r, e] for k in range(K)) for r in range(S) for e in range(E)),
        name='tf_def'
    )
    
    # Flow conservation: for each commodity k, scenario r, node n
    model.addConstrs(
        (gp.quicksum(x[k, r, e] for e in out_edges[n]) - gp.quicksum(x[k, r, e] for e in in_edges[n]) == demands[r][k][n]
         for k in range(K) for r in range(S) for n in range(num_nodes)),
        name='fc'
    )
    
    # Edge count limit
    model.addConstr(gp.quicksum(z[e] for e in range(E)) <= c_0, name='edge_limit')
    
    # Objective
    obj = gp.QuadExpr()
    for e in range(E):
        obj += construction_costs[e] * z[e]
    
    inv_S = 1.0 / S
    inv_2gamma = 1.0 / (2.0 * gamma)
    
    for r in range(S):
        for e in range(E):
            obj += inv_S * flow_costs[e] * tf[r, e]
            obj += inv_S * inv_2gamma * tf[r, e] * tf[r, e]
    
    model.setObjective(obj, GRB.MINIMIZE)
    
    # Callback for logging incumbents
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            z_vals = [model.cbGetSolution(z[e]) for e in range(E)]
            
            model_vars = {}
            z_dict = {}
            for e in range(E):
                v = z_vals[e]
                rv = 1.0 if v > 0.5 else 0.0
                z_dict[str(e)] = rv
                if rv > 0.5:
                    model_vars[f'z[{e}]'] = rv
            
            for e in range(E):
                if z_vals[e] > 0.5:
                    for k in range(K):
                        for r in range(S):
                            val = model.cbGetSolution(x[k, r, e])
                            if abs(val) > 1e-6:
                                model_vars[f'x[{k},{r},{e}]'] = val
                    for r in range(S):
                        val = model.cbGetSolution(tf[r, e])
                        if abs(val) > 1e-6:
                            model_vars[f'tf[{r},{e}]'] = val
            
            sol_dict = {
                'objective_value': obj_val,
                'model_variables': model_vars,
                'z': z_dict
            }
            
            if logger:
                logger.log_solution(obj_val, sol_dict)
    
    model.optimize(callback=callback if logger else None)
    
    # Extract final solution
    if model.SolCount > 0:
        obj_val = model.ObjVal
        
        model_vars = {}
        z_dict = {}
        for e in range(E):
            v = z[e].X
            rv = 1.0 if v > 0.5 else 0.0
            z_dict[str(e)] = rv
            if rv > 0.5:
                model_vars[f'z[{e}]'] = rv
        
        for k in range(K):
            for r in range(S):
                for e in range(E):
                    val = x[k, r, e].X
                    if abs(val) > 1e-6:
                        model_vars[f'x[{k},{r},{e}]'] = val
        
        for r in range(S):
            for e in range(E):
                val = tf[r, e].X
                if abs(val) > 1e-6:
                    model_vars[f'tf[{r},{e}]'] = val
        
        solution = {
            'objective_value': obj_val,
            'model_variables': model_vars,
            'z': z_dict
        }
    else:
        # No solution found
        solution = {
            'objective_value': 0.0,
            'model_variables': {},
            'z': {str(e): 0.0 for e in range(E)}
        }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == '__main__':
    main()