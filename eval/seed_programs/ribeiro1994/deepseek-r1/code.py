import argparse
import json
from collections import defaultdict
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
    
    logger = SolutionLogger(args.log_path, "minimize") if args.log_path else None

    with open(args.instance_path, 'r') as f:
        data = json.load(f)
    
    n_trips = data['parameters']['n_trips']
    m_depots = data['parameters']['m_depots']
    trips = data['trips']
    vehicle_capacities = data['vehicle_capacities']
    
    trip_id_to_index = {}
    for idx, trip in enumerate(trips):
        trip_id_to_index[trip['trip_id']] = idx
        
    depot_start_arcs = {}
    for item in data['depot_to_trip_costs']:
        depot_id = item['depot']
        trip_id = item['trip']
        cost_val = item['cost']
        d_index = depot_id - 1
        trip_index = trip_id_to_index[trip_id]
        depot_start_arcs[(d_index, trip_index)] = cost_val
        
    depot_end_arcs = {}
    for item in data['trip_to_depot_costs']:
        trip_id = item['trip']
        depot_id = item['depot']
        cost_val = item['cost']
        d_index = depot_id - 1
        trip_index = trip_id_to_index[trip_id]
        depot_end_arcs[(trip_index, d_index)] = cost_val
        
    compat_arcs = []
    for item in data['compatible_arcs']:
        from_trip_id = item['from_trip']
        to_trip_id = item['to_trip']
        cost_val = item['cost']
        i_index = trip_id_to_index[from_trip_id]
        j_index = trip_id_to_index[to_trip_id]
        compat_arcs.append((i_index, j_index, cost_val))
        
    in_compat = [[] for _ in range(n_trips)]
    out_compat = [[] for _ in range(n_trips)]
    for (i, j, cost_val) in compat_arcs:
        out_compat[i].append(j)
        in_compat[j].append(i)
        
    model = gp.Model()
    model.setParam("Threads", 1)
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', args.time_limit)
    
    x = {}
    
    for d in range(m_depots):
        for i in range(n_trips):
            from_node = n_trips + d
            to_node = i
            cost_arc = depot_start_arcs.get((d, i), 1000000)
            var = model.addVar(vtype=GRB.BINARY, name=f"s_{d}_{i}")
            x[(d, from_node, to_node)] = (var, cost_arc)
            
        for i in range(n_trips):
            from_node = i
            to_node = n_trips + m_depots + d
            cost_arc = depot_end_arcs.get((i, d), 1000000)
            var = model.addVar(vtype=GRB.BINARY, name=f"e_{i}_{d}")
            x[(d, from_node, to_node)] = (var, cost_arc)
            
        for (i, j, cost_val) in compat_arcs:
            from_node = i
            to_node = j
            var = model.addVar(vtype=GRB.BINARY, name=f"c_{i}_{j}_{d}")
            x[(d, from_node, to_node)] = (var, cost_val)
            
    obj = gp.LinExpr()
    for key, (var, cost) in x.items():
        obj += cost * var
    model.setObjective(obj, GRB.MINIMIZE)
    
    for i in range(n_trips):
        expr = gp.LinExpr()
        for d in range(m_depots):
            key_start = (d, n_trips + d, i)
            if key_start in x:
                var, _ = x[key_start]
                expr += var
            for j in in_compat[i]:
                key = (d, j, i)
                if key in x:
                    var, _ = x[key]
                    expr += var
        model.addConstr(expr == 1, f"cover_{i}")
        
    for d in range(m_depots):
        for i in range(n_trips):
            lhs = gp.LinExpr()
            key_start = (d, n_trips + d, i)
            if key_start in x:
                var, _ = x[key_start]
                lhs += var
            for j in in_compat[i]:
                key = (d, j, i)
                if key in x:
                    var, _ = x[key]
                    lhs += var
                    
            rhs = gp.LinExpr()
            key_end = (d, i, n_trips + m_depots + d)
            if key_end in x:
                var, _ = x[key_end]
                rhs += var
            for j in out_compat[i]:
                key = (d, i, j)
                if key in x:
                    var, _ = x[key]
                    rhs += var
                    
            model.addConstr(lhs == rhs, f"flow_{d}_{i}")
            
    for d in range(m_depots):
        expr = gp.LinExpr()
        for i in range(n_trips):
            key = (d, n_trips + d, i)
            if key in x:
                var, _ = x[key]
                expr += var
        model.addConstr(expr <= vehicle_capacities[d], f"cap_{d}")
        
    if logger:
        def mycallback(model, where):
            if where == GRB.Callback.MIPSOL:
                obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                logger.log(obj)
        model.optimize(mycallback)
    else:
        model.optimize()
        
    if model.SolCount > 0:
        routes = []
        for d in range(m_depots):
            for i in range(n_trips):
                key = (d, n_trips + d, i)
                if key in x:
                    var, _ = x[key]
                    if var.X > 0.5:
                        chain = [i]
                        current = i
                        while True:
                            key_end = (d, current, n_trips + m_depots + d)
                            if key_end in x:
                                var_end, _ = x[key_end]
                                if var_end.X > 0.5:
                                    break
                            found_next = False
                            for j in out_compat[current]:
                                key_arc = (d, current, j)
                                if key_arc in x:
                                    var_arc, _ = x[key_arc]
                                    if var_arc.X > 0.5:
                                        chain.append(j)
                                        current = j
                                        found_next = True
                                        break
                            if not found_next:
                                break
                        trip_ids_chain = [trips[idx]['trip_id'] for idx in chain]
                        routes.append({"depot": d + 1, "trips": trip_ids_chain})
        objective_value = model.ObjVal
        num_vehicles = len(routes)
        solution = {
            "objective_value": float(objective_value),
            "routes": routes,
            "num_vehicles": num_vehicles
        }
    else:
        solution = {
            "objective_value": 0.0,
            "routes": [],
            "num_vehicles": 0
        }
    
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == '__main__':
    main()