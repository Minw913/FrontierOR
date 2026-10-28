import json
import argparse
import time
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

    network = data['network']
    commodities = data['commodities']
    sense = data.get('objective', 'minimize')

    logger = SolutionLogger(args.log_path, sense=sense) if args.log_path else None

    nodes = network['nodes']
    arcs = network['arcs']
    comm_list = commodities['commodity_list']

    arc_list = [(a['from_node'], a['to_node'], a['arc_id'], a['capacity']) for a in arcs]
    arc_dict = {(a['from_node'], a['to_node']): (a['arc_id'], a['capacity']) for a in arcs}

    # Precompute possible paths per commodity using DFS with capacity filtering (only arcs that can carry the demand)
    def find_paths(origin, dest, demand):
        paths = []
        def dfs(current, path):
            if len(path) > len(nodes) * 2:  # prevent cycles
                return
            if current == dest:
                paths.append(path[:])
                return
            for fr, to, aid, cap in arc_list:
                if fr == current and cap >= demand and (to not in path or to == dest):
                    path.append((fr, to, aid))
                    dfs(to, path)
                    path.pop()
        dfs(origin, [])
        return paths

    comm_data = []
    for c in comm_list:
        cid = c['commodity_id']
        o = c['origin']
        d = c['destination']
        dem = c['demand']
        rev = c['revenue']
        art_cost = c.get('artificial_arc_cost', rev)
        paths = find_paths(o, d, dem)
        # Add artificial "path" represented as None
        comm_data.append({
            'cid': cid,
            'o': o,
            'd': d,
            'demand': dem,
            'art_cost': art_cost,
            'paths': paths  # list of lists of (from, to, arc_id)
        })

    # Build Gurobi model
    model = gp.Model("Unsplittable_Routing")
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('Threads', 1)
    model.setParam('MIPGap', 0.01)  # Allow some gap for larger instances
    if sense.lower() == 'maximize':
        model.ModelSense = GRB.MAXIMIZE
    else:
        model.ModelSense = GRB.MINIMIZE

    # Variables: for each commodity, one path choice (including artificial)
    x = {}
    for idx, c in enumerate(comm_data):
        # Artificial choice
        x[(idx, -1)] = model.addVar(vtype=GRB.BINARY, obj=c['art_cost'], name=f"x_{c['cid']}_art")
        for pidx, _ in enumerate(c['paths']):
            x[(idx, pidx)] = model.addVar(vtype=GRB.BINARY, obj=0.0, name=f"x_{c['cid']}_p{pidx}")

    # Each commodity chooses exactly one option
    for idx, c in enumerate(comm_data):
        model.addConstr(gp.quicksum(x[(idx, pidx)] for pidx in range(-1, len(c['paths']))) == 1,
                        name=f"choose_one_{c['cid']}")

    # Capacity constraints
    arc_usage = {}
    for a in arc_list:
        fr, to, aid, cap = a
        arc_usage[aid] = []
        for idx, c in enumerate(comm_data):
            dem = c['demand']
            for pidx, path in enumerate(c['paths']):
                if any((fr2 == fr and to2 == to) for fr2, to2, _ in path):
                    arc_usage[aid].append((dem, x[(idx, pidx)]))
        if arc_usage[aid]:
            model.addConstr(gp.quicksum(d * var for d, var in arc_usage[aid]) <= cap,
                            name=f"cap_{aid}")

    # Objective is sum of artificial costs (minimize lost revenue)
    model.setObjective(gp.quicksum(var * c['art_cost'] * x[(i, -1)] for i, c in enumerate(comm_data) for var in [1]))

    # Callback to log incumbent solutions
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if logger:
                # Respect sense
                if sense.lower() == 'maximize':
                    logger.log(-obj)  # because we model as minimize lost revenue
                else:
                    logger.log(obj)

    start_time = time.time()
    model.optimize(callback)

    # Extract solution
    objective_value = model.ObjVal
    if sense.lower() == 'maximize':
        objective_value = -objective_value  # convert back if we flipped

    solution_commodities = []
    for idx, c in enumerate(comm_data):
        cid = c['cid']
        chosen = None
        for pidx in range(-1, len(c['paths'])):
            if x[(idx, pidx)].X > 0.5:
                chosen = pidx
                break
        rejected = (chosen == -1)
        path_arcs = []
        if not rejected:
            path = c['paths'][chosen]
            for fr, to, aid in path:
                path_arcs.append({
                    "from": fr,
                    "to": to,
                    "arc_id": aid
                })
        solution_commodities.append({
            "commodity_id": cid,
            "rejected": rejected,
            "path_arcs": path_arcs
        })

    # Write solution
    solution = {
        "objective_value": float(objective_value),
        "commodities": solution_commodities
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

    # Final log of best solution
    if logger:
        logger.log(float(objective_value))

if __name__ == "__main__":
    main()