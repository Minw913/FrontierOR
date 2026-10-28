import json
import argparse
import time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    num_nodes = instance['num_nodes']
    depot = instance['depot']
    arcs = instance['arcs']  # list of [i,j,cost]
    profitable_arcs = instance['profitable_arcs']  # list of [i,j,profit]
    VI_nodes = instance['VI_nodes']
    strong_inc = instance['strong_incompatibilities']
    weak_inc = instance['weak_incompatibilities']

    # Map from (i,j) to index for profitable arcs
    prof_idx = {}
    prof_list = []
    for k, (i, j, p) in enumerate(profitable_arcs):
        prof_idx[(i, j)] = k
        prof_list.append((i, j, p))

    # All arcs indexed
    arc_list = [(a[0], a[1], a[2]) for a in arcs]
    arc_idx = {(a[0], a[1]): idx for idx, a in enumerate(arc_list)}

    # VI node mapping
    vi_map = {node: idx for idx, node in enumerate(VI_nodes)}
    num_vi = len(VI_nodes)

    # Build model
    model = gp.Model("HaulerRouting")
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('Threads', 1)
    model.setParam('MIPGap', 0.01)
    model.setParam('LogToConsole', 0)

    # Variables
    # x_a: number of traversals of arc a (integer)
    x = {}
    for idx, (i, j, c) in enumerate(arc_list):
        x[idx] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"x_{i}_{j}")

    # y_k: whether profitable arc k is served (binary)
    y = {}
    for k in range(len(profitable_arcs)):
        y[k] = model.addVar(vtype=GRB.BINARY, name=f"y_{k}")

    # z_v: whether VI node v is active (binary)
    z = {}
    for v in range(num_vi):
        z[v] = model.addVar(vtype=GRB.BINARY, name=f"z_{v}")

    # w_e: whether weak incompatibility e is overridden (binary)
    w = {}
    for e in range(len(weak_inc)):
        w[e] = model.addVar(vtype=GRB.BINARY, name=f"w_{e}")

    # Objective: sum p*y - sum c*x - sum penalty*w
    profit_term = gp.quicksum(prof_list[k][2] * y[k] for k in range(len(prof_list)))
    travel_term = gp.quicksum(arc_list[a][2] * x[a] for a in range(len(arc_list)))
    penalty_term = gp.quicksum(weak_inc[e][2] * w[e] for e in range(len(weak_inc)))
    model.setObjective(profit_term - travel_term - penalty_term, GRB.MAXIMIZE)

    # Flow conservation
    for v in range(num_nodes):
        inflow = gp.quicksum(x[a] for a in range(len(arc_list)) if arc_list[a][1] == v)
        outflow = gp.quicksum(x[a] for a in range(len(arc_list)) if arc_list[a][0] == v)
        model.addConstr(inflow == outflow, name=f"flow_{v}")

    # y implies traversal
    for k in range(len(prof_list)):
        i, j, _ = prof_list[k]
        a_idx = arc_idx[(i, j)]
        model.addConstr(y[k] <= x[a_idx], name=f"serve_{k}")

    # z activation: if any profitable arc from VI node is served, z=1
    for v_idx, v_node in enumerate(VI_nodes):
        outgoing_prof = [k for k, (i, j, p) in enumerate(prof_list) if i == v_node]
        if outgoing_prof:
            model.addConstr(gp.quicksum(y[k] for k in outgoing_prof) <= 1000 * z[v_idx], name=f"z_activate_{v_idx}")
            model.addConstr(z[v_idx] <= gp.quicksum(y[k] for k in outgoing_prof), name=f"z_force_{v_idx}")

    # Strong incompatibilities
    for pair in strong_inc:
        u = vi_map[pair[0]]
        v = vi_map[pair[1]]
        model.addConstr(z[u] + z[v] <= 1, name=f"strong_{u}_{v}")

    # Weak incompatibilities
    weak_map = {}
    for e_idx, (u_node, v_node, pen) in enumerate(weak_inc):
        u = vi_map[u_node]
        v = vi_map[v_node]
        weak_map[(u, v)] = e_idx
        weak_map[(v, u)] = e_idx
        model.addConstr(z[u] + z[v] <= 1 + w[e_idx], name=f"weak_{u}_{v}")

    # Connectivity constraints (subtour elimination for served profitable arcs)
    # We use MTZ-like or flow-based connectivity. Since it's a directed graph and we have to respect the tour connectivity
    # when a profitable arc is inside a subset, we need to ensure connectivity.
    # A standard way is to use additional flow variables for connectivity from depot.
    # To keep model size reasonable, we introduce a continuous flow variable for each arc indicating if it carries "connection" flow.

    # Let f_a be the flow on arc a for connectivity, f_a >=0
    f = {}
    for a in range(len(arc_list)):
        f[a] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=num_nodes, name=f"f_{a}")

    M = num_nodes + 1
    # Each served profitable arc must "demand" 1 unit of connection from depot
    total_served = gp.quicksum(y[k] for k in range(len(prof_list)))
    model.addConstr(gp.quicksum(f[a] for a in range(len(arc_list)) if arc_list[a][0] == depot) == total_served,
                    name="depot_outflow")

    for v in range(num_nodes):
        if v == depot:
            continue
        inflow_f = gp.quicksum(f[a] for a in range(len(arc_list)) if arc_list[a][1] == v)
        outflow_f = gp.quicksum(f[a] for a in range(len(arc_list)) if arc_list[a][0] == v)
        # Node balance: inflow - outflow = sum of y for profitable arcs ending at v ? No.
        # Better way: each served profitable arc "requires" one unit of flow to be "covered" by a path from depot.
        # A common formulation for orienteering/prize-collecting with connectivity is:
        # for every node v != depot, inflow_f(v) >= sum y for profitable arcs entering v (or something).
        # More precisely, we can assign each served profitable arc a demand of 1 at its head or use cut-set like.

        # Let's use a cut-set inspired but with node potentials or flow per served arc.
        # A practical formulation for directed case with subtour elimination for profitable arcs:
        # We send one unit of flow from depot to each "served profitable arc" by attaching demand to the tail of profitable arcs or head.

        # Attach demand to the starting node of each served profitable arc.
        # If a profitable arc is served from u, then u (if not depot) needs to receive 1 unit of "connection" flow from depot.
        # This ensures all active starting points are connected via the tour to depot.
        # Since tour is Eulerian (in=out), and strongly connected component must link to depot if profitable inside.

        demand = {}
        for v in range(num_nodes):
            demand[v] = gp.quicksum(y[k] for k, (i, j, p) in enumerate(prof_list) if i == v and i != depot)

        for v in range(num_nodes):
            if v == depot:
                continue
            inflow_f = gp.quicksum(f[a] for a in range(len(arc_list)) if arc_list[a][1] == v)
            outflow_f = gp.quicksum(f[a] for a in range(len(arc_list)) if arc_list[a][0] == v)
            model.addConstr(inflow_f - outflow_f == demand.get(v, 0), name=f"connect_balance_{v}")

        # Capacity on flow: flow on arc cannot exceed number of traversals * M, but since we want connection only when used
        for a in range(len(arc_list)):
            i, j, _ = arc_list[a]
            model.addConstr(f[a] <= (num_nodes) * x[a], name=f"connect_capacity_{a}")

    # Also need to make sure if there is a served profitable arc inside a set, the cut must be crossed.
    # The above should cover it because each served arc's tail (if not depot) has a demand that must be satisfied by flow from depot,
    # and flow only travels on used arcs.

    # If a profitable arc is from depot, its "demand" is at depot, which is source, so no extra.

    model.update()

    # Callback for logging incumbent solutions
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if logger:
                logger.log(float(obj))

    # Optimize
    start_time = time.time()
    model.optimize(callback)

    # Extract solution
    if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED, GRB.SOLUTION_LIMIT]:
        objective_value = model.objVal if model.objVal is not None else 0.0

        # served arcs
        served = []
        for k in range(len(prof_list)):
            if y[k].X > 0.5:
                i, j, p = prof_list[k]
                a_idx = arc_idx[(i, j)]
                c = arc_list[a_idx][2]
                served.append({
                    "from": int(i),
                    "to": int(j),
                    "profit": float(p),
                    "cost": float(c)
                })

        # tour arcs (those with x > 0.5)
        tour = []
        for a_idx in range(len(arc_list)):
            count = x[a_idx].X
            if count > 0.5:
                i, j, c = arc_list[a_idx]
                tour.append({
                    "from": int(i),
                    "to": int(j),
                    "count": int(round(count)),
                    "cost": float(c)
                })

        solution = {
            "objective_value": float(objective_value),
            "served_arcs": served,
            "tour_arcs": tour
        }

        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
    else:
        # Infeasible or no solution - write empty
        solution = {
            "objective_value": 0.0,
            "served_arcs": [],
            "tour_arcs": []
        }
        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()