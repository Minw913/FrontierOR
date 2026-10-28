import sys
import json
import math
import argparse
import gurobipy as gp
from gurobipy import GRB

try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, required=False)
    args = parser.parse_args()

    logger = None
    if args.log_path and SolutionLogger:
        logger = SolutionLogger(args.log_path, sense="minimize")

    with open(args.instance_path, 'r') as f:
        data = json.load(f)

    T = data['parameters']['T']
    W = data['parameters']['W']
    L_param = data['parameters']['L']
    source_id = data['source_item_id']
    sink_id = data['sink_item_id']

    real_items = [item for item in data['items'] if item['type'] == 'treatment']
    item_w = {item['item_id']: item['weight'] for item in real_items}
    
    nodes = [item['item_id'] for item in data['items']]
    
    # Bellman-Ford algorithm to compute earliest sequence times (E_dist)
    E_dist = {v: -math.inf for v in nodes}
    E_dist[source_id] = 0

    for _ in range(len(nodes)):
        changed = False
        for arc in data['arcs']:
            u = arc['from']
            v = arc['to']
            l = arc['lag']
            if E_dist[u] != -math.inf and E_dist[u] + l > E_dist[v]:
                E_dist[v] = E_dist[u] + l
                changed = True
        if not changed:
            break

    # Bellman-Ford backwards pass for latest sequence times (L_dist)
    L_dist = {v: math.inf for v in nodes}
    L_dist[sink_id] = T + 1

    for _ in range(len(nodes)):
        changed = False
        for arc in data['arcs']:
            u = arc['from']
            v = arc['to']
            l = arc['lag']
            if L_dist[v] != math.inf and L_dist[v] - l < L_dist[u]:
                L_dist[u] = L_dist[v] - l
                changed = True
        if not changed:
            break

    # Clean up bounds into integers
    E = {}
    L = {}
    for v in nodes:
        if v == source_id:
            E[v] = 0
            L[v] = 0
        elif v == sink_id:
            E[v] = T + 1
            L[v] = T + 1
        else:
            E[v] = int(E_dist[v]) if E_dist[v] != -math.inf else 1
            L[v] = int(L_dist[v]) if L_dist[v] != math.inf else T

    # Determine maximum theoretically required bins for each time period
    max_bins_time = {}
    for t in range(1, T + 1):
        items_t = [i['item_id'] for i in real_items if E[i['item_id']] <= t <= L[i['item_id']]]
        b_max = len(items_t)
        if L_param != 'infinity':
            b_max = min(b_max, int(L_param))
        max_bins_time[t] = b_max

    # Setup Optimization Model
    model = gp.Model("SchedulingCapacitatedBins")
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('Threads', 1)
    model.setParam('MIPFocus', 1)  # Prioritize finding high quality feasible integer solutions 

    # Variables formulation
    # x[i, t]: 1 if item i is scheduled at period t
    x = {}
    for i in real_items:
        iid = i['item_id']
        for t in range(E[iid], L[iid] + 1):
            x[iid, t] = model.addVar(vtype=GRB.BINARY, name=f"x_{iid}_{t}")

    # y[t, b]: 1 if bin b at period t is used
    y = {}
    for t in range(1, T + 1):
        for b in range(max_bins_time[t]):
            y[t, b] = model.addVar(vtype=GRB.BINARY, name=f"y_{t}_{b}")

    # z[i, t, b]: 1 if item i is packed into bin b at period t
    z = {}
    for t in range(1, T + 1):
        items_t = [i['item_id'] for i in real_items if E[i['item_id']] <= t <= L[i['item_id']]]
        b_max = max_bins_time[t]
        for b in range(b_max):
            for iid in items_t:
                z[iid, t, b] = model.addVar(vtype=GRB.BINARY, name=f"z_{iid}_{t}_{b}")

    # Constraints Definitions
    # 1. Start-time unique assignments
    for i in real_items:
        iid = i['item_id']
        model.addConstr(gp.quicksum(x[iid, t] for t in range(E[iid], L[iid] + 1)) == 1, name=f"assign_{iid}")

    # 2. Assignment mappings & capacity mappings per time per bin
    for t in range(1, T + 1):
        items_t = [i['item_id'] for i in real_items if E[i['item_id']] <= t <= L[i['item_id']]]
        b_max = max_bins_time[t]
        
        # Link x and z
        for iid in items_t:
            model.addConstr(x[iid, t] == gp.quicksum(z[iid, t, b] for b in range(b_max)), name=f"link_{iid}_{t}")

        # Total bin capacities vs maximum potential space
        for b in range(b_max):
            model.addConstr(
                gp.quicksum(item_w[iid] * z[iid, t, b] for iid in items_t) <= W * y[t, b],
                name=f"cap_{t}_{b}"
            )
            
        # Optional implied boundaries for faster convergence
        if len(items_t) > 0 and b_max > 0:
            model.addConstr(
                W * gp.quicksum(y[t, b] for b in range(b_max)) >= gp.quicksum(item_w[iid] * x[iid, t] for iid in items_t),
                name=f"lb_bins_{t}"
            )

        # Break symmetrical bin equivalence
        if b_max > 1:
            for b in range(b_max - 1):
                model.addConstr(y[t, b] >= y[t, b+1], name=f"sym_{t}_{b}")

    # 3. Directed timing logic
    T_var = {}
    for i in real_items:
        iid = i['item_id']
        T_var[iid] = gp.quicksum(t * x[iid, t] for t in range(E[iid], L[iid] + 1))

    for arc in data['arcs']:
        u = arc['from']
        v = arc['to']
        l = arc['lag']
        # Apply strict timing to real actionable items only
        if u in item_w and v in item_w:
            # 3.1 Linear explicit logic representation constraints
            model.addConstr(T_var[v] >= T_var[u] + l, name=f"prec_simple_{u}_{v}")
            # 3.2 Step integer tightening
            for q in range(E[v], L[v] + 1):
                lhs = gp.quicksum(x[v, tau] for tau in range(E[v], q + 1))
                upper_tau = min(L[u], q - l)
                if upper_tau >= E[u]:
                    rhs = gp.quicksum(x[u, tau] for tau in range(E[u], upper_tau + 1))
                else:
                    rhs = 0
                model.addConstr(lhs <= rhs, name=f"prec_step_{u}_{v}_{q}")

    # Objective logic setting
    model.setObjective(gp.quicksum(y[t, b] for t in range(1, T + 1) for b in range(max_bins_time[t])), GRB.MINIMIZE)

    # Callback routine
    def mip_callback(model, where):
        if where == gp.GRB.Callback.MIPSOL:
            obj = model.cbGet(gp.GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj)

    # Perform structural solve cycle
    model.optimize(mip_callback)

    # Solution Extraction Formatter
    assignments = []
    active_bp = {}
    obj_val = 0.0

    if model.SolCount > 0:
        for i in real_items:
            iid = i['item_id']
            for t in range(E[iid], L[iid] + 1):
                if x[iid, t].X > 0.5:
                    assigned_t = t
                    assigned_b = None
                    for b in range(max_bins_time[t]):
                        if z[iid, t, b].X > 0.5:
                            assigned_b = b
                            break
                    
                    if assigned_b is not None:
                        w = item_w[iid]
                        assignments.append({
                            "item_id": iid,
                            "bin": assigned_b + 1,  # 1-indexed to keep representation clean
                            "period": assigned_t,
                            "weight": w
                        })
                        
                        key = (assigned_t, assigned_b)
                        if key not in active_bp:
                            active_bp[key] = {
                                "bin": assigned_b + 1,
                                "period": assigned_t,
                                "items": [],
                                "total_weight": 0
                            }
                        active_bp[key]["items"].append(iid)
                        active_bp[key]["total_weight"] += w

        active_bin_periods = list(active_bp.values())
        obj_val = len(active_bin_periods)
        
        for abp in active_bin_periods:
            abp["items"].sort()
            
        active_bin_periods.sort(key=lambda item: (item["period"], item["bin"]))

    else:
        active_bin_periods = []

    ans = {
        "objective_value": float(obj_val),
        "assignments": assignments,
        "active_bin_periods": active_bin_periods,
        "num_active_bin_periods": len(active_bin_periods)
    }

    with open(args.solution_path, 'w') as f:
        json.dump(ans, f, indent=2)

if __name__ == "__main__":
    main()