import argparse
import json
import time
import math
import heapq
import sys
import gc
import os

try:
    from solution_logger import SolutionLogger
except Exception:
    SolutionLogger = None

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception:
    gp = None
    GRB = None

OUTPUT_TOL = 1e-7
EPS = 1e-9

PATTERNS = {
    "z[e]": "binary design variable for edge e",
    "x[k,r,e]": "flow of commodity k in scenario r on edge e",
    "tf[r,e]": "total flow in scenario r on edge e",
}


def safe_float(x, default=0.0):
    try:
        v = float(x)
        if math.isnan(v):
            return default
        if math.isinf(v):
            return 1e100 if v > 0 else -1e100
        return v
    except Exception:
        return default


def safe_log(logger, obj):
    if logger is None:
        return
    try:
        logger.log(float(obj))
    except Exception:
        pass


def safe_log_solution(logger, obj, sol):
    if logger is None:
        return
    try:
        logger.log_solution(float(obj), sol)
    except Exception:
        pass


def write_solution(path, sol):
    if not path:
        return
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w") as f:
        json.dump(sol, f)


def load_instance(path):
    with open(path, "r") as f:
        data = json.load(f)
    data["edges"] = [tuple(e) for e in data.get("edges", [])]
    return data


def preprocess(inst):
    N = int(inst.get("num_nodes", 0))
    K = int(inst.get("num_commodities", 0))
    S = int(inst.get("num_scenarios", 0))
    E = int(inst.get("num_edges", 0))

    edges = inst.get("edges", [])
    out_edges = [[] for _ in range(N)]
    in_edges = [[] for _ in range(N)]
    has_incident = [False] * N

    for e, uv in enumerate(edges):
        if len(uv) < 2:
            continue
        u, v = int(uv[0]), int(uv[1])
        if 0 <= u < N and 0 <= v < N:
            out_edges[u].append(e)
            in_edges[v].append(e)
            has_incident[u] = True
            has_incident[v] = True

    demands = inst.get("demands", [])
    active_pairs = []
    scenario_pairs = [[] for _ in range(S)]
    scenario_supply = [0.0] * S

    for r in range(S):
        total_supply = 0.0
        for k in range(K):
            if r < len(demands) and k < len(demands[r]):
                dem = demands[r][k]
            else:
                dem = [0.0] * N

            has_nonzero = False
            pos_supply = 0.0
            for i in range(min(N, len(dem))):
                val = safe_float(dem[i], 0.0)
                if val > EPS:
                    pos_supply += val
                if abs(val) > EPS:
                    has_nonzero = True

            if has_nonzero:
                p = len(active_pairs)
                active_pairs.append((k, r))
                scenario_pairs[r].append(p)

            total_supply += pos_supply

        scenario_supply[r] = total_supply

    caps = []
    for c in inst.get("capacities", []):
        cap = safe_float(c, 0.0)
        if cap < 0.0:
            cap = 0.0
        caps.append(cap)
    while len(caps) < E:
        caps.append(0.0)

    flow_costs = [safe_float(x, 0.0) for x in inst.get("flow_costs", [])]
    while len(flow_costs) < E:
        flow_costs.append(0.0)

    min_fc = min(flow_costs) if flow_costs else 0.0
    use_supply_bound = min_fc >= -1e-8
    gamma = safe_float(inst.get("gamma", 1.0), 1.0)
    if gamma <= 0.0:
        gamma = 1.0
    neg_bound = gamma * max(0.0, -min_fc) * max(1, K) + 1.0

    max_k_per_scenario = max((len(x) for x in scenario_pairs), default=0)

    return {
        "N": N,
        "K": K,
        "S": S,
        "E": E,
        "out_edges": out_edges,
        "in_edges": in_edges,
        "has_incident": has_incident,
        "active_pairs": active_pairs,
        "scenario_pairs": scenario_pairs,
        "scenario_supply": scenario_supply,
        "caps": caps,
        "flow_costs": flow_costs,
        "use_supply_bound": use_supply_bound,
        "neg_bound": neg_bound,
        "max_k_per_scenario": max_k_per_scenario,
    }


def build_solution_from_values(inst, active_pairs, z_vals, x_vals, tf_vals):
    E = int(inst.get("num_edges", 0)) if inst else 0
    S = int(inst.get("num_scenarios", 0)) if inst else 0
    gamma = safe_float(inst.get("gamma", 1.0), 1.0) if inst else 1.0
    if gamma <= 0.0:
        gamma = 1.0

    invS = 1.0 / S if S > 0 else 0.0
    quad_coeff = 1.0 / (2.0 * gamma) if gamma > 0 else 0.0

    construction_costs = inst.get("construction_costs", []) if inst else []
    flow_costs = inst.get("flow_costs", []) if inst else []

    model_variables = {}
    z_out = {}
    obj = 0.0

    if z_vals is None:
        z_vals = []

    for e in range(E):
        zv = z_vals[e] if e < len(z_vals) else 0
        zi = 1 if zv is not None and zv > 0.5 else 0
        z_out[str(e)] = zi
        if zi:
            cc = safe_float(construction_costs[e], 0.0) if e < len(construction_costs) else 0.0
            obj += cc
            model_variables[f"z[{e}]"] = 1

    if tf_vals is None:
        tf_vals = []

    for idx, val in enumerate(tf_vals):
        if E <= 0 or S <= 0 or idx >= S * E:
            break
        r = idx // E
        e = idx % E
        if e >= E:
            continue
        if e >= len(z_vals) or z_vals[e] <= 0.5:
            continue
        val = safe_float(val, 0.0)
        if val > OUTPUT_TOL:
            fc = safe_float(flow_costs[e], 0.0) if e < len(flow_costs) else 0.0
            if invS != 0.0:
                obj += invS * (fc * val + quad_coeff * val * val)
            model_variables[f"tf[{r},{e}]"] = float(val)

    if x_vals:
        for idx, val in enumerate(x_vals):
            if E <= 0 or idx >= len(active_pairs) * E:
                continue
            val = safe_float(val, 0.0)
            if val <= OUTPUT_TOL:
                continue
            p = idx // E
            e = idx % E
            if e >= len(z_vals) or z_vals[e] <= 0.5:
                continue
            if p >= len(active_pairs):
                continue
            k, r = active_pairs[p]
            model_variables[f"x[{k},{r},{e}]"] = float(val)

    sol = {
        "objective_value": float(obj),
        "model_variables": model_variables,
        "z": z_out,
        "model_variable_name_patterns": PATTERNS,
    }
    return float(obj), sol


def default_zero_solution(inst, active_pairs):
    E = int(inst.get("num_edges", 0)) if inst else 0
    c0 = int(inst.get("c_0", 0)) if inst else 0
    construction_costs = inst.get("construction_costs", []) if inst else []

    z = [0] * E
    if c0 > 0 and E > 0:
        neg = []
        for e in range(E):
            c = safe_float(construction_costs[e], 0.0) if e < len(construction_costs) else 0.0
            if c < -1e-9:
                neg.append((c, e))
        neg.sort()
        for _, e in neg[: min(c0, E)]:
            z[e] = 1

    return build_solution_from_values(inst, active_pairs, z, [], [])


def dijkstra_to_destination(dest, rev_adj, weights, N):
    parent = [-1] * N
    if N <= 0 or dest < 0 or dest >= N:
        return parent

    inf = 1e100
    dist = [inf] * N
    dist[dest] = 0.0
    heap = [(0.0, dest)]

    while heap:
        d, v = heapq.heappop(heap)
        if d > dist[v] + 1e-12:
            continue
        for u, e in rev_adj[v]:
            nd = d + weights[e]
            if nd < dist[u] - 1e-12:
                dist[u] = nd
                parent[u] = e
                heapq.heappush(heap, (nd, u))

    return parent


def generate_candidates(inst, pre, max_candidates=5):
    N = pre["N"]
    K = pre["K"]
    S = pre["S"]
    E = pre["E"]

    if E == 0:
        return [[]]

    c0 = int(inst.get("c_0", 0))
    c0 = max(0, min(c0, E))

    edges = inst.get("edges", [])
    flow_costs = inst.get("flow_costs", [])
    edge_lengths = inst.get("edge_lengths", [])
    construction_costs = inst.get("construction_costs", [])

    weights = []
    for e in range(E):
        fc = safe_float(flow_costs[e], 0.0) if e < len(flow_costs) else 0.0
        el = safe_float(edge_lengths[e], 0.0) if e < len(edge_lengths) else 0.0
        if fc < 0.0 or math.isnan(fc):
            w = max(0.0, el) + 1e-9
        else:
            w = max(0.0, fc) + 1e-6 * max(0.0, el) + 1e-9
        weights.append(w)

    rev_adj = [[] for _ in range(N)]
    for e, uv in enumerate(edges):
        if len(uv) < 2:
            continue
        u, v = int(uv[0]), int(uv[1])
        if 0 <= u < N and 0 <= v < N:
            rev_adj[v].append((u, e))

    destinations = inst.get("commodity_destinations", [])
    parents = []
    for k in range(K):
        dest = int(destinations[k]) if k < len(destinations) else 0
        parents.append(dijkstra_to_destination(dest, rev_adj, weights, N))

    invS = 1.0 / S if S > 0 else 0.0
    demands = inst.get("demands", [])

    avg_supply = [[0.0] * N for _ in range(K)]
    for r in range(S):
        if r >= len(demands):
            continue
        for k in range(K):
            if k >= len(demands[r]):
                continue
            dem = demands[r][k]
            for i in range(min(N, len(dem))):
                b = safe_float(dem[i], 0.0)
                if b > EPS:
                    avg_supply[k][i] += b * invS

    usage = [0.0] * E
    for k in range(K):
        parent = parents[k] if k < len(parents) else [-1] * N
        dest = int(destinations[k]) if k < len(destinations) else 0
        for i in range(N):
            s = avg_supply[k][i]
            if s <= OUTPUT_TOL:
                continue
            cur = i
            steps = 0
            while cur != dest and steps <= N:
                if cur < 0 or cur >= N:
                    break
                e = parent[cur] if cur < len(parent) else -1
                if e < 0 or e >= E:
                    break
                usage[e] += s
                if e >= len(edges) or len(edges[e]) < 2:
                    break
                cur = int(edges[e][1])
                steps += 1

    def key_cost(e):
        c = safe_float(construction_costs[e], 0.0) if e < len(construction_costs) else 0.0
        den = c + 1.0
        if den <= 1e-6:
            den = 1e-6
        return (usage[e] / den, -c)

    def key_usage(e):
        c = safe_float(construction_costs[e], 0.0) if e < len(construction_costs) else 0.0
        return (usage[e], -c)

    candidates = []
    seen = set()

    def add_set(s):
        s = set(int(e) for e in s if 0 <= int(e) < E)
        if c0 == 0:
            s = set()
        elif len(s) > c0:
            s = set(sorted(s, key=key_cost, reverse=True)[:c0])

        z = [0] * E
        for e in s:
            z[e] = 1
        tup = tuple(z)
        if tup not in seen:
            seen.add(tup)
            candidates.append(z)

    union = set(e for e in range(E) if usage[e] > OUTPUT_TOL)
    if len(union) <= c0:
        add_set(union)

    fill = set(union)
    order = sorted([e for e in range(E) if e not in fill], key=key_cost, reverse=True)
    for e in order:
        if len(fill) >= c0:
            break
        fill.add(e)
    add_set(fill)

    top_cost = set(sorted(range(E), key=key_cost, reverse=True)[:c0])
    add_set(top_cost)

    top_usage = set(sorted(range(E), key=key_usage, reverse=True)[:c0])
    add_set(top_usage)

    cheap = set(
        sorted(
            range(E),
            key=lambda e: safe_float(construction_costs[e], 0.0)
            if e < len(construction_costs)
            else 0.0,
        )[:c0]
    )
    add_set(cheap)

    if not candidates:
        add_set(set(range(min(c0, E))))

    return candidates[:max_candidates]


def set_model_params(model, time_alloc=None):
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    if time_alloc is not None:
        model.Params.TimeLimit = max(0.01, float(time_alloc))


def link_cap_value(cap_e, supply, use_supply_bound, neg_bound):
    cap_e = safe_float(cap_e, 0.0)
    if cap_e < 0.0:
        cap_e = 0.0
    supply = safe_float(supply, 0.0)
    if supply < 0.0:
        supply = 0.0

    if use_supply_bound:
        if cap_e > 1e30:
            m = supply
        else:
            m = min(cap_e, supply)
    else:
        if cap_e > 1e30:
            m = supply + neg_bound
        else:
            m = cap_e

    if m < 0.0:
        m = 0.0
    return m


def add_flow_conservation(model, x_vars, active_pairs, inst, pre):
    N = pre["N"]
    E = pre["E"]
    edges = inst.get("edges", [])
    demands = inst.get("demands", [])
    has_incident = pre["has_incident"]

    for p, (k, r) in enumerate(active_pairs):
        base = p * E
        if r < len(demands) and k < len(demands[r]):
            dem = demands[r][k]
        else:
            dem = [0.0] * N

        if E == 0:
            for i in range(N):
                b = safe_float(dem[i], 0.0) if i < len(dem) else 0.0
                if abs(b) > EPS:
                    model.addConstr(0.0 == b)
            continue

        exprs = [gp.LinExpr() for _ in range(N)]
        for e, uv in enumerate(edges):
            if len(uv) < 2:
                continue
            u, v = int(uv[0]), int(uv[1])
            if not (0 <= u < N and 0 <= v < N):
                continue
            var = x_vars[base + e]
            exprs[u] += var
            exprs[v] -= var

        for i in range(N):
            b = safe_float(dem[i], 0.0) if i < len(dem) else 0.0
            if not has_incident[i] and abs(b) <= EPS:
                continue
            model.addConstr(exprs[i] == b)


def add_tf_equalities(model, x_vars, tf_vars, scenario_pairs, E):
    for r, plist in enumerate(scenario_pairs):
        if not plist:
            continue
        base_tf = r * E
        if len(plist) == 1:
            base_x = plist[0] * E
            for e in range(E):
                model.addConstr(tf_vars[base_tf + e] == x_vars[base_x + e])
        else:
            for e in range(E):
                model.addConstr(
                    tf_vars[base_tf + e]
                    == gp.quicksum(x_vars[p * E + e] for p in plist)
                )


def solve_full(inst, pre, env, time_alloc, logger, best_obj):
    if gp is None or time_alloc <= 0:
        return None, None

    N = pre["N"]
    K = pre["K"]
    S = pre["S"]
    E = pre["E"]
    active_pairs = pre["active_pairs"]
    scenario_pairs = pre["scenario_pairs"]
    scenario_supply = pre["scenario_supply"]
    caps = pre["caps"]
    use_supply_bound = pre["use_supply_bound"]
    neg_bound = pre["neg_bound"]

    P = len(active_pairs)
    total_x = P * E
    tf_count = S * E

    m = gp.Model(env=env)
    set_model_params(m, time_alloc)
    m.Params.MIPFocus = 1

    if best_obj is not None and math.isfinite(best_obj):
        m.Params.Cutoff = best_obj + 1e-4 * max(1.0, abs(best_obj))

    z_vars = m.addVars(E, vtype=GRB.BINARY, name="z") if E > 0 else {}
    x_vars = m.addVars(total_x, lb=0.0, name="x") if total_x > 0 else {}

    use_tf = pre["max_k_per_scenario"] > 2
    tf_vars = {}

    if use_tf:
        tf_ub = []
        for r in range(S):
            if scenario_pairs[r] and scenario_supply[r] > EPS:
                sup = scenario_supply[r]
                row = [
                    link_cap_value(caps[e], sup, use_supply_bound, neg_bound)
                    for e in range(E)
                ]
            else:
                row = [0.0] * E
            tf_ub.extend(row)
        tf_vars = m.addVars(tf_count, lb=0.0, ub=tf_ub, name="tf") if tf_count > 0 else {}

    c0 = int(inst.get("c_0", 0))
    if E > 0 and c0 < E:
        m.addConstr(gp.quicksum(z_vars[e] for e in range(E)) <= c0)

    for r, plist in enumerate(scenario_pairs):
        if not plist:
            continue
        sup = scenario_supply[r]
        base_tf = r * E
        for e in range(E):
            M = link_cap_value(caps[e], sup, use_supply_bound, neg_bound)
            if M <= 0.0:
                if not use_tf:
                    if len(plist) == 1:
                        m.addConstr(x_vars[plist[0] * E + e] <= 0.0)
                    else:
                        m.addConstr(
                            gp.quicksum(x_vars[p * E + e] for p in plist) <= 0.0
                        )
            else:
                if use_tf:
                    m.addConstr(tf_vars[base_tf + e] <= M * z_vars[e])
                else:
                    if len(plist) == 1:
                        m.addConstr(x_vars[plist[0] * E + e] <= M * z_vars[e])
                    else:
                        m.addConstr(
                            gp.quicksum(x_vars[p * E + e] for p in plist)
                            <= M * z_vars[e]
                        )

    if use_tf:
        add_tf_equalities(m, x_vars, tf_vars, scenario_pairs, E)

    add_flow_conservation(m, x_vars, active_pairs, inst, pre)

    construction_costs = [safe_float(x, 0.0) for x in inst.get("construction_costs", [])]
    while len(construction_costs) < E:
        construction_costs.append(0.0)

    flow_costs = [safe_float(x, 0.0) for x in inst.get("flow_costs", [])]
    while len(flow_costs) < E:
        flow_costs.append(0.0)

    invS = 1.0 / S if S > 0 else 0.0
    gamma = safe_float(inst.get("gamma", 1.0), 1.0)
    if gamma <= 0.0:
        gamma = 1.0
    quad_coeff = 1.0 / (2.0 * gamma * S) if (S > 0 and gamma > 0.0) else 0.0

    lin = gp.LinExpr()
    for e in range(E):
        lin += construction_costs[e] * z_vars[e]

    if use_tf:
        if invS != 0.0:
            for r, plist in enumerate(scenario_pairs):
                if not plist:
                    continue
                base = r * E
                for e in range(E):
                    lin += (flow_costs[e] * invS) * tf_vars[base + e]

        if quad_coeff != 0.0:
            quad = gp.QuadExpr()
            for r, plist in enumerate(scenario_pairs):
                if not plist:
                    continue
                base = r * E
                for e in range(E):
                    t = tf_vars[base + e]
                    quad += quad_coeff * t * t
            m.setObjective(lin + quad, GRB.MINIMIZE)
        else:
            m.setObjective(lin, GRB.MINIMIZE)
    else:
        if invS != 0.0:
            for p in range(P):
                base = p * E
                for e in range(E):
                    lin += (flow_costs[e] * invS) * x_vars[base + e]

        if quad_coeff != 0.0:
            quad = gp.QuadExpr()
            for r, plist in enumerate(scenario_pairs):
                if not plist:
                    continue
                for e in range(E):
                    if len(plist) == 1:
                        var = x_vars[plist[0] * E + e]
                        quad += quad_coeff * var * var
                    else:
                        expr = gp.quicksum(x_vars[p * E + e] for p in plist)
                        quad += quad_coeff * expr * expr
            m.setObjective(lin + quad, GRB.MINIMIZE)
        else:
            m.setObjective(lin, GRB.MINIMIZE)

    m.update()

    if logger is not None:
        def callback(model, where):
            if where == GRB.Callback.MIPSOL:
                try:
                    sol_obj = float(model.cbGet(GRB.Callback.MIPSOL_OBJ))
                except Exception:
                    return
                if sol_obj < model._cb_best - 1e-6:
                    model._cb_best = sol_obj
                    if sol_obj < best_obj - 1e-6:
                        safe_log(logger, sol_obj)

        m._cb_best = (
            float(best_obj)
            if best_obj is not None and math.isfinite(best_obj)
            else float("inf")
        )
        m.optimize(callback)
    else:
        m.optimize()

    if m.SolCount > 0:
        z_vals = []
        for e in range(E):
            try:
                val = z_vars[e].X
            except Exception:
                val = 0.0
            z_vals.append(1 if val > 0.5 else 0)

        x_vals = []
        if total_x > 0:
            x_list = [x_vars[i] for i in range(total_x)]
            x_vals = m.getAttr("X", x_list)

        if use_tf:
            tf_vals = []
            if tf_count > 0:
                tf_list = [tf_vars[i] for i in range(tf_count)]
                tf_vals = m.getAttr("X", tf_list)
        else:
            tf_vals = [0.0] * (S * E)
            if E > 0:
                for idx, val in enumerate(x_vals):
                    val = safe_float(val, 0.0)
                    if val > OUTPUT_TOL:
                        p = idx // E
                        e = idx % E
                        if e < E and z_vals[e] == 1:
                            r = active_pairs[p][1]
                            tf_vals[r * E + e] += val

        obj, sol = build_solution_from_values(inst, active_pairs, z_vals, x_vals, tf_vals)
        del m
        gc.collect()
        return obj, sol

    del m
    gc.collect()
    return None, None


class ContinuousEvaluator:
    def __init__(self, inst, pre, env, quadratic=True):
        self.inst = inst
        self.pre = pre
        self.N = pre["N"]
        self.K = pre["K"]
        self.S = pre["S"]
        self.E = pre["E"]
        self.active_pairs = pre["active_pairs"]
        self.scenario_pairs = pre["scenario_pairs"]
        self.scenario_supply = pre["scenario_supply"]
        self.caps = pre["caps"]
        self.use_supply_bound = pre["use_supply_bound"]
        self.neg_bound = pre["neg_bound"]

        self.P = len(self.active_pairs)
        self.total_x = self.P * self.E
        self.tf_count = self.S * self.E

        self.m = gp.Model(env=env)
        set_model_params(self.m, None)

        self.x = (
            self.m.addVars(self.total_x, lb=0.0, name="x")
            if self.total_x > 0
            else {}
        )

        tf_ub = []
        for r in range(self.S):
            if self.scenario_pairs[r] and self.scenario_supply[r] > EPS:
                sup = self.scenario_supply[r]
                row = [
                    link_cap_value(self.caps[e], sup, self.use_supply_bound, self.neg_bound)
                    for e in range(self.E)
                ]
            else:
                row = [0.0] * self.E
            tf_ub.extend(row)

        self.tf = (
            self.m.addVars(self.tf_count, lb=0.0, ub=tf_ub, name="tf")
            if self.tf_count > 0
            else {}
        )

        add_tf_equalities(self.m, self.x, self.tf, self.scenario_pairs, self.E)
        add_flow_conservation(self.m, self.x, self.active_pairs, inst, pre)

        flow_costs = [safe_float(x, 0.0) for x in inst.get("flow_costs", [])]
        while len(flow_costs) < self.E:
            flow_costs.append(0.0)

        invS = 1.0 / self.S if self.S > 0 else 0.0
        gamma = safe_float(inst.get("gamma", 1.0), 1.0)
        if gamma <= 0.0:
            gamma = 1.0

        lin = gp.LinExpr()
        if invS != 0.0:
            for r, plist in enumerate(self.scenario_pairs):
                if not plist:
                    continue
                base = r * self.E
                for e in range(self.E):
                    lin += (flow_costs[e] * invS) * self.tf[base + e]

        if quadratic and self.S > 0 and gamma > 0.0:
            coeff = 1.0 / (2.0 * gamma * self.S)
            quad = gp.QuadExpr()
            for r, plist in enumerate(self.scenario_pairs):
                if not plist:
                    continue
                base = r * self.E
                for e in range(self.E):
                    t = self.tf[base + e]
                    quad += coeff * t * t
            self.m.setObjective(lin + quad, GRB.MINIMIZE)
        else:
            self.m.setObjective(lin, GRB.MINIMIZE)

        self.m.update()

    def set_z(self, z_vals):
        for r, plist in enumerate(self.scenario_pairs):
            if not plist:
                continue
            sup = self.scenario_supply[r]
            base = r * self.E
            for e in range(self.E):
                if e < len(z_vals) and z_vals[e] > 0.5:
                    ub = link_cap_value(
                        self.caps[e], sup, self.use_supply_bound, self.neg_bound
                    )
                else:
                    ub = 0.0
                self.tf[base + e].setAttr("UB", ub)
        self.m.update()

    def solve(self, time_alloc):
        if time_alloc <= 0:
            return None
        self.m.reset()
        self.m.Params.TimeLimit = max(0.01, float(time_alloc))
        self.m.optimize()

        if self.m.SolCount > 0:
            x_vals = []
            tf_vals = []
            if self.total_x > 0:
                x_vals = self.m.getAttr("X", [self.x[i] for i in range(self.total_x)])
            if self.tf_count > 0:
                tf_vals = self.m.getAttr("X", [self.tf[i] for i in range(self.tf_count)])
            return x_vals, tf_vals

        return None


def run_continuous_heuristic(
    inst, pre, env, max_time, logger, best_obj, best_sol, quadratic=False
):
    if gp is None or max_time <= 2.0:
        return best_obj, best_sol

    candidates = generate_candidates(inst, pre, max_candidates=5)
    if not candidates:
        return best_obj, best_sol

    try:
        evaluator = ContinuousEvaluator(inst, pre, env, quadratic=quadratic)
    except Exception:
        return best_obj, best_sol

    start = time.time()
    per = max(2.0, max_time / max(1, len(candidates)))

    for z_vals in candidates:
        elapsed = time.time() - start
        rem = max_time - elapsed
        if rem < 2.0:
            break

        try:
            evaluator.set_z(z_vals)
            res = evaluator.solve(min(per, rem - 1.0))
        except Exception:
            res = None

        if res is not None:
            x_vals, tf_vals = res
            obj, sol = build_solution_from_values(
                inst, pre["active_pairs"], z_vals, x_vals, tf_vals
            )
            if obj < best_obj - 1e-8:
                best_obj = obj
                best_sol = sol
                safe_log_solution(logger, obj, sol)

    del evaluator
    gc.collect()
    return best_obj, best_sol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=float, default=60.0)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    start_time = time.time()

    try:
        inst = load_instance(args.instance_path)
    except Exception:
        inst = {}

    pre = preprocess(inst)

    logger = None
    if args.log_path and SolutionLogger is not None:
        try:
            logger = SolutionLogger(args.log_path, sense="minimize")
        except Exception:
            logger = None

    active_pairs = pre["active_pairs"]
    E = pre["E"]
    N = pre["N"]
    S = pre["S"]

    if not active_pairs:
        obj, sol = default_zero_solution(inst, active_pairs)
        safe_log_solution(logger, obj, sol)
        write_solution(args.solution_path, sol)
        return

    time_limit = max(0.0, float(args.time_limit))
    if time_limit <= 0.0 or gp is None:
        obj, sol = default_zero_solution(inst, active_pairs)
        safe_log_solution(logger, obj, sol)
        write_solution(args.solution_path, sol)
        return

    env = None
    try:
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()
    except Exception:
        env = None

    best_obj = float("inf")
    best_sol = None

    total_x = len(active_pairs) * E
    size_score = total_x + len(active_pairs) * N + S * E

    # Fast linear continuous heuristic for an initial feasible incumbent.
    if (
        env is not None
        and time_limit >= 30.0
        and total_x <= 80000
        and size_score <= 600000
    ):
        rem = time_limit - (time.time() - start_time)
        if rem > 10.0:
            heur_time = min(30.0, max(5.0, 0.15 * rem))
            best_obj, best_sol = run_continuous_heuristic(
                inst,
                pre,
                env,
                heur_time,
                logger,
                best_obj,
                best_sol,
                quadratic=False,
            )

    # Full mixed-integer convex quadratic model when size is manageable.
    full_attempted = False
    if env is not None and total_x <= 180000 and size_score <= 800000:
        rem = time_limit - (time.time() - start_time)
        if rem > 5.0:
            full_time = rem - 5.0 if rem > 30.0 else max(1.0, 0.8 * rem)
            try:
                obj, sol = solve_full(inst, pre, env, full_time, logger, best_obj)
                full_attempted = True
                if obj is not None and obj < best_obj - 1e-8:
                    best_obj = obj
                    best_sol = sol
                    safe_log_solution(logger, obj, sol)
            except Exception as ex:
                sys.stderr.write(f"Full model failed: {ex}\n")
                gc.collect()

    # Fallback / improvement if full model was not solved or found nothing.
    rem = time_limit - (time.time() - start_time)
    if env is not None and rem > 10.0 and ((not full_attempted) or best_sol is None):
        quad = (
            (not full_attempted)
            and total_x <= 80000
            and size_score <= 400000
            and rem > 60.0
        )
        try:
            best_obj, best_sol = run_continuous_heuristic(
                inst,
                pre,
                env,
                rem - 2.0,
                logger,
                best_obj,
                best_sol,
                quadratic=quad,
            )
        except Exception as ex:
            sys.stderr.write(f"Heuristic failed: {ex}\n")
            gc.collect()

    if best_sol is None:
        best_obj, best_sol = default_zero_solution(inst, active_pairs)
        safe_log_solution(logger, best_obj, best_sol)

    write_solution(args.solution_path, best_sol)


if __name__ == "__main__":
    main()