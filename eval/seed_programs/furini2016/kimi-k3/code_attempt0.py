import argparse
import json
import time
from collections import defaultdict

from solution_logger import SolutionLogger

REACH_LIMIT = 50000      # max panel dim for exact reachable-sum computation
DP_GRID_CAP = 90         # grid cap for the DP fast path
MIP_GRID_CAP = 65        # grid cap for the MIP
VARIABLE_CAP = 400000    # max cut variables before further grid reduction


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", required=True)
    ap.add_argument("--solution_path", required=True)
    ap.add_argument("--time_limit", type=int, required=True)
    ap.add_argument("--log_path", default=None)
    return ap.parse_args()


def read_instance(path):
    with open(path) as f:
        data = json.load(f)
    L = int(data["panel"]["length"])
    W = int(data["panel"]["width"])
    items = []
    for it in data.get("items", []):
        items.append({
            "id": int(it["id"]),
            "length": int(it["length"]),
            "width": int(it["width"]),
            "profit": int(it["profit"]),
            "copies": int(it["copies"]),
        })
    return L, W, items


def build_solution_dict(items, yvals):
    sel = []
    total = 0
    for i, it in enumerate(items):
        c = int(round(yvals[i]))
        if c > 0:
            sel.append({
                "item_id": it["id"],
                "plate_dims": [int(it["length"]), int(it["width"])],
                "copies": int(c),
            })
            total += it["profit"] * c
    return {"objective_value": float(total), "items_selected": sel}


def compute_full_coords(dim_vals, cap):
    """All positive reachable sums of item dims (unlimited repetition) up to cap,
    plus item dims and the panel dim itself."""
    B = min(cap, REACH_LIMIT)
    reach = bytearray(B + 1)
    reach[0] = 1
    vals = sorted(set(v for v in dim_vals if 1 <= v <= B))
    for s in range(B + 1):
        if reach[s]:
            for v in vals:
                ns = s + v
                if ns <= B:
                    reach[ns] = 1
    coords = set(s for s in range(1, B + 1) if reach[s])
    coords.update(v for v in dim_vals if 1 <= v <= cap)
    coords.add(cap)
    return sorted(coords)


def reduce_coords(coords, dim_vals, cap, grid_cap):
    """Reduce coordinate set to at most grid_cap entries, keeping item dims and panel."""
    if len(coords) <= grid_cap:
        return coords
    essential = sorted(set(v for v in dim_vals if 1 <= v <= cap) | {cap})
    ess_set = set(essential)
    if len(essential) >= grid_cap:
        freq = defaultdict(int)
        for d in dim_vals:
            if 1 <= d <= cap:
                freq[d] += 1
        ranked = sorted(ess_set, key=lambda d: (-freq[d], d))
        keep = ranked[:grid_cap - 1] + [cap]
        return sorted(set(keep))
    others = [c for c in coords if c not in ess_set]
    keep = essential + others[:grid_cap - len(essential)]
    return sorted(set(keep))


def solve_unconstrained(L, W, items, XL, YW):
    """DP for the unconstrained (unlimited copies) guillotine problem on the grid."""
    nx, ny = len(XL), len(YW)
    xidx = {v: i for i, v in enumerate(XL)}
    yidx = {v: i for i, v in enumerate(YW)}
    best_item = {}
    for it in items:
        key = (it["length"], it["width"])
        if key not in best_item or it["profit"] > best_item[key][0]:
            best_item[key] = (it["profit"], it["id"])
    vcuts = []
    for i, l in enumerate(XL):
        opts = []
        for ai in range(i):
            opts.append((ai, xidx.get(l - XL[ai], -1)))
        vcuts.append(opts)
    hcuts = []
    for j, w in enumerate(YW):
        opts = []
        for bj in range(j):
            opts.append((bj, yidx.get(w - YW[bj], -1)))
        hcuts.append(opts)
    f = [[0] * ny for _ in range(nx)]
    choice = [[None] * ny for _ in range(nx)]
    for i in range(nx):
        l = XL[i]
        fi = f[i]
        ci = choice[i]
        vc = vcuts[i]
        for j in range(ny):
            w = YW[j]
            best = 0
            ch = None
            bi = best_item.get((l, w))
            if bi is not None:
                best = bi[0]
                ch = (0, bi[1])
            for ai, ri in vc:
                cand = f[ai][j]
                if ri >= 0:
                    cand += f[ri][j]
                if cand > best:
                    best = cand
                    ch = (1, ai, ri)
            hc = hcuts[j]
            for bj, rj in hc:
                cand = fi[bj]
                if rj >= 0:
                    cand += fi[rj]
                if cand > best:
                    best = cand
                    ch = (2, bj, rj)
            fi[j] = best
            ci[j] = ch
    return f, choice, xidx, yidx


def reconstruct(L, W, choice, xidx, yidx, XL, YW):
    counts = {}
    stack = [(L, W)]
    while stack:
        l, w = stack.pop()
        i = xidx.get(l)
        j = yidx.get(w)
        if i is None or j is None:
            continue
        ch = choice[i][j]
        if ch is None:
            continue
        if ch[0] == 0:
            counts[ch[1]] = counts.get(ch[1], 0) + 1
        elif ch[0] == 1:
            a = XL[ch[1]]
            stack.append((a, w))
            stack.append((l - a, w))
        else:
            b = YW[ch[1]]
            stack.append((l, b))
            stack.append((l, w - b))
    return counts


def generate_cut_data(items, XL, YW, L, W):
    """Generate cut variables (with pruning) for the flow MIP on the given grid."""
    XLset = set(XL)
    YWset = set(YW)
    sorted_by_len = sorted(items, key=lambda it: it["length"])
    minw = {}
    idx = 0
    cur = None
    for l in XL:
        while idx < len(sorted_by_len) and sorted_by_len[idx]["length"] <= l:
            wv = sorted_by_len[idx]["width"]
            cur = wv if cur is None else (wv if wv < cur else cur)
            idx += 1
        minw[l] = cur

    def can(l, w):
        mw = minw.get(l)
        return mw is not None and mw <= w

    if not can(L, W):
        return None
    tracked = [(l, w) for l in XL for w in YW if can(l, w)]
    items_matching = {}
    for ii, it in enumerate(items):
        items_matching.setdefault((it["length"], it["width"]), []).append(ii)
    h_keys = []
    v_keys = []
    for (l, w) in tracked:
        for a in XL:
            if a >= l:
                break
            r = l - a
            if can(a, w) or (r in XLset and can(r, w)):
                h_keys.append((l, w, a))
        for b in YW:
            if b >= w:
                break
            r = w - b
            if can(l, b) or (r in YWset and can(l, r)):
                v_keys.append((l, w, b))
    return h_keys, v_keys, tracked, items_matching, can, XLset, YWset


def solve_mip(L, W, items, data, time_limit, logger, start_time, best_so_far):
    import gurobipy as gp
    from gurobipy import GRB
    h_keys, v_keys, tracked, items_matching, can, XLset, YWset = data
    model = gp.Model("guillotine")
    model.Params.Seed = 0
    model.Params.MIPGap = 1e-4
    model.Params.NumericFocus = 0
    model.Params.Threads = 1
    model.Params.OutputFlag = 0
    h_vars = model.addVars(h_keys, vtype=GRB.INTEGER, lb=0.0)
    v_vars = model.addVars(v_keys, vtype=GRB.INTEGER, lb=0.0)
    y_vars = model.addVars(len(items), vtype=GRB.INTEGER, lb=0.0)
    for i, it in enumerate(items):
        y_vars[i].UB = float(it["copies"])
    # Conservation: produced + init >= consumed + kept   (per plate size)
    terms_coeff = defaultdict(list)
    terms_var = defaultdict(list)
    for key, var in h_vars.items():
        l, w, a = key
        terms_var[(l, w)].append(var)
        terms_coeff[(l, w)].append(-1.0)
        r = l - a
        if a == r:
            if can(a, w):
                terms_var[(a, w)].append(var)
                terms_coeff[(a, w)].append(2.0)
        else:
            if can(a, w):
                terms_var[(a, w)].append(var)
                terms_coeff[(a, w)].append(1.0)
            if (r in XLset) and can(r, w):
                terms_var[(r, w)].append(var)
                terms_coeff[(r, w)].append(1.0)
    for key, var in v_vars.items():
        l, w, b = key
        terms_var[(l, w)].append(var)
        terms_coeff[(l, w)].append(-1.0)
        r = w - b
        if b == r:
            if can(l, b):
                terms_var[(l, b)].append(var)
                terms_coeff[(l, b)].append(2.0)
        else:
            if can(l, b):
                terms_var[(l, b)].append(var)
                terms_coeff[(l, b)].append(1.0)
            if (r in YWset) and can(l, r):
                terms_var[(l, r)].append(var)
                terms_coeff[(l, r)].append(1.0)
    for p, iis in items_matching.items():
        for ii in iis:
            terms_var[p].append(y_vars[ii])
            terms_coeff[p].append(-1.0)
    plates_with_constr = set(terms_var.keys()) | {(L, W)}
    for p in plates_with_constr:
        initval = 1.0 if p == (L, W) else 0.0
        expr = gp.LinExpr()
        expr.addTerms(terms_coeff[p], terms_var[p])
        model.addConstr(expr >= -initval)
    obj = gp.LinExpr()
    for i, it in enumerate(items):
        obj.addTerms(float(it["profit"]), y_vars[i])
    model.setObjective(obj, GRB.MAXIMIZE)
    y_list = [y_vars[i] for i in range(len(items))]
    state = {"best": float(best_so_far)}

    def cb(m, where):
        if where == GRB.Callback.MIPSOL:
            try:
                objval = m.cbGet(GRB.Callback.MIPSOL_OBJ)
            except Exception:
                return
            if objval > state["best"] + 1e-9:
                state["best"] = objval
                if logger is not None:
                    try:
                        yvals = m.cbGetSolution(y_list)
                        sol = build_solution_dict(items, yvals)
                        logger.log_solution(float(objval), sol)
                    except Exception:
                        pass

    remaining = max(1, int(time_limit - (time.time() - start_time)))
    model.Params.TimeLimit = remaining
    model.optimize(cb)
    if model.SolCount > 0:
        yvals = [y_vars[i].X for i in range(len(items))]
        return build_solution_dict(items, yvals)
    return None


def write_solution(path, sol):
    with open(path, "w") as f:
        json.dump(sol, f, indent=2)


def main():
    args = parse_args()
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None
    start_time = time.time()
    best_sol = {"objective_value": 0.0, "items_selected": []}
    try:
        L, W, all_items = read_instance(args.instance_path)
        items = [it for it in all_items
                 if it["length"] <= L and it["width"] <= W and it["copies"] > 0]
        if items:
            # Baseline feasible solution: one copy of the most profitable item.
            bi = max(items, key=lambda it: it["profit"])
            best_sol = {
                "objective_value": float(bi["profit"]),
                "items_selected": [{
                    "item_id": bi["id"],
                    "plate_dims": [bi["length"], bi["width"]],
                    "copies": 1,
                }],
            }
            if logger:
                logger.log_solution(best_sol["objective_value"], best_sol)
            item_lengths = [it["length"] for it in items]
            item_widths = [it["width"] for it in items]
            full_X = compute_full_coords(item_lengths, L)
            full_Y = compute_full_coords(item_widths, W)
            # --- DP fast path (unconstrained); optimal if copy limits don't bind ---
            dp_feasible = False
            dpX = reduce_coords(full_X, item_lengths, L, DP_GRID_CAP)
            dpY = reduce_coords(full_Y, item_widths, W, DP_GRID_CAP)
            dp_ops = len(dpX) * len(dpY) * (len(dpX) + len(dpY))
            if dp_ops <= 3_000_000:
                try:
                    f, choice, xidx, yidx = solve_unconstrained(L, W, items, dpX, dpY)
                    counts = reconstruct(L, W, choice, xidx, yidx, dpX, dpY)
                    if all(counts.get(it["id"], 0) <= it["copies"] for it in items):
                        yvals = [counts.get(it["id"], 0) for it in items]
                        sol = build_solution_dict(items, yvals)
                        dp_feasible = True
                        if sol["objective_value"] > best_sol["objective_value"]:
                            best_sol = sol
                            if logger:
                                logger.log_solution(best_sol["objective_value"], best_sol)
                except Exception:
                    dp_feasible = False
            # --- Flow-based MIP for the constrained problem ---
            if not dp_feasible:
                mip_cap = MIP_GRID_CAP
                mipX = reduce_coords(full_X, item_lengths, L, mip_cap)
                mipY = reduce_coords(full_Y, item_widths, W, mip_cap)
                data = generate_cut_data(items, mipX, mipY, L, W)
                while (data is not None
                       and len(data[0]) + len(data[1]) > VARIABLE_CAP
                       and mip_cap > 20):
                    mip_cap = max(20, mip_cap - 15)
                    mipX = reduce_coords(full_X, item_lengths, L, mip_cap)
                    mipY = reduce_coords(full_Y, item_widths, W, mip_cap)
                    data = generate_cut_data(items, mipX, mipY, L, W)
                if data is not None:
                    remaining = args.time_limit - (time.time() - start_time)
                    if remaining > 1:
                        try:
                            mip_sol = solve_mip(L, W, items, data, args.time_limit,
                                                logger, start_time,
                                                best_sol["objective_value"])
                            if (mip_sol is not None
                                    and mip_sol["objective_value"] > best_sol["objective_value"]):
                                best_sol = mip_sol
                        except Exception:
                            pass
    except Exception:
        pass
    write_solution(args.solution_path, best_sol)


if __name__ == "__main__":
    main()