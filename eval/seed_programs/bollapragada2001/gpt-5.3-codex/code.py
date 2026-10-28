import argparse
import json
import math
from collections import defaultdict

import gurobipy as gp
from gurobipy import GRB

from solution_logger import SolutionLogger


def get_direction_index(direction: str, dim: int) -> int:
    d = direction.strip().lower()
    if d.startswith("x"):
        return 0
    if d.startswith("y"):
        return 1
    if d.startswith("z"):
        return 2
    raise ValueError(f"Unknown direction: {direction}")


def extract_group_bar_ids(group_dict):
    # Robust parser for possible key names
    for k in ["bar_ids", "bars", "members", "bar_group"]:
        if k in group_dict and isinstance(group_dict[k], list):
            return group_dict[k]
    # fallback: first list[int] value
    for v in group_dict.values():
        if isinstance(v, list) and all(isinstance(x, int) for x in v):
            return v
    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        inst = json.load(f)

    bars = inst["bars"]
    dofs = inst["degrees_of_freedom"]
    loads_data = inst["loading_conditions"]
    areas = inst["discrete_areas"]

    B = inst["num_bars"]
    D = inst["num_free_dofs"]
    L = inst["num_loading_conditions"]
    K = inst["num_discrete_areas"]
    dim = inst["dimension"]

    E = float(inst["material_properties"]["modulus_of_elasticity"])
    cost_density = float(inst["material_properties"]["cost_density"])

    # Maps
    bar_ids = [b["bar_id"] for b in bars]
    bar_id_to_idx = {bid: i for i, bid in enumerate(bar_ids)}
    dof_ids = [d["dof_id"] for d in dofs]
    dof_id_to_idx = {did: i for i, did in enumerate(dof_ids)}
    load_ids = [ld["load_id"] for ld in loads_data]

    # Bar geometry arrays
    bar_len = [float(b["length"]) for b in bars]
    node_i = [b["node_i"] for b in bars]
    node_j = [b["node_j"] for b in bars]
    dir_cos = [b["direction_cosines"] for b in bars]
    kappa = [E / bar_len[b] for b in range(B)]  # E/L

    # Build equilibrium/compatibility coefficient matrix C[b][dof]
    # C(b,d) = +cos if dof at node_i, -cos if dof at node_j, else 0
    dof_node = [d["node"] for d in dofs]
    dof_dir_idx = [get_direction_index(d["direction"], dim) for d in dofs]

    C = [[0.0] * D for _ in range(B)]
    for b in range(B):
        ni, nj = node_i[b], node_j[b]
        dc = dir_cos[b]
        for d in range(D):
            if dof_node[d] == ni:
                C[b][d] = float(dc[dof_dir_idx[d]])
            elif dof_node[d] == nj:
                C[b][d] = -float(dc[dof_dir_idx[d]])

    # External loads P[l][dof]
    P = [[0.0] * D for _ in range(L)]
    for l_idx, ld in enumerate(loads_data):
        for entry in ld.get("loads", []):
            did = entry["dof_id"]
            if did in dof_id_to_idx:
                P[l_idx][dof_id_to_idx[did]] += float(entry["force"])

    # Stress bounds per bar
    stress_default = inst.get("stress_bounds", {})
    sL_def = float(stress_default.get("lower", -GRB.INFINITY))
    sU_def = float(stress_default.get("upper", GRB.INFINITY))
    sL = [sL_def] * B
    sU = [sU_def] * B
    for sb in inst.get("bar_specific_stress_bounds", []):
        bidx = bar_id_to_idx.get(sb["bar_id"], None)
        if bidx is not None:
            sL[bidx] = float(sb.get("lower", sL[bidx]))
            sU[bidx] = float(sb.get("upper", sU[bidx]))

    # Optional elongation bounds (if present)
    eL = [-GRB.INFINITY] * B
    eU = [GRB.INFINITY] * B
    if "elongation_bounds" in inst:
        eb = inst["elongation_bounds"]
        if eb.get("lower", None) is not None:
            val = float(eb["lower"])
            eL = [max(x, val) for x in eL]
        if eb.get("upper", None) is not None:
            val = float(eb["upper"])
            eU = [min(x, val) for x in eU]
    if "bar_specific_elongation_bounds" in inst:
        for eb in inst["bar_specific_elongation_bounds"]:
            bidx = bar_id_to_idx.get(eb["bar_id"], None)
            if bidx is not None:
                if eb.get("lower", None) is not None:
                    eL[bidx] = max(eL[bidx], float(eb["lower"]))
                if eb.get("upper", None) is not None:
                    eU[bidx] = min(eU[bidx], float(eb["upper"]))

    # Intersect with stress-implied elongation bounds: stress = (E/L)*elong
    for b in range(B):
        e_from_sL = sL[b] / kappa[b] if math.isfinite(sL[b]) else -GRB.INFINITY
        e_from_sU = sU[b] / kappa[b] if math.isfinite(sU[b]) else GRB.INFINITY
        eL[b] = max(eL[b], e_from_sL)
        eU[b] = min(eU[b], e_from_sU)

    # If still unbounded, use conservative big bounds
    BIG_E = 1e6
    for b in range(B):
        if not math.isfinite(eL[b]):
            eL[b] = -BIG_E
        if not math.isfinite(eU[b]):
            eU[b] = BIG_E
        if eL[b] > eU[b]:
            raise ValueError(f"Infeasible bounds for bar {bar_ids[b]}: eL>eU")

    # Displacement bounds
    disp_b = inst.get("displacement_bounds", {})
    uL = disp_b.get("lower", None)
    uU = disp_b.get("upper", None)
    uL = -GRB.INFINITY if uL is None else float(uL)
    uU = GRB.INFINITY if uU is None else float(uU)

    model = gp.Model("discrete_truss")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = float(args.time_limit)
    model.Params.Threads = 1
    model.Params.MIPFocus = 1

    # Variables
    z = model.addVars(B, K, vtype=GRB.BINARY, name="z")
    u = model.addVars(L, D, lb=uL, ub=uU, vtype=GRB.CONTINUOUS, name="u")
    e = model.addVars(B, L, K, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="e")

    # Exactly one area per bar
    for b in range(B):
        model.addConstr(gp.quicksum(z[b, k] for k in range(K)) == 1, name=f"one_area_{b}")

    # Linking groups (same area selection pattern)
    for g in inst.get("linking_groups", []):
        ids = extract_group_bar_ids(g)
        idxs = [bar_id_to_idx[x] for x in ids if x in bar_id_to_idx]
        if len(idxs) >= 2:
            ref = idxs[0]
            for bb in idxs[1:]:
                for k in range(K):
                    model.addConstr(z[bb, k] == z[ref, k], name=f"link_b{bb}_r{ref}_k{k}")

    # e bounds activated by z, and e=0 when not selected
    for b in range(B):
        lb_b, ub_b = eL[b], eU[b]
        for l in range(L):
            for k in range(K):
                model.addConstr(e[b, l, k] >= lb_b * z[b, k], name=f"elb_b{b}_l{l}_k{k}")
                model.addConstr(e[b, l, k] <= ub_b * z[b, k], name=f"eub_b{b}_l{l}_k{k}")

    # Compatibility: C*u = total elongation
    for b in range(B):
        for l in range(L):
            model.addConstr(
                gp.quicksum(C[b][d] * u[l, d] for d in range(D)) ==
                gp.quicksum(e[b, l, k] for k in range(K)),
                name=f"compat_b{b}_l{l}"
            )

    # Equilibrium: sum_b C[b,d] * f[b,l] = P[l,d], with f[b,l]=sum_k kappa*A_k*e[b,l,k]
    for l in range(L):
        for d in range(D):
            model.addConstr(
                gp.quicksum(
                    C[b][d] * gp.quicksum(kappa[b] * float(areas[k]) * e[b, l, k] for k in range(K))
                    for b in range(B)
                ) == P[l][d],
                name=f"eq_l{l}_d{d}"
            )

    # Objective
    model.setObjective(
        gp.quicksum(cost_density * bar_len[b] * float(areas[k]) * z[b, k] for b in range(B) for k in range(K)),
        GRB.MINIMIZE
    )

    # Incumbent logging callback
    callback = None
    if logger is not None:
        best_seen = [float("inf")]

        def cb(m, where):
            if where == GRB.Callback.MIPSOL:
                obj = m.cbGet(GRB.Callback.MIPSOL_OBJ)
                if obj + 1e-9 < best_seen[0]:
                    best_seen[0] = obj
                    logger.log(float(obj))

        callback = cb

    model.optimize(callback)

    if model.SolCount == 0:
        raise RuntimeError("No feasible solution found within time limit.")

    # Extract selected areas
    bar_area_out = []
    chosen_k = [0] * B
    for b in range(B):
        vals = [z[b, k].X for k in range(K)]
        k_star = max(range(K), key=lambda k: vals[k])
        chosen_k[b] = k_star
        bar_area_out.append({
            "bar_id": int(bar_ids[b]),
            "area": float(areas[k_star]),
            "area_index": int(k_star)
        })

    # Displacements output
    disp_out = []
    for l in range(L):
        for d in range(D):
            disp_out.append({
                "dof_id": int(dof_ids[d]),
                "load": int(load_ids[l]),
                "value": float(u[l, d].X)
            })

    # Forces output
    force_out = []
    for b in range(B):
        for l in range(L):
            f_bl = 0.0
            for k in range(K):
                f_bl += kappa[b] * float(areas[k]) * float(e[b, l, k].X)
            force_out.append({
                "bar_id": int(bar_ids[b]),
                "load": int(load_ids[l]),
                "force": float(f_bl)
            })

    sol = {
        "objective_value": float(model.ObjVal),
        "bar_areas": bar_area_out,
        "displacements": disp_out,
        "bar_forces": force_out
    }

    with open(args.solution_path, "w") as f:
        json.dump(sol, f, indent=2)


if __name__ == "__main__":
    main()