import argparse
import json
import time
import math

import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

VARIANT = "integrated_primal_ucgna"


def clean(x, binary=False):
    if binary:
        return 1.0 if x >= 0.5 else 0.0
    if x is None:
        return 0.0
    if abs(x) < 1e-7:
        return 0.0
    return float(x)


class UCGNASolver:
    def __init__(self, inst, time_limit, logger):
        self.inst = inst
        self.logger = logger
        self.time_limit = max(1, int(time_limit))
        self.t0 = time.time()
        self.best = None
        self._parse()
        self._build_model()

    # ---------------- parsing ----------------
    def _parse(self):
        inst = self.inst
        self.beta = float(inst["objective_weight_beta"])
        self.T = int(inst["time_periods"])
        stress = inst.get("stress_parameters", {})
        self.eta_p = float(stress.get("eta_p", 1.0))
        self.eta_g = float(stress.get("eta_g", 1.0))

        en = inst["electricity_network"]
        self.buses = en["buses"]
        self.lines = en["lines"]
        self.gens = inst["generators"]["generators"]
        gn = inst["gas_network"]
        self.junctions = gn["junctions"]
        self.connections = gn["connections"]
        self.zones = gn["pricing_zones"]
        self.gmin = float(gn["min_gas_price_mmBtu"])
        self.gmax = float(gn["max_gas_price_mmBtu"])
        self.junction_zone = {}
        for z in self.zones:
            for j in z["junctions"]:
                self.junction_zone[j] = z["id"]

    # ---------------- model build ----------------
    def _build_model(self):
        T = self.T
        m = gp.Model("ucgna")
        m.Params.Seed = 0
        m.Params.MIPGap = 1e-4
        m.Params.NumericFocus = 0
        m.Params.Threads = 1
        remaining = self.time_limit - (time.time() - self.t0) - 2.0
        m.Params.TimeLimit = max(1.0, remaining)
        self.m = m

        self.o = {}
        self.vp = {}
        self.vm = {}
        self.w = {}
        self.r = {}
        self.phimax = {}
        self.p = {}
        self.se = {}
        self.f = {}
        self.theta = {}
        self.sg = {}
        self.sgs = {}
        self.pisq = {}
        self.phig = {}
        self.lg = {}
        self.qg = {}
        self.gamma = {}
        self.psi = {}
        self.aux_y = {}   # auxiliary McCormick variable psi*o per GFPP

        gfpp_at_junction = {j["id"]: [] for j in self.junctions}
        elec_terms = []
        gas_terms = []

        # ----- generators -----
        for g in self.gens:
            gid = g["id"]
            gname = f"o_{gid}"
            p0 = float(g["initial_gen"])
            minp = float(g["min_power"])
            maxp = float(g["max_power"])
            ru = float(g["ramp_up"])
            rd = float(g["ramp_down"])
            nlc = float(g["no_load_cost"])
            mut = int(g.get("min_up_time", 0))
            mdt = int(g.get("min_down_time", 0))
            istat = int(g["initial_status"])
            act = int(g.get("initial_active_periods", 0))
            inact = int(g.get("initial_inactive_periods", 0))
            tiers = g.get("startup_cost_params", []) or []
            bids = g.get("bids", []) or []
            isgf = bool(g.get("is_gfpp", False))
            gj = g.get("gas_junction", None)

            for t in range(T + 1):
                v = m.addVar(vtype=GRB.BINARY, name=f"o_{gid}_{t}")
                if t == 0:
                    v.LB = v.UB = float(istat)
                self.o[(gid, t)] = v
                v = m.addVar(lb=0.0, ub=maxp if maxp > 0 else max(p0, 0.0),
                             name=f"p_{gid}_{t}")
                if t == 0:
                    v.LB = v.UB = p0
                self.p[(gid, t)] = v
            # fix initial trajectory
            if istat == 1 and mut > 0:
                for t in range(1, min(T, max(0, mut - act)) + 1):
                    self.o[(gid, t)].LB = self.o[(gid, t)].UB = 1.0
            if istat == 0 and mdt > 0:
                for t in range(1, min(T, max(0, mdt - inact)) + 1):
                    self.o[(gid, t)].LB = self.o[(gid, t)].UB = 0.0

            max_tier = 0.0
            for tier in tiers:
                try:
                    max_tier = max(max_tier, float(tier[1]))
                except Exception:
                    pass

            for t in range(1, T + 1):
                self.vp[(gid, t)] = m.addVar(vtype=GRB.BINARY,
                                              name=f"vp_{gid}_{t}")
                self.vm[(gid, t)] = m.addVar(vtype=GRB.BINARY,
                                              name=f"vm_{gid}_{t}")
                self.r[(gid, t)] = m.addVar(lb=0.0, ub=max_tier,
                                             name=f"r_{gid}_{t}")
                # startup/shutdown relation
                m.addConstr(self.vp[(gid, t)] - self.vm[(gid, t)] ==
                            self.o[(gid, t)] - self.o[(gid, t - 1)])
                # startup cost tiers (history window clipped to [0, t-1])
                for tier in tiers:
                    Lw = int(round(float(tier[0])))
                    Cw = float(tier[1])
                    lo = max(0, t - Lw)
                    hist = 0
                    for tau in range(lo, t):
                        hist = hist + self.o[(gid, tau)]
                    m.addConstr(self.r[(gid, t)] >=
                                Cw * self.o[(gid, t)] - Cw * hist)
                elec_terms.append(nlc * self.o[(gid, t)])
                elec_terms.append(self.r[(gid, t)])

            # min up: sum of startups over window <= on-status of last period
            if mut >= 1:
                for t in range(mut, T + 1):
                    expr = 0
                    for tau in range(t - mut + 1, t + 1):
                        expr = expr + self.vp[(gid, tau)]
                    m.addConstr(expr <= self.o[(gid, t)])
            # min down: sum of startups over window <= 1 - status before window
            if mdt >= 1:
                for t in range(mdt, T + 1):
                    expr = 0
                    for tau in range(t - mdt + 1, t + 1):
                        expr = expr + self.vp[(gid, tau)]
                    m.addConstr(expr <= 1 - self.o[(gid, t - mdt)])

            # bids
            for b in bids:
                bid = b["id"]
                smax = float(b["max_amount"])
                price = float(b["price"])
                for t in range(1, T + 1):
                    self.w[(gid, bid, t)] = m.addVar(vtype=GRB.BINARY,
                        name=f"w_{gid}_{bid}_{t}")
                    self.se[(gid, bid, t)] = m.addVar(lb=0.0, ub=smax,
                        name=f"se_{gid}_{bid}_{t}")
                    # selection only if on; output limited by selection
                    m.addConstr(self.w[(gid, bid, t)] <= self.o[(gid, t)])
                    m.addConstr(self.se[(gid, bid, t)] <=
                                smax * self.w[(gid, bid, t)])
                    elec_terms.append(price * self.se[(gid, bid, t)])

            # total output = sum of bids; output bounds; ramping
            for t in range(1, T + 1):
                se_sum = 0
                for b in bids:
                    se_sum = se_sum + self.se[(gid, b["id"], t)]
                pv = self.p[(gid, t)]
                m.addConstr(pv == se_sum)
                ov = self.o[(gid, t)]
                if minp > 1e-9:
                    m.addConstr(pv >= minp * ov)
                m.addConstr(pv <= maxp * ov)
                p_prev = self.p[(gid, t - 1)]
                o_prev = self.o[(gid, t - 1)]
                m.addConstr(pv - p_prev <=
                            ru * o_prev + maxp * self.vp[(gid, t)])
                m.addConstr(p_prev - pv <=
                            rd * ov + minp * self.vm[(gid, t)])

            # GFPP-specific logic
            if isgf:
                if gj is not None:
                    gfpp_at_junction[gj].append(g)
                phi_b = []
                phi_all = self.gmax * float(g.get("max_gas_price_fraction", 1.0))
                for b in bids:
                    th = None
                    for key in ("profitability_threshold", "max_gas_price",
                                "gas_price_threshold"):
                        if key in b and b[key] is not None:
                            th = float(b[key])
                            break
                    phi_b.append(th if th is not None else phi_all)
                K = len(bids)
                for t in range(1, T + 1):
                    # sequential bid activation
                    for k in range(K - 1):
                        b_cur = bids[k]["id"]
                        b_nxt = bids[k + 1]["id"]
                        smax_cur = float(bids[k]["max_amount"])
                        m.addConstr(
                            self.se[(gid, b_cur, t)] >=
                            smax_cur * self.w[(gid, b_nxt, t)])
                    # phimax linear expression
                    px = m.addVar(lb=0.0, ub=max(self.gmax, phi_all, max(phi_b) if phi_b else 0.0),
                                  name=f"phimax_{gid}_{t}")
                    self.phimax[(gid, t)] = px
                    expr = 0
                    for k in range(K):
                        if k < K - 1:
                            bk = bids[k]["id"]
                            bk1 = bids[k + 1]["id"]
                            expr = expr + phi_b[k] * (
                                self.w[(gid, bk, t)] - self.w[(gid, bk1, t)])
                        else:
                            bk = bids[k]["id"]
                            expr = expr + phi_b[k] * self.w[(gid, bk, t)]
                    m.addConstr(px == expr)
                # coupling with zonal price via y = psi * o (McCormick, exact for binary o)
                zone = self.junction_zone.get(gj, None) if gj is not None else None
                if zone is not None:
                    Lp, Up = self.gmin, self.gmax
                    for t in range(1, T + 1):
                        y = m.addVar(lb=min(0.0, Lp), ub=max(0.0, Up),
                                     name=f"__y_{gid}_{t}")
                        self.aux_y[(gid, t)] = y
                        ps = self.psi.get((zone, t))
                        # psi vars created later; use lazy linking list
            # non-GFPP: nothing extra

        # ----- electricity network variables -----
        for line in self.lines:
            lid = line["id"]
            lim = float(line["thermal_limit"])
            B = float(line["susceptance"])
            alim = float(line.get("angle_diff_limit", 1e9))
            for t in range(1, T + 1):
                self.f[(lid, t)] = m.addVar(lb=-lim, ub=lim,
                                             name=f"f_{lid}_{t}")
        for bus in self.buses:
            bid = bus["id"]
            lo = float(bus["voltage_angle_lb"])
            hi = float(bus["voltage_angle_ub"])
            for t in range(1, T + 1):
                self.theta[(bid, t)] = m.addVar(lb=lo, ub=hi,
                                                 name=f"theta_{bid}_{t}")

        # ----- gas network variables -----
        for j in self.junctions:
            jid = j["id"]
            is_src = bool(j.get("is_source", False))
            dem = [self.eta_g * float(d) for d in j["gas_demand_profile"]]
            shedc = float(j.get("demand_shedding_cost", 0.0))
            plo = float(j["pressure_lb_squared"])
            phi = float(j["pressure_ub_squared"])
            n_gf = len(gfpp_at_junction[jid])
            for t in range(1, T + 1):
                dt = dem[t - 1] if t - 1 < len(dem) else dem[-1]
                self.sg[(jid, t)] = m.addVar(lb=0.0,
                                              name=f"sg_{jid}_{t}")
                if not is_src:
                    self.sg[(jid, t)].UB = 0.0
                self.pisq[(jid, t)] = m.addVar(lb=plo, ub=phi,
                                                name=f"pisq_{jid}_{t}")
                self.lg[(jid, t)] = m.addVar(lb=0.0, ub=dt,
                                              name=f"lg_{jid}_{t}")
                self.qg[(jid, t)] = m.addVar(lb=0.0, ub=dt,
                                              name=f"qg_{jid}_{t}")
                self.gamma[(jid, t)] = m.addVar(lb=0.0,
                                                 name=f"gamma_{jid}_{t}")
                if n_gf == 0:
                    self.gamma[(jid, t)].UB = 0.0
                gas_terms.append(shedc * self.qg[(jid, t)])
            if is_src:
                for s in j.get("supply_intervals", []):
                    sid = s["id"]
                    lo = float(s["interval_lb"])
                    hi = float(s["interval_ub"])
                    slope = float(s["slope"])
                    for t in range(1, T + 1):
                        self.sgs[(jid, sid, t)] = m.addVar(lb=lo, ub=hi,
                            name=f"sgs_{jid}_{sid}_{t}")
                        gas_terms.append(slope * self.sgs[(jid, sid, t)])
        for c in self.connections:
            cid = c["id"]
            for t in range(1, T + 1):
                self.phig[(cid, t)] = m.addVar(lb=0.0,
                                                name=f"phig_{cid}_{t}")

        # ----- psi vars -----
        for z in self.zones:
            zid = z["id"]
            for t in range(1, T + 1):
                self.psi[(zid, t)] = m.addVar(lb=self.gmin, ub=self.gmax,
                                               name=f"psi_{zid}_{t}")

        m.update()

        # ----- electrical link / balance constraints -----
        gen_by_bus = {}
        for g in self.gens:
            gen_by_bus.setdefault(g["bus"], []).append(g["id"])
        for line in self.lines:
            lid = line["id"]
            B = float(line["susceptance"])
            alim = float(line.get("angle_diff_limit", 1e9))
            fb = line["from_bus"]
            tb = line["to_bus"]
            for t in range(1, T + 1):
                fv = self.f[(lid, t)]
                d_ang = self.theta[(fb, t)] - self.theta[(tb, t)]
                if alim < 1e8:
                    m.addConstr(d_ang <= alim)
                    m.addConstr(d_ang >= -alim)
                m.addConstr(fv == B * d_ang)
        for bus in self.buses:
            bid = bus["id"]
            dem = [self.eta_p * float(d) for d in bus["demand_profile"]]
            for t in range(1, T + 1):
                dt = dem[t - 1] if t - 1 < len(dem) else dem[-1]
                lhs = 0
                for gid in gen_by_bus.get(bid, []):
                    lhs = lhs + self.p[(gid, t)]
                flow = 0
                for line in self.lines:
                    lid = line["id"]
                    if line["from_bus"] == bid:
                        flow = flow + self.f[(lid, t)]
                    elif line["to_bus"] == bid:
                        flow = flow - self.f[(lid, t)]
                m.addConstr(lhs - dt == flow)

        # ----- gas link / balance constraints -----
        for j in self.junctions:
            jid = j["id"]
            for t in range(1, T + 1):
                nl = 0  # net out - in
                for c in self.connections:
                    cid = c["id"]
                    if c["from_junction"] == jid:
                        nl = nl + self.phig[(cid, t)]
                    elif c["to_junction"] == jid:
                        nl = nl - self.phig[(cid, t)]
                m.addConstr(self.sg[(jid, t)] - self.lg[(jid, t)]
                            - self.gamma[(jid, t)] == nl)
                demv = [self.eta_g * float(d) for d in j["gas_demand_profile"]]
                dt = demv[t - 1] if t - 1 < len(demv) else demv[-1]
                m.addConstr(self.lg[(jid, t)] + self.qg[(jid, t)] == dt)
                if j.get("is_source", False):
                    ss = 0
                    for s in j.get("supply_intervals", []):
                        ss = ss + self.sgs[(jid, s["id"], t)]
                    m.addConstr(self.sg[(jid, t)] == ss)

        # heat-rate consumption (convex relaxation)
        for jid, glist in gfpp_at_junction.items():
            if not glist:
                continue
            for g in glist:
                gid = g["id"]
                H = g.get("heat_rate_coefficients", {}) or {}
                h2 = float(H.get("H_u2") or 0.0)
                h1 = float(H.get("H_u1") or 0.0)
                h0 = float(H.get("H_u0") or 0.0)
                for t in range(1, T + 1):
                    pv = self.p[(gid, t)]
                    ov = self.o[(gid, t)]
                    gm = self.gamma[(jid, t)]
                    try:
                        if h2 > 0:
                            m.addConstr(gm >= h2 * pv * pv + h1 * pv + h0 * ov)
                        else:
                            m.addConstr(gm >= h1 * pv + h0 * ov)
                    except gp.GurobiError:
                        m.addConstr(gm >= h1 * pv + h0 * ov)

        # ----- connection physics -----
        for c in self.connections:
            cid = c["id"]
            fj = c["from_junction"]
            tj = c["to_junction"]
            typ = c.get("type", "pipeline")
            for t in range(1, T + 1):
                fl = self.phig[(cid, t)]
                pf = self.pisq[(fj, t)]
                pt = self.pisq[(tj, t)]
                if typ == "pipeline":
                    Wc = float(c.get("weymouth_factor") or 0.0)
                    try:
                        if Wc > 0:
                            m.addConstr(pf - pt >= Wc * fl * fl)
                        else:
                            m.addConstr(pf >= pt)
                    except gp.GurobiError:
                        m.addConstr(pf >= pt)
                elif typ == "compressor":
                    rl = c.get("compression_ratio_lb")
                    ruq = c.get("compression_ratio_ub")
                    if rl is not None:
                        m.addConstr(pt >= float(rl) ** 2 * pf)
                    if ruq is not None:
                        m.addConstr(pt <= float(ruq) ** 2 * pf)
                else:  # control valve or other type with control ratios
                    rl = c.get("control_ratio_lb")
                    ruq = c.get("control_ratio_ub")
                    if rl is None or ruq is None:
                        # fallback: treat as pipeline-like
                        Wc = float(c.get("weymouth_factor") or 0.0)
                        try:
                            if Wc > 0:
                                m.addConstr(pf - pt >= Wc * fl * fl)
                        except gp.GurobiError:
                            pass
                    else:
                        m.addConstr(pt >= float(rl) ** 2 * pf)
                        m.addConstr(pt <= float(ruq) ** 2 * pf)

        # ----- psi * o coupling (McCormick) -----
        for g in self.gens:
            if not g.get("is_gfpp", False):
                continue
            gid = g["id"]
            gj = g.get("gas_junction", None)
            zone = self.junction_zone.get(gj, None) if gj is not None else None
            if zone is None:
                continue
            Lp, Up = self.gmin, self.gmax
            for t in range(1, T + 1):
                if (gid, t) not in self.phimax:
                    continue
                y = self.aux_y.get((gid, t))
                if y is None:
                    continue
                ps = self.psi[(zone, t)]
                ov = self.o[(gid, t)]
                m.addConstr(y >= Lp * ov)
                m.addConstr(y <= Up * ov)
                m.addConstr(y >= ps - Up * (1 - ov))
                m.addConstr(y <= ps - Lp * (1 - ov))
                m.addConstr(self.phimax[(gid, t)] >= y)

        # ----- objective -----
        obj = self.beta * gp.quicksum(elec_terms) \
              + (1 - self.beta) * gp.quicksum(gas_terms)
        m.setObjective(obj, GRB.MINIMIZE)

        self._allvars = []
        self._allnames = []
        m.update()
        for v in m.getVars():
            if not v.VarName.startswith("__"):
                self._allvars.append(v)
                self._allnames.append(v.VarName)
        self._gen_info = {g["id"]: g for g in self.gens}
        self._jun_info = {j["id"]: j for j in self.junctions}

    # ---------------- solution building ----------------
    def _build_solution(self, vals):
        """vals: dict varname -> value (aligned with self._allvars)."""
        T = self.T
        primary = {}
        binary_prefixes = ("o_", "vp_", "vm_", "w_")
        for name, var in zip(self._allnames, self._allvars):
            is_bin = var.VType == GRB.BINARY
            primary[name] = clean(vals[name], binary=is_bin)

        # recompute objective
        elec = 0.0
        for g in self.gens:
            gid = g["id"]
            nlc = float(g["no_load_cost"])
            for t in range(1, T + 1):
                elec += nlc * primary[f"o_{gid}_{t}"]
                elec += primary.get(f"r_{gid}_{t}", 0.0)
                for b in g.get("bids", []):
                    elec += float(b["price"]) * primary.get(
                        f"se_{gid}_{b['id']}_{t}", 0.0)
        gas = 0.0
        for j in self.junctions:
            jid = j["id"]
            shedc = float(j.get("demand_shedding_cost", 0.0))
            for t in range(1, T + 1):
                gas += shedc * primary.get(f"qg_{jid}_{t}", 0.0)
                for s in j.get("supply_intervals", []):
                    gas += float(s["slope"]) * primary.get(
                        f"sgs_{jid}_{s['id']}_{t}", 0.0)
        objv = self.beta * elec + (1 - self.beta) * gas
        return {
            "objective_value": float(objv),
            "released_problem_variant": VARIANT,
            "objective_weight_beta": float(self.beta),
            "primary_variables": primary,
        }

    # ---------------- solve ----------------
    def solve(self):
        solver_ref = self

        def callback(model, where):
            if where == GRB.Callback.MIPSOL:
                vals_list = model.cbGetSolution(solver_ref._allvars)
                vals = {n: v for n, v in zip(solver_ref._allnames, vals_list)}
                sol = solver_ref._build_solution(vals)
                solver_ref.best = sol
                if solver_ref.logger:
                    solver_ref.logger.log_solution(
                        sol["objective_value"], sol)

        try:
            self.m.optimize(callback)
        except gp.GurobiError:
            pass

        if self.m.SolCount > 0:
            try:
                vals = {n: v.X for n, v in zip(self._allnames, self._allvars)}
                sol = self._build_solution(vals)
                if self.best is None or \
                        sol["objective_value"] < self.best["objective_value"]:
                    self.best = sol
            except Exception:
                pass

        if self.best is None:
            self.best = self._fallback()

        if self.logger:
            self.logger.log_solution(self.best["objective_value"], self.best)
        return self.best

    # ---------------- fallback ----------------
    def _fallback(self):
        T = self.T
        vals = {}
        for g in self.gens:
            gid = g["id"]
            istat = int(g["initial_status"])
            p0 = float(g["initial_gen"])
            minp = float(g["min_power"])
            maxp = float(g["max_power"])
            for t in range(T + 1):
                vals[f"o_{gid}_{t}"] = float(istat)
                vals[f"p_{gid}_{t}"] = p0 if t == 0 else (
                    min(max(p0, minp), maxp) if istat == 1 else 0.0)
            for t in range(1, T + 1):
                vals[f"vp_{gid}_{t}"] = 0.0
                vals[f"vm_{gid}_{t}"] = 0.0
                vals[f"r_{gid}_{t}"] = 0.0
                pt = vals[f"p_{gid}_{t}"]
                rem = pt
                for b in g.get("bids", []):
                    bid = b["id"]
                    amt = min(rem, float(b["max_amount"]))
                    vals[f"se_{gid}_{bid}_{t}"] = amt
                    vals[f"w_{gid}_{bid}_{t}"] = 1.0 if amt > 1e-9 else 0.0
                    rem -= amt
                if g.get("is_gfpp", False):
                    frac = float(g.get("max_gas_price_fraction", 1.0) or 1.0)
                    vals[f"phimax_{gid}_{t}"] = frac * self.gmax
        for line in self.lines:
            lid = line["id"]
            for t in range(1, T + 1):
                vals[f"f_{lid}_{t}"] = 0.0
        for bus in self.buses:
            bid = bus["id"]
            lo = float(bus["voltage_angle_lb"])
            hi = float(bus["voltage_angle_ub"])
            v = 0.0 if lo <= 0.0 <= hi else (lo + hi) / 2
            for t in range(1, T + 1):
                vals[f"theta_{bid}_{t}"] = v
        for j in self.junctions:
            jid = j["id"]
            lo = float(j["pressure_lb_squared"])
            hi = float(j["pressure_ub_squared"])
            for t in range(1, T + 1):
                vals[f"sg_{jid}_{t}"] = 0.0
                vals[f"pisq_{jid}_{t}"] = (lo + hi) / 2
                demv = [self.eta_g * float(d) for d in j["gas_demand_profile"]]
                dt = demv[t - 1] if t - 1 < len(demv) else demv[-1]
                vals[f"lg_{jid}_{t}"] = dt
                vals[f"qg_{jid}_{t}"] = 0.0
                vals[f"gamma_{jid}_{t}"] = 0.0
                if j.get("is_source", False):
                    for s in j.get("supply_intervals", []):
                        vals[f"sgs_{jid}_{s['id']}_{t}"] = float(s["interval_lb"])
        for c in self.connections:
            cid = c["id"]
            for t in range(1, T + 1):
                vals[f"phig_{cid}_{t}"] = 0.0
        for z in self.zones:
            zid = z["id"]
            for t in range(1, T + 1):
                vals[f"psi_{zid}_{t}"] = self.gmin
        return self._build_solution(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance_path", required=True)
    ap.add_argument("--solution_path", required=True)
    ap.add_argument("--time_limit", type=int, required=True)
    ap.add_argument("--log_path", default=None)
    args = ap.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") \
        if args.log_path else None

    with open(args.instance_path, "r") as fh:
        inst = json.load(fh)

    solver = UCGNASolver(inst, args.time_limit, logger)
    sol = solver.solve()

    with open(args.solution_path, "w") as fh:
        json.dump(sol, fh)


if __name__ == "__main__":
    main()