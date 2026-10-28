import argparse
import json
import time
import random

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--log_path", default="")
    parser.add_argument("--time_limit", type=int, default=60)
    args = parser.parse_args()

    start_time = time.time()
    deadline = start_time + max(1, args.time_limit)

    with open(args.instance_path, "r") as f:
        data = json.load(f)

    num_customers = int(data["num_customers"])
    num_nodes = int(data["num_nodes"])
    depot = int(data["depot_index"])
    customers = [int(x) for x in data.get("customer_indices", [])]

    params = data.get("parameters", {})
    D = int(params.get("num_drones", 0))

    cap_raw = params.get("drone_battery_capacities", []) if D > 0 else []
    cap = [float(x) for x in cap_raw]
    while len(cap) < D:
        cap.append(0.0)

    C = data["truck_travel_times"]
    E = data.get("drone_battery_consumption", []) if D > 0 else []
    Tmat = data.get("drone_delivery_time", []) if D > 0 else []

    EPS = 1e-7
    virtual_sink = num_nodes
    truck_nodes = [depot] + customers

    options_by_customer = {j: [] for j in customers}
    total_options = 0

    if D > 0:
        for j in customers:
            for i in truck_nodes:
                if i == j:
                    continue
                for d in range(D):
                    try:
                        e = float(E[d][i][j])
                    except Exception:
                        e = float("inf")
                    if e <= cap[d] + 1e-9:
                        try:
                            t = float(Tmat[d][i][j])
                        except Exception:
                            t = 1e100
                        options_by_customer[j].append((d, i, e, t))
                        total_options += 1
            options_by_customer[j].sort(key=lambda x: x[3])

    logger = None
    if args.log_path and SolutionLogger is not None:
        try:
            logger = SolutionLogger(args.log_path, sense="minimize")
        except Exception:
            try:
                logger = SolutionLogger(args.log_path)
            except Exception:
                logger = None

    best = {"obj": float("inf"), "sol": None, "state": None}

    def route_cost(route):
        if not route:
            return 0.0
        cost = 0.0
        for idx in range(len(route) - 1):
            cost += float(C[route[idx]][route[idx + 1]])
        cost += float(C[route[-1]][depot])
        return cost

    def nn_route(nodes):
        if not nodes:
            return [depot]
        unvisited = set(nodes)
        cur = depot
        route = [depot]
        while unvisited:
            nxt = min(unvisited, key=lambda x: float(C[cur][x]))
            route.append(nxt)
            unvisited.remove(nxt)
            cur = nxt
        return route

    def insertion_route(nodes):
        if not nodes:
            return [depot]
        route = [depot]
        remaining = list(nodes)
        while remaining:
            best_inc = None
            best_j = None
            best_pos = None
            for j in remaining:
                for idx in range(len(route)):
                    prev = route[idx]
                    nxt = route[idx + 1] if idx + 1 < len(route) else depot
                    inc = float(C[prev][j]) + float(C[j][nxt]) - float(C[prev][nxt])
                    if best_inc is None or inc < best_inc:
                        best_inc = inc
                        best_j = j
                        best_pos = idx
            if best_j is None:
                break
            route.insert(best_pos + 1, best_j)
            remaining.remove(best_j)
        return route

    def build_initial_route(nodes):
        if not nodes:
            return [depot]
        if len(nodes) <= 80:
            return insertion_route(nodes)
        return nn_route(nodes)

    def load_wait(row):
        if D == 0 or not row:
            return 0.0
        m = max(row)
        return m if m > 0.0 else 0.0

    class State:
        def __init__(self, route, assignment=None):
            self.route = list(route)
            self.travel = route_cost(self.route)
            self.visited = set(self.route)
            self.pos = {node: idx for idx, node in enumerate(self.route)}
            self.loads = [[0.0 for _ in range(D)] for _ in range(num_nodes)]
            self.wait = [0.0 for _ in range(num_nodes)]
            self.battery = [0.0 for _ in range(D)]
            self.dispatch_count = [0 for _ in range(num_nodes)]
            self.assignment = {}
            if assignment:
                for j, di in assignment.items():
                    d, i = di
                    self.add_assignment(j, d, i)

        def rebuild_pos(self):
            self.pos = {node: idx for idx, node in enumerate(self.route)}

        def update_wait(self, i):
            if D == 0:
                self.wait[i] = 0.0
            else:
                m = max(self.loads[i]) if self.loads[i] else 0.0
                self.wait[i] = m if m > 0.0 else 0.0

        def add_assignment(self, j, d, i):
            e = float(E[d][i][j])
            t = float(Tmat[d][i][j])
            self.assignment[j] = (d, i)
            self.battery[d] += e
            self.loads[i][d] += t
            self.dispatch_count[i] += 1
            self.update_wait(i)

        def remove_assignment(self, j):
            d, i = self.assignment.pop(j)
            e = float(E[d][i][j])
            t = float(Tmat[d][i][j])
            self.battery[d] -= e
            self.loads[i][d] -= t
            if abs(self.loads[i][d]) < 1e-9:
                self.loads[i][d] = 0.0
            self.dispatch_count[i] -= 1
            self.update_wait(i)
            return d, i

        def removal_saving(self, k):
            idx = self.pos[k]
            prev = self.route[idx - 1]
            nxt = self.route[idx + 1] if idx + 1 < len(self.route) else depot
            return float(C[prev][k]) + float(C[k][nxt]) - float(C[prev][nxt])

        def remove_node(self, k):
            idx = self.pos[k]
            prev = self.route[idx - 1]
            nxt = self.route[idx + 1] if idx + 1 < len(self.route) else depot
            saving = float(C[prev][k]) + float(C[k][nxt]) - float(C[prev][nxt])
            self.route.pop(idx)
            self.travel -= saving
            self.visited.remove(k)
            self.rebuild_pos()
            return saving

        def insert_after_node(self, node, after_node):
            idx = self.pos[after_node]
            nxt = self.route[idx + 1] if idx + 1 < len(self.route) else depot
            inc = float(C[after_node][node]) + float(C[node][nxt]) - float(C[after_node][nxt])
            self.route.insert(idx + 1, node)
            self.travel += inc
            self.visited.add(node)
            self.rebuild_pos()
            return inc

        def objective(self):
            return float(self.travel + sum(self.wait[i] for i in self.route))

        def clone(self):
            new = State.__new__(State)
            new.route = self.route[:]
            new.travel = self.travel
            new.visited = set(self.visited)
            new.pos = dict(self.pos)
            new.loads = [row[:] for row in self.loads]
            new.wait = self.wait[:]
            new.battery = self.battery[:]
            new.dispatch_count = self.dispatch_count[:]
            new.assignment = dict(self.assignment)
            return new

    def build_solution(state):
        r = state.route
        arcs = []
        if len(r) == 1:
            arcs.append([int(depot), int(virtual_sink)])
        else:
            for a, b in zip(r, r[1:]):
                arcs.append([int(a), int(b)])
            arcs.append([int(r[-1]), int(virtual_sink)])

        waiting = {}
        for node in r:
            v = float(state.wait[node])
            if v < 1e-5:
                v = 0.0
            if abs(v - round(v)) < 1e-6:
                waiting[str(node)] = int(round(v))
            else:
                waiting[str(node)] = v

        drone_assignments = []
        for j in sorted(state.assignment.keys()):
            d, i = state.assignment[j]
            drone_assignments.append(
                {
                    "drone": int(d),
                    "dispatch_node": int(i),
                    "delivery_node": int(j),
                }
            )

        return {
            "objective_value": float(state.objective()),
            "truck_route": arcs,
            "drone_assignments": drone_assignments,
            "waiting_times": waiting,
        }

    def update_best(state):
        obj = float(state.objective())
        if obj < best["obj"] - EPS:
            sol = build_solution(state)
            best["obj"] = obj
            best["sol"] = sol
            best["state"] = state.clone()
            if logger is not None:
                try:
                    logger.log_solution(obj, sol)
                except Exception:
                    pass

    def find_best_swap(state, end_time=None):
        route = state.route
        m = len(route)
        best_delta = -EPS
        best = None
        for i in range(1, m - 1):
            if end_time is not None and time.time() > end_time:
                break
            for j in range(i + 1, m):
                a = route[i]
                b = route[j]
                prev_a = route[i - 1]
                next_a = route[i + 1] if i + 1 < m else depot
                prev_b = route[j - 1]
                next_b = route[j + 1] if j + 1 < m else depot

                if i + 1 == j:
                    old = float(C[prev_a][a]) + float(C[a][b]) + float(C[b][next_b])
                    new = float(C[prev_a][b]) + float(C[b][a]) + float(C[a][next_b])
                else:
                    old = (
                        float(C[prev_a][a])
                        + float(C[a][next_a])
                        + float(C[prev_b][b])
                        + float(C[b][next_b])
                    )
                    new = (
                        float(C[prev_a][b])
                        + float(C[b][next_a])
                        + float(C[prev_b][a])
                        + float(C[a][next_b])
                    )

                delta = new - old
                if delta < best_delta:
                    best_delta = delta
                    best = (i, j, delta)
        if best is None:
            return None
        return (best_delta, best[0], best[1])

    def find_best_relocate(state, end_time=None):
        route = state.route
        m = len(route)
        best_delta = -EPS
        best = None

        for idx in range(1, m):
            if end_time is not None and time.time() > end_time:
                break
            k = route[idx]
            prev = route[idx - 1]
            nxt = route[idx + 1] if idx + 1 < m else depot
            saving = float(C[prev][k]) + float(C[k][nxt]) - float(C[prev][nxt])

            for q in range(m):
                if q == idx:
                    continue
                after = route[q]
                if q + 1 < m:
                    nxt2 = route[q + 1]
                    if nxt2 == k:
                        nxt2 = route[idx + 1] if idx + 1 < m else depot
                else:
                    nxt2 = depot

                inc = float(C[after][k]) + float(C[k][nxt2]) - float(C[after][nxt2])
                delta = inc - saving
                if delta < best_delta:
                    best_delta = delta
                    best = (k, after)

        if best is None:
            return None
        return (best_delta, best[0], best[1])

    def optimize_route(state, max_time):
        if max_time <= 0:
            return
        end = time.time() + max_time
        m = len(state.route)
        max_passes = 5 if m <= 80 else 2
        passes = 0
        improved = True

        while improved and passes < max_passes and time.time() < end:
            improved = False
            passes += 1

            sw = find_best_swap(state, end)
            if sw is not None:
                delta, i, j = sw
                if delta < -EPS:
                    state.route[i], state.route[j] = state.route[j], state.route[i]
                    state.travel += delta
                    state.rebuild_pos()
                    improved = True
                    if time.time() >= end:
                        break

            rel = find_best_relocate(state, end)
            if rel is not None:
                delta, k, after = rel
                if delta < -EPS:
                    state.remove_node(k)
                    state.insert_after_node(k, after)
                    improved = True

    def evaluate_combined_remove(state, k):
        if D == 0:
            return None

        dispatched = [j for j, di in state.assignment.items() if di[1] == k]
        affected = [k] + dispatched
        if len(affected) > 25:
            return None

        saving = state.removal_saving(k)
        old_wait_k = state.wait[k]

        visited_after = set(state.visited)
        visited_after.remove(k)

        temp_loads = [row[:] for row in state.loads]
        temp_battery = state.battery[:]

        for j in dispatched:
            d, _ = state.assignment[j]
            e = float(E[d][k][j])
            t = float(Tmat[d][k][j])
            temp_loads[k][d] -= t
            temp_battery[d] -= e

        remaining = set(affected)
        choices = {}
        added_wait = 0.0

        while remaining:
            best_j = None
            best_opts = None
            best_maxt = -1.0

            for j in remaining:
                opts = []
                for d, i, e, t in options_by_customer[j]:
                    if i not in visited_after or i == j:
                        continue
                    if temp_battery[d] + e > cap[d] + 1e-6:
                        continue
                    oldw = load_wait(temp_loads[i])
                    delta = max(0.0, temp_loads[i][d] + t - oldw)
                    score = delta + 1e-6 * (e / (cap[d] if cap[d] > 0 else 1.0))
                    opts.append((score, delta, d, i, e, t))

                if not opts:
                    return None

                maxt = max(x[5] for x in opts)
                if (
                    best_j is None
                    or len(opts) < len(best_opts)
                    or (len(opts) == len(best_opts) and maxt > best_maxt)
                ):
                    best_j = j
                    best_opts = opts
                    best_maxt = maxt

            if best_j is None:
                break

            best_opts.sort(key=lambda x: x[0])
            _, delta, d, i, e, t = best_opts[0]
            temp_loads[i][d] += t
            temp_battery[d] += e
            added_wait += delta
            choices[best_j] = (d, i)
            remaining.remove(best_j)

        total_delta = -saving - old_wait_k + added_wait
        return (total_delta, choices, dispatched)

    def find_best_drone_move(state, end_time):
        if D == 0:
            return None

        best_delta = -EPS
        best_move = None

        # Reassign drone-served customers.
        for j, (old_d, old_i) in list(state.assignment.items()):
            if time.time() > end_time:
                break

            old_e = float(E[old_d][old_i][j])
            old_t = float(Tmat[old_d][old_i][j])

            for d, i, e, t in options_by_customer[j]:
                if i not in state.visited:
                    continue
                if d == old_d and i == old_i:
                    continue
                if state.battery[d] - old_e + e > cap[d] + 1e-6:
                    continue

                if i == old_i:
                    before = state.wait[i]
                    state.loads[i][old_d] -= old_t
                    state.loads[i][d] += t
                    after = load_wait(state.loads[i])
                    state.loads[i][old_d] += old_t
                    state.loads[i][d] -= t
                    delta = after - before
                else:
                    before = state.wait[old_i] + state.wait[i]
                    state.loads[old_i][old_d] -= old_t
                    state.loads[i][d] += t
                    after = load_wait(state.loads[old_i]) + load_wait(state.loads[i])
                    state.loads[old_i][old_d] += old_t
                    state.loads[i][d] -= t
                    delta = after - before

                if delta < best_delta:
                    best_delta = delta
                    best_move = ("reassign", j, d, i)

        # Remove truck-served customers to drone service.
        for k in list(state.route[1:]):
            if time.time() > end_time:
                break

            if state.dispatch_count[k] == 0:
                saving = state.removal_saving(k)
                for d, i, e, t in options_by_customer[k]:
                    if i == k or i not in state.visited:
                        continue
                    if state.battery[d] + e > cap[d] + 1e-6:
                        continue
                    delta_wait = max(0.0, state.loads[i][d] + t - state.wait[i])
                    delta = -saving + delta_wait
                    if delta < best_delta:
                        best_delta = delta
                        best_move = ("remove_simple", k, d, i)
            else:
                res = evaluate_combined_remove(state, k)
                if res is not None:
                    delta, choices, dispatched = res
                    if delta < best_delta:
                        best_delta = delta
                        best_move = ("remove_combined", k, choices, dispatched)

        # Add drone-served customers back to truck route.
        for j, (old_d, old_i) in list(state.assignment.items()):
            if time.time() > end_time:
                break

            old_t = float(Tmat[old_d][old_i][j])
            before = state.wait[old_i]
            state.loads[old_i][old_d] -= old_t
            after = load_wait(state.loads[old_i])
            state.loads[old_i][old_d] += old_t
            delta_wait = after - before

            best_inc = float("inf")
            best_after = state.route[0]
            route = state.route
            m = len(route)

            for idx in range(m):
                prev = route[idx]
                nxt = route[idx + 1] if idx + 1 < m else depot
                inc = float(C[prev][j]) + float(C[j][nxt]) - float(C[prev][nxt])
                if inc < best_inc:
                    best_inc = inc
                    best_after = prev

            delta = best_inc + delta_wait
            if delta < best_delta:
                best_delta = delta
                best_move = ("add", j, best_after)

        return best_move

    def apply_move(state, move):
        typ = move[0]

        if typ == "reassign":
            _, j, d, i = move
            state.remove_assignment(j)
            state.add_assignment(j, d, i)

        elif typ == "remove_simple":
            _, k, d, i = move
            state.remove_node(k)
            state.add_assignment(k, d, i)

        elif typ == "remove_combined":
            _, k, choices, dispatched = move
            for j in dispatched:
                state.remove_assignment(j)
            state.remove_node(k)
            for j, di in choices.items():
                state.add_assignment(j, di[0], di[1])

        elif typ == "add":
            _, j, after = move
            state.remove_assignment(j)
            state.insert_after_node(j, after)

    def improve_heuristic(state, end_time):
        if time.time() >= end_time:
            return

        opt_time = min(3.0, max(0.2, (end_time - time.time()) * 0.2))
        old_travel = state.travel
        optimize_route(state, opt_time)
        if state.travel < old_travel - EPS:
            update_best(state)

        moves_since_route = 0

        while time.time() < end_time:
            move = find_best_drone_move(state, end_time)

            if move is None:
                old_travel = state.travel
                optimize_route(state, min(2.0, max(0.1, end_time - time.time())))
                if state.travel < old_travel - EPS:
                    update_best(state)
                    continue
                break

            old_obj = state.objective()
            backup = state.clone() if move[0] == "remove_combined" else None

            apply_move(state, move)
            new_obj = state.objective()

            if backup is not None and new_obj >= old_obj - EPS:
                state.__dict__.update(backup.__dict__)
                old_travel = state.travel
                optimize_route(state, min(1.0, max(0.05, end_time - time.time())))
                if state.travel < old_travel - EPS:
                    update_best(state)
                    continue
                break

            if new_obj < best["obj"] - EPS:
                update_best(state)

            moves_since_route += 1
            if moves_since_route >= 8:
                old_travel = state.travel
                optimize_route(state, min(1.0, max(0.05, end_time - time.time())))
                if state.travel < old_travel - EPS:
                    update_best(state)
                moves_since_route = 0

    def construct_mixed_state():
        state = State([depot])
        if not customers:
            return state

        sorted_cust = sorted(
            customers,
            key=lambda j: float(C[depot][j]) + float(C[j][depot]),
            reverse=True,
        )

        for j in sorted_cust:
            best_opt = None
            best_score = float("inf")
            best_delta = 0.0

            if D > 0:
                for d, i, e, t in options_by_customer[j]:
                    if i not in state.visited:
                        continue
                    if state.battery[d] + e > cap[d] + 1e-6:
                        continue
                    delta = max(0.0, state.loads[i][d] + t - state.wait[i])
                    score = delta + 1e-6 * (e / (cap[d] if cap[d] > 0 else 1.0))
                    if score < best_score:
                        best_score = score
                        best_opt = (d, i)
                        best_delta = delta

            best_inc = float("inf")
            best_after = depot
            for idx in range(len(state.route)):
                prev = state.route[idx]
                nxt = state.route[idx + 1] if idx + 1 < len(state.route) else depot
                inc = float(C[prev][j]) + float(C[j][nxt]) - float(C[prev][nxt])
                if inc < best_inc:
                    best_inc = inc
                    best_after = prev

            if best_opt is not None and best_delta <= best_inc + EPS:
                state.add_assignment(j, best_opt[0], best_opt[1])
            else:
                state.insert_after_node(j, best_after)

        return state

    def random_perturb(state, rng):
        changed = False

        for _ in range(rng.randint(2, 6)):
            possible = []

            if D > 0 and state.assignment:
                possible.append("reassign")
                possible.append("add")

            feasible_remove = []
            if D > 0:
                for k in state.route[1:]:
                    if state.dispatch_count[k] != 0:
                        continue
                    for d, i, e, t in options_by_customer[k]:
                        if i == k or i not in state.visited:
                            continue
                        if state.battery[d] + e <= cap[d] + 1e-6:
                            feasible_remove.append((k, d, i))
                            break

            if feasible_remove:
                possible.append("remove")

            if not possible:
                break

            typ = rng.choice(possible)

            if typ == "reassign":
                j = rng.choice(list(state.assignment.keys()))
                old_d, old_i = state.assignment[j]
                old_e = float(E[old_d][old_i][j])
                feasible = []

                for d, i, e, t in options_by_customer[j]:
                    if i not in state.visited:
                        continue
                    if d == old_d and i == old_i:
                        continue
                    if state.battery[d] - old_e + e <= cap[d] + 1e-6:
                        feasible.append((d, i))

                if feasible:
                    d, i = rng.choice(feasible)
                    state.remove_assignment(j)
                    state.add_assignment(j, d, i)
                    changed = True

            elif typ == "add":
                j = rng.choice(list(state.assignment.keys()))
                best_inc = float("inf")
                best_after = state.route[0]

                for idx in range(len(state.route)):
                    prev = state.route[idx]
                    nxt = state.route[idx + 1] if idx + 1 < len(state.route) else depot
                    inc = float(C[prev][j]) + float(C[j][nxt]) - float(C[prev][nxt])
                    if inc < best_inc:
                        best_inc = inc
                        best_after = prev

                state.remove_assignment(j)
                state.insert_after_node(j, best_after)
                changed = True

            elif typ == "remove":
                k, d, i = rng.choice(feasible_remove)
                state.remove_node(k)
                state.add_assignment(k, d, i)
                changed = True

        return changed

    def make_solution_state(route, assign):
        try:
            st = State(route, assign)
        except Exception:
            return None

        if not st.route or st.route[0] != depot:
            return None
        if len(set(st.route)) != len(st.route):
            return None

        truck_cust = set(st.route[1:])
        if len(truck_cust) + len(st.assignment) != num_customers:
            return None

        for j in customers:
            in_truck = j in truck_cust
            in_assign = j in st.assignment
            if in_truck == in_assign:
                return None

        for j, (d, i) in st.assignment.items():
            if i not in st.visited or i == j:
                return None

        for d in range(D):
            if st.battery[d] > cap[d] + 1e-5:
                return None

        sol = build_solution(st)
        return sol, st.objective(), st

    def route_assign_from_x_z(x_sel, z_sel):
        x_set = set(x_sel)
        assign = {}

        for d, i, j in z_sel:
            if j in assign:
                return None, None
            assign[j] = (d, i)

        if (depot, depot) in x_set:
            if len(x_set) != 1:
                return None, None
            return [depot], assign

        out = {}
        for i, j in x_sel:
            out[i] = j

        route = [depot]
        route_set = {depot}
        cur = depot
        returned = False

        for _ in range(num_customers + 1):
            nxt = out.get(cur)
            if nxt is None:
                return None, None
            if nxt == depot:
                returned = True
                break
            if nxt in route_set:
                return None, None
            route.append(nxt)
            route_set.add(nxt)
            cur = nxt

        if not returned:
            return None, None

        return route, assign

    def solve_assignment_mip(state, time_alloc):
        if gp is None or GRB is None or D == 0:
            return None

        R = list(state.route)
        U = [j for j in customers if j not in state.visited]
        if not U:
            return None

        var_count = 0
        for j in U:
            has = False
            for d, i, e, t in options_by_customer[j]:
                if i in state.visited:
                    var_count += 1
                    has = True
            if not has:
                return None

        if var_count > 20000:
            return None

        try:
            model = gp.Model()
            model.setParam("OutputFlag", 0)
            model.setParam("Seed", 0)
            model.setParam("MIPGap", 1e-4)
            model.setParam("NumericFocus", 0)
            model.setParam("Threads", 1)
            model.setParam("TimeLimit", max(0.5, float(time_alloc)))
            model.ModelSense = GRB.MINIMIZE

            z = {}
            z_by_j = {j: [] for j in U}
            z_by_d = {d: [] for d in range(D)}
            z_by_di = {}

            for j in U:
                for d, i, e, t in options_by_customer[j]:
                    if i not in state.visited:
                        continue
                    var = model.addVar(vtype=GRB.BINARY, obj=0.0)
                    z[(d, i, j)] = var
                    z_by_j[j].append(var)
                    z_by_d[d].append((e, var))
                    z_by_di.setdefault((d, i), []).append((t, var))

            w = {i: model.addVar(lb=0.0, obj=1.0) for i in R}
            model.update()

            for j in U:
                model.addConstr(gp.quicksum(z_by_j[j]) == 1)

            for d in range(D):
                if z_by_d[d]:
                    model.addConstr(gp.quicksum(e * var for e, var in z_by_d[d]) <= cap[d])

            for (d, i), lst in z_by_di.items():
                if i in w:
                    model.addConstr(w[i] >= gp.quicksum(t * var for t, var in lst))

            for var in z.values():
                var.Start = 0.0
            for i in R:
                w[i].Start = state.wait[i] if i in state.visited else 0.0

            for j in U:
                if j in state.assignment:
                    d, i = state.assignment[j]
                    if (d, i, j) in z:
                        z[(d, i, j)].Start = 1.0

            model.update()
            model.optimize()

            if model.SolCount > 0:
                new_assign = {}
                for (d, i, j), var in z.items():
                    if var.X > 0.5:
                        if j in new_assign:
                            return None
                        new_assign[j] = (d, i)

                if len(new_assign) == len(U):
                    new_state = State(state.route, new_assign)
                    feasible = True
                    for d in range(D):
                        if new_state.battery[d] > cap[d] + 1e-5:
                            feasible = False
                            break
                    if feasible and new_state.objective() < state.objective() - EPS:
                        return new_state
        except Exception:
            return None

        return None

    def solve_full_mip(initial_state, time_alloc):
        if gp is None or GRB is None:
            return

        try:
            model = gp.Model()
            model.setParam("OutputFlag", 0)
            model.setParam("Seed", 0)
            model.setParam("MIPGap", 1e-4)
            model.setParam("NumericFocus", 0)
            model.setParam("Threads", 1)
            model.setParam("TimeLimit", max(0.5, float(time_alloc)))
            model.ModelSense = GRB.MINIMIZE

            K = num_customers

            u = {j: model.addVar(vtype=GRB.BINARY) for j in customers}
            q = {
                j: model.addVar(lb=0.0, ub=float(max(1, K)), vtype=GRB.CONTINUOUS)
                for j in customers
            }

            x = {}
            for i in truck_nodes:
                for j in truck_nodes:
                    if i == j:
                        if i == depot:
                            x[(i, j)] = model.addVar(vtype=GRB.BINARY, obj=float(C[i][j]))
                        continue
                    x[(i, j)] = model.addVar(vtype=GRB.BINARY, obj=float(C[i][j]))

            w = {i: model.addVar(lb=0.0, obj=1.0) for i in truck_nodes}
            model.update()

            model.addConstr(
                gp.quicksum(x[(depot, j)] for j in truck_nodes if (depot, j) in x) == 1
            )
            model.addConstr(
                gp.quicksum(x[(i, depot)] for i in truck_nodes if (i, depot) in x) == 1
            )

            for k in customers:
                model.addConstr(
                    gp.quicksum(x[(k, j)] for j in truck_nodes if (k, j) in x) == u[k]
                )
                model.addConstr(
                    gp.quicksum(x[(i, k)] for i in truck_nodes if (i, k) in x) == u[k]
                )
                model.addConstr(q[k] >= u[k])
                model.addConstr(q[k] <= K * u[k])

            if K >= 2:
                for i in customers:
                    for j in customers:
                        if i != j:
                            model.addConstr(
                                q[i] - q[j] + 1 <= K * (1 - x[(i, j)])
                            )

            z = {}
            z_by_j = {j: [] for j in customers}
            z_by_d = {d: [] for d in range(D)}
            z_by_di = {}

            for j in customers:
                for d, i, e, t in options_by_customer[j]:
                    var = model.addVar(vtype=GRB.BINARY, obj=0.0)
                    z[(d, i, j)] = var
                    z_by_j[j].append(var)
                    z_by_d[d].append((e, var))
                    z_by_di.setdefault((d, i), []).append((t, var))
                    if i in u:
                        model.addConstr(var <= u[i])

            for j in customers:
                model.addConstr(u[j] + gp.quicksum(z_by_j[j]) == 1)

            for d in range(D):
                if z_by_d[d]:
                    model.addConstr(gp.quicksum(e * var for e, var in z_by_d[d]) <= cap[d])

            for (d, i), lst in z_by_di.items():
                if i in w:
                    model.addConstr(w[i] >= gp.quicksum(t * var for t, var in lst))

            for var in x.values():
                var.Start = 0.0
            for var in u.values():
                var.Start = 0.0
            for var in q.values():
                var.Start = 0.0
            for var in z.values():
                var.Start = 0.0
            for var in w.values():
                var.Start = 0.0

            r = initial_state.route
            if len(r) == 1:
                if (depot, depot) in x:
                    x[(depot, depot)].Start = 1.0
            else:
                for a, b in zip(r, r[1:]):
                    if (a, b) in x:
                        x[(a, b)].Start = 1.0
                if (r[-1], depot) in x:
                    x[(r[-1], depot)].Start = 1.0

            for idx, node in enumerate(r):
                if node != depot:
                    u[node].Start = 1.0
                    if node in q:
                        q[node].Start = float(idx)

            for j, (d, i) in initial_state.assignment.items():
                if (d, i, j) in z:
                    z[(d, i, j)].Start = 1.0

            for i in truck_nodes:
                w[i].Start = initial_state.wait[i] if i in initial_state.visited else 0.0

            model.update()

            x_list = [(i, j, var) for (i, j), var in x.items()]
            z_list = [(d, i, j, var) for (d, i, j), var in z.items()]

            def callback(model, where):
                try:
                    if where == GRB.Callback.MIPSOL:
                        sol_obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                        if sol_obj >= best["obj"] - EPS:
                            return

                        x_sel = []
                        for i, j, var in x_list:
                            if model.cbGetSolution(var) > 0.5:
                                x_sel.append((i, j))

                        z_sel = []
                        for d, i, j, var in z_list:
                            if model.cbGetSolution(var) > 0.5:
                                z_sel.append((d, i, j))

                        route, assign = route_assign_from_x_z(x_sel, z_sel)
                        if route is None:
                            return

                        res = make_solution_state(route, assign)
                        if res is None:
                            return

                        sol, obj, st = res
                        if obj < best["obj"] - EPS:
                            best["obj"] = obj
                            best["sol"] = sol
                            best["state"] = st.clone()
                            if logger is not None:
                                try:
                                    logger.log_solution(obj, sol)
                                except Exception:
                                    pass
                except Exception:
                    pass

            model.optimize(callback)

            if model.SolCount > 0:
                x_sel = []
                z_sel = []

                for i, j, var in x_list:
                    try:
                        if var.X > 0.5:
                            x_sel.append((i, j))
                    except Exception:
                        pass

                for d, i, j, var in z_list:
                    try:
                        if var.X > 0.5:
                            z_sel.append((d, i, j))
                    except Exception:
                        pass

                route, assign = route_assign_from_x_z(x_sel, z_sel)
                if route is not None:
                    res = make_solution_state(route, assign)
                    if res is not None:
                        sol, obj, st = res
                        if obj < best["obj"] - EPS:
                            best["obj"] = obj
                            best["sol"] = sol
                            best["state"] = st.clone()
                            if logger is not None:
                                try:
                                    logger.log_solution(obj, sol)
                                except Exception:
                                    pass
        except Exception:
            pass

    route = build_initial_route(customers)
    state = State(route)
    update_best(state)

    full_mip_allowed = False
    if gp is not None and GRB is not None and args.time_limit >= 8:
        if num_customers <= 22:
            full_mip_allowed = True
        elif num_customers <= 40 and total_options <= 12000:
            full_mip_allowed = True
        elif num_customers <= 55 and total_options <= 20000 and args.time_limit >= 60:
            full_mip_allowed = True
        if D == 0 and num_customers > 30:
            full_mip_allowed = False

    mip_reserve = 0
    if full_mip_allowed:
        mip_reserve = max(5, min(int(0.55 * args.time_limit), args.time_limit - 3))
        if mip_reserve < 3:
            full_mip_allowed = False
            mip_reserve = 0

    pre_mip_end = deadline - (mip_reserve if full_mip_allowed else 0) - 0.5
    if pre_mip_end < time.time():
        pre_mip_end = time.time() + 0.5

    if full_mip_allowed:
        heur_cap = min(120.0, max(5.0, pre_mip_end - start_time))
        main_end = start_time + max(2.0, heur_cap * 0.65)
    else:
        main_end = start_time + max(2.0, (pre_mip_end - start_time) * 0.60)

    main_end = min(main_end, pre_mip_end)

    if time.time() < main_end:
        improve_heuristic(state, main_end)

    if D > 0 and time.time() < pre_mip_end - 2.0:
        try:
            mixed = construct_mixed_state()
            update_best(mixed)
            mixed_end = min(
                pre_mip_end,
                time.time() + max(2.0, (pre_mip_end - time.time()) * 0.5),
            )
            improve_heuristic(mixed, mixed_end)
        except Exception:
            pass

    if (
        gp is not None
        and GRB is not None
        and D > 0
        and best["state"] is not None
        and time.time() < pre_mip_end - 2.0
    ):
        try:
            st = best["state"].clone()
            U = [j for j in customers if j not in st.visited]
            var_est = 0
            for j in U:
                for opt in options_by_customer[j]:
                    if opt[1] in st.visited:
                        var_est += 1

            if var_est <= 15000:
                new_st = solve_assignment_mip(
                    st,
                    min(5.0, max(1.0, pre_mip_end - time.time() - 1.0)),
                )
                if new_st is not None:
                    update_best(new_st)
                    improve_heuristic(new_st, min(pre_mip_end, time.time() + 2.0))
        except Exception:
            pass

    if not full_mip_allowed and best["state"] is not None:
        rng = random.Random(0)
        while time.time() < deadline - 2.0:
            st = best["state"].clone()
            changed = random_perturb(st, rng)
            if not changed:
                break
            iter_end = min(
                deadline - 1.0,
                time.time() + max(2.0, (deadline - time.time()) * 0.2),
            )
            improve_heuristic(st, iter_end)

    if full_mip_allowed and best["state"] is not None:
        time_alloc = max(1.0, deadline - time.time() - 0.5)
        if time_alloc > 1.0:
            try:
                solve_full_mip(best["state"], time_alloc)
            except Exception:
                pass

    if best["sol"] is None:
        fallback = State([depot] + customers)
        best["sol"] = build_solution(fallback)
        best["obj"] = fallback.objective()

    with open(args.solution_path, "w") as f:
        json.dump(best["sol"], f, indent=2)


if __name__ == "__main__":
    main()