import argparse
import json
import time
import sys

sys.setrecursionlimit(100000)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path) as f:
        instance = json.load(f)

    n = instance['num_vertices']
    edges = instance['edges']

    # Build adjacency as bitsets (Python integers)
    adj = [0] * n
    for u, v in edges:
        if u == v:
            continue
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    deg = [adj[i].bit_count() for i in range(n)]
    order = sorted(range(n), key=lambda v: -deg[v])
    order_index = [0] * n
    for i, v in enumerate(order):
        order_index[v] = i

    class MaxClique:
        def __init__(self):
            self.n = n
            self.adj = adj
            self.deg = deg
            self.order = order
            self.order_index = order_index
            self.best = 0
            self.best_clique = []
            self.R = []
            self.timeout = False
            self.start_time = time.time()
            self.time_limit = args.time_limit
            self.logger = logger

        def check_time(self):
            if time.time() - self.start_time > self.time_limit:
                self.timeout = True

        def greedy_initial(self):
            """Quick greedy heuristic to obtain a good lower bound."""
            best_clique = []
            starts = self.order[:min(3, self.n)]
            for s in starts:
                clique = [s]
                P = self.adj[s]
                while P:
                    best_v = -1
                    best_deg = -1
                    temp = P
                    while temp:
                        v = (temp & -temp).bit_length() - 1
                        d = self.deg[v]
                        if d > best_deg:
                            best_deg = d
                            best_v = v
                        temp &= temp - 1
                    if best_v == -1:
                        break
                    clique.append(best_v)
                    P &= self.adj[best_v]
                if len(clique) > len(best_clique):
                    best_clique = clique
            return best_clique

        def color_sort(self, P):
            """Greedy coloring of the candidate set P.
               Returns a list of (vertex, color) sorted by color descending."""
            verts = []
            temp = P
            while temp:
                v = (temp & -temp).bit_length() - 1
                verts.append(v)
                temp &= temp - 1
            verts.sort(key=lambda v: self.order_index[v])

            colors = []
            color_classes = {}
            for v in verts:
                c = 1
                while c in color_classes:
                    if (color_classes[c] & self.adj[v] & P) == 0:
                        break
                    c += 1
                if c not in color_classes:
                    color_classes[c] = 0
                color_classes[c] |= (1 << v)
                colors.append((v, c))
            colors.sort(key=lambda x: -x[1])
            return colors

        def expand(self, P):
            if self.timeout:
                return
            self.check_time()
            if self.timeout:
                return
            if P == 0:
                if len(self.R) > self.best:
                    self.best = len(self.R)
                    self.best_clique = list(self.R)
                    if self.logger:
                        self.logger.log_solution(self.best, {"objective_value": self.best, "clique_vertices": self.best_clique})
                return
            # Trivial bound
            if len(self.R) + P.bit_count() <= self.best:
                return
            # Tighter bound via greedy coloring
            colors = self.color_sort(P)
            for v, c in colors:
                if self.timeout:
                    return
                if len(self.R) + c <= self.best:
                    break
                self.R.append(v)
                self.expand(P & self.adj[v])
                self.R.pop()
                P &= ~(1 << v)

    mc = MaxClique()
    initial = mc.greedy_initial()
    if len(initial) > mc.best:
        mc.best = len(initial)
        mc.best_clique = initial
        if logger:
            logger.log_solution(mc.best, {"objective_value": mc.best, "clique_vertices": mc.best_clique})

    if n > 0:
        full_P = (1 << n) - 1
        mc.expand(full_P)

    # Output
    mc.best_clique.sort()
    solution = {
        "objective_value": mc.best,
        "clique_vertices": mc.best_clique
    }
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)

if __name__ == "__main__":
    main()