import argparse
import json
import time
from solution_logger import SolutionLogger


class Knapsack:
    def __init__(self, instance, logger, time_limit):
        self.n = instance["n"]
        self.capacity = instance["capacity"]
        self.items = instance["items"]
        self.logger = logger
        self.time_limit = time_limit
        self.start_time = time.time()
        
        # Sort items by profit/weight ratio descending for better branching
        self.items.sort(key=lambda x: x["profit"] / x["weight"], reverse=True)
        
        self.profits = [item["profit"] for item in self.items]
        self.weights = [item["weight"] for item in self.items]
        self.ids = [item["id"] for item in self.items]
        
        self.best_profit = 0
        self.best_sol = []
        
        # Greedy initial solution
        cw = 0
        cp = 0
        cs = []
        for i in range(self.n):
            if cw + self.weights[i] <= self.capacity:
                cw += self.weights[i]
                cp += self.profits[i]
                cs.append(i)
                
        if cp > self.best_profit:
            self.best_profit = cp
            self.best_sol = [self.ids[i] for i in cs]
            if self.logger:
                self.logger.log_solution(self.best_profit, {
                    "objective_value": int(self.best_profit),
                    "selected_items": self.best_sol
                })

        # Precompute prefix sums for fast linear relaxation upper bound
        self.pref_w = [0] * (self.n + 1)
        self.pref_p = [0] * (self.n + 1)
        for i in range(self.n):
            self.pref_w[i+1] = self.pref_w[i] + self.weights[i]
            self.pref_p[i+1] = self.pref_p[i] + self.profits[i]

    def upper_bound(self, i, curr_w, curr_p):
        """Calculate linear relaxation upper bound using binary search."""
        rem_cap = self.capacity - curr_w
        lo = i
        hi = self.n
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.pref_w[mid] - self.pref_w[i] <= rem_cap:
                lo = mid
            else:
                hi = mid - 1
        j = lo
        ub = curr_p + self.pref_p[j] - self.pref_p[i]
        if j < self.n:
            taken_w = self.pref_w[j] - self.pref_w[i]
            remaining = rem_cap - taken_w
            ub += self.profits[j] * (remaining / self.weights[j])
        return ub

    def solve(self):
        x = [0] * self.n
        stack = [(0, 0, 0, 0)]  # (i, curr_w, curr_p, state) where state 0=evaluate, 3=cleanup
        
        while stack:
            if time.time() - self.start_time > self.time_limit:
                break
                
            i, cw, cp, state = stack.pop()
            
            if state == 3:
                x[i] = 0
                continue
                
            if cp > self.best_profit:
                self.best_profit = cp
                self.best_sol = [self.ids[j] for j in range(i) if x[j] == 1]
                if self.logger:
                    self.logger.log_solution(self.best_profit, {
                        "objective_value": int(self.best_profit),
                        "selected_items": self.best_sol
                    })
                    
            if i == self.n:
                continue
                
            if self.upper_bound(i, cw, cp) <= self.best_profit + 1e-6:
                continue
                
            # Skip branch
            x[i] = 0
            stack.append((i + 1, cw, cp, 0))
            
            # Take branch
            if cw + self.weights[i] <= self.capacity:
                x[i] = 1
                # Push cleanup state to reset x[i] after take branch is fully explored
                stack.append((i, 0, 0, 3))
                stack.append((i + 1, cw + self.weights[i], cp + self.profits[i], 0))


def main():
    parser = argparse.ArgumentParser(description="0/1 Knapsack solver")
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    knapsack = Knapsack(instance, logger, args.time_limit)
    knapsack.solve()

    solution = {
        "objective_value": int(knapsack.best_profit),
        "selected_items": knapsack.best_sol
    }

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()