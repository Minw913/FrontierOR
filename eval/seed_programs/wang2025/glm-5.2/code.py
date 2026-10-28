import argparse
import json
import time
import heapq
import sys

try:
    from solution_logger import SolutionLogger
except ImportError:
    SolutionLogger = None


class Instance:
    def __init__(self, data):
        self.type = data['instance_type']
        self.n = data['n']
        self.m = data['m']
        self.k = data['k']
        self.gamma = data.get('gamma', 1.0)
        self.type_specific_param = data.get('type_specific_param', 0.0)
        self.data_matrix = data['data_matrix']
        self.node_weights = data['node_weights']
        self.element_weights = data['element_weights']


class State:
    def __init__(self, instance):
        self.inst = instance
        self.S = set()
        self.obj = 0.0
        if instance.type == 'LOC':
            self.current_max = [0.0] * instance.m
            self.count_max = [0] * instance.m
        elif instance.type == 'COV':
            self.cover_count = [0] * instance.m
        elif instance.type == 'INF':
            self.prob_not_reached = [1.0] * instance.m

    def add(self, i):
        """Add element i, return (old_obj, changes) for undo."""
        old_obj = self.obj
        changes = []
        inst = self.inst
        if inst.type == 'LOC':
            for j in range(inst.m):
                val = inst.data_matrix[j][i]
                if val > self.current_max[j]:
                    changes.append((j, self.current_max[j], self.count_max[j]))
                    self.obj += val - self.current_max[j]
                    self.current_max[j] = val
                    self.count_max[j] = 1
                elif val == self.current_max[j]:
                    changes.append((j, self.current_max[j], self.count_max[j]))
                    self.count_max[j] += 1
        elif inst.type == 'COV':
            for j in range(inst.m):
                if inst.data_matrix[j][i] == 1:
                    if self.cover_count[j] == 0:
                        self.obj += inst.node_weights[j]
                    self.cover_count[j] += 1
                    changes.append(j)
        elif inst.type == 'INF':
            p = inst.element_weights[i]
            for j in range(inst.m):
                if inst.data_matrix[j][i] == 1:
                    gain = self.prob_not_reached[j] * p
                    self.obj += gain
                    changes.append((j, self.prob_not_reached[j]))
                    self.prob_not_reached[j] *= (1.0 - p)
        self.S.add(i)
        return old_obj, changes

    def undo_add(self, i, old_obj, changes):
        self.S.remove(i)
        self.obj = old_obj
        inst = self.inst
        if inst.type == 'LOC':
            for j, old_max, old_count in changes:
                self.current_max[j] = old_max
                self.count_max[j] = old_count
        elif inst.type == 'COV':
            for j in changes:
                self.cover_count[j] -= 1
        elif inst.type == 'INF':
            for j, old_prob in changes:
                self.prob_not_reached[j] = old_prob

    def rebuild(self, S_list):
        self.S = set()
        self.obj = 0.0
        inst = self.inst
        if inst.type == 'LOC':
            self.current_max = [0.0] * inst.m
            self.count_max = [0] * inst.m
        elif inst.type == 'COV':
            self.cover_count = [0] * inst.m
        elif inst.type == 'INF':
            self.prob_not_reached = [1.0] * inst.m
        for x in S_list:
            self.add(x)


def greedy(instance):
    state = State(instance)
    n = instance.n
    k = instance.k
    pq = []
    for i in range(n):
        old_obj, changes = state.add(i)
        gain = state.obj - old_obj
        state.undo_add(i, old_obj, changes)
        pq.append((-gain, i))
    heapq.heapify(pq)

    selected = []
    while len(selected) < k and pq:
        neg_gain, i = heapq.heappop(pq)
        old_obj, changes = state.add(i)
        gain = state.obj - old_obj
        state.undo_add(i, old_obj, changes)
        if gain <= 1e-12:
            continue
        if not pq or gain >= -pq[0][0]:
            state.add(i)
            selected.append(i)
        else:
            heapq.heappush(pq, (-gain, i))
    return selected, state.obj


def local_search(initial_S, instance, time_limit, start_time, logger=None):
    state = State(instance)
    for i in initial_S:
        state.add(i)
    best_obj = state.obj
    best_S = list(state.S)
    if logger:
        logger.log_solution(best_obj, {
            'objective_value': float(best_obj),
            'selected_elements': [int(x) for x in best_S]
        })

    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        S_list = list(state.S)
        for a in S_list:
            if time.time() - start_time >= time_limit:
                break
            state_without_a = State(instance)
            for x in S_list:
                if x != a:
                    state_without_a.add(x)
            for b in range(instance.n):
                if b in state.S or b == a:
                    continue
                if time.time() - start_time >= time_limit:
                    break
                old_obj, changes = state_without_a.add(b)
                found = state_without_a.obj > state.obj + 1e-6
                state_without_a.undo_add(b, old_obj, changes)
                if found:
                    new_S = [x for x in state.S if x != a]
                    new_S.append(b)
                    state.rebuild(new_S)
                    best_obj = state.obj
                    best_S = list(state.S)
                    improved = True
                    if logger:
                        logger.log_solution(best_obj, {
                            'objective_value': float(best_obj),
                            'selected_elements': [int(x) for x in best_S]
                        })
                    break
            if improved:
                break
    return best_obj, best_S


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    with open(args.instance_path) as f:
        data = json.load(f)
    instance = Instance(data)

    logger = None
    if args.log_path and SolutionLogger:
        logger = SolutionLogger(args.log_path, sense="maximize")

    start_time = time.time()

    if instance.k == 0:
        best_S = []
        best_obj = 0.0
        if logger:
            logger.log_solution(best_obj, {
                'objective_value': 0.0,
                'selected_elements': []
            })
    else:
        greedy_S, greedy_obj = greedy(instance)
        best_S = list(greedy_S)
        best_obj = greedy_obj
        if logger:
            logger.log_solution(best_obj, {
                'objective_value': float(best_obj),
                'selected_elements': [int(x) for x in best_S]
            })

        remaining_time = args.time_limit - (time.time() - start_time)
        if remaining_time > 0 and instance.n > 0:
            best_obj, best_S = local_search(
                best_S, instance, args.time_limit, start_time, logger
            )

    solution = {
        'objective_value': float(best_obj),
        'selected_elements': [int(x) for x in best_S]
    }
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == '__main__':
    main()