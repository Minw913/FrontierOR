import json
import argparse
import time
import random
import math
from copy import deepcopy
from solution_logger import SolutionLogger


class BatchSchedulingSolver:
    def __init__(self, instance):
        self.num_jobs = instance['num_jobs']
        self.num_machines = instance['num_machines']
        self.cap = instance['batch_capacity']
        self.jobs = {j['job_id']: j for j in instance['jobs']}
        self.job_ids = sorted(self.jobs.keys())
        self.start_time = None

    def batch_size(self, batch):
        return sum(self.jobs[j]['size'] for j in batch)

    def batch_proc_time(self, batch):
        return max(self.jobs[j]['processing_time'] for j in batch)

    def batch_release(self, batch):
        return max(self.jobs[j]['release_date'] for j in batch)

    def compute_schedule(self, machines_batches):
        """Compute total flow time and job completions given batch assignments."""
        total_flow = 0
        job_completions = {}

        for m in range(self.num_machines):
            machine_time = 0
            for batch in machines_batches[m]:
                if not batch:
                    continue
                pt = self.batch_proc_time(batch)
                rel = self.batch_release(batch)
                st = max(machine_time, rel)
                et = st + pt
                machine_time = et
                for j in batch:
                    job_completions[j] = et
                    total_flow += et

        return total_flow, job_completions

    def construct_greedy(self, sort_key):
        """Greedy construction: sort jobs, form batches, assign to machines."""
        sorted_jobs = sorted(self.job_ids, key=sort_key)

        batches = []
        current_batch = []
        current_size = 0

        for jid in sorted_jobs:
            jsize = self.jobs[jid]['size']
            if current_size + jsize <= self.cap:
                current_batch.append(jid)
                current_size += jsize
            else:
                if current_batch:
                    batches.append(current_batch)
                current_batch = [jid]
                current_size = jsize
        if current_batch:
            batches.append(current_batch)

        return self.assign_to_machines(batches)

    def construct_best_fit(self, sort_key):
        """Best-fit decreasing style: for each job, find the batch with least remaining space."""
        sorted_jobs = sorted(self.job_ids, key=sort_key)

        batches = []  # list of (current_size, [job_ids])

        for jid in sorted_jobs:
            jsize = self.jobs[jid]['size']
            best_idx = -1
            best_remaining = float('inf')

            for i, (bsize, bjobs) in enumerate(batches):
                remaining = self.cap - bsize
                if jsize <= remaining and remaining < best_remaining:
                    best_remaining = remaining
                    best_idx = i

            if best_idx >= 0:
                batches[best_idx][0] += jsize
                batches[best_idx][1].append(jid)
            else:
                batches.append([jsize, [jid]])

        batch_list = [b[1] for b in batches]
        return self.assign_to_machines(batch_list)

    def assign_to_machines(self, batches):
        """Assign batches to machines using list scheduling (earliest completion)."""
        if not batches:
            return [[] for _ in range(self.num_machines)]

        # Sort batches by release date for assignment order
        batches_sorted = sorted(batches, key=lambda b: (self.batch_release(b), self.batch_proc_time(b)))

        machine_end = [0] * self.num_machines
        machines_batches = [[] for _ in range(self.num_machines)]

        for batch in batches_sorted:
            pt = self.batch_proc_time(batch)
            rel = self.batch_release(batch)

            best_m = 0
            best_end = float('inf')
            for m in range(self.num_machines):
                end = max(machine_end[m], rel) + pt
                if end < best_end:
                    best_end = end
                    best_m = m

            machines_batches[best_m].append(batch)
            machine_end[best_m] = best_end

        return machines_batches

    def get_all_batch_positions(self, sol):
        """Get all (machine_idx, batch_idx) for non-empty batches."""
        result = []
        for m in range(self.num_machines):
            for b_idx in range(len(sol[m])):
                if sol[m][b_idx]:
                    result.append((m, b_idx))
        return result

    def try_move(self, sol):
        """Move a random job from one batch to another."""
        all_pos = self.get_all_batch_positions(sol)
        if len(all_pos) < 2:
            return False

        sm, sb = random.choice(all_pos)
        src_batch = sol[sm][sb]
        if not src_batch:
            return False

        job_idx = random.randint(0, len(src_batch) - 1)
        job_id = src_batch[job_idx]
        job_size = self.jobs[job_id]['size']

        dst_options = [(m, b) for m, b in all_pos if (m, b) != (sm, sb)]
        if not dst_options:
            return False

        dm, db = random.choice(dst_options)
        dst_batch = sol[dm][db]

        if self.batch_size(dst_batch) + job_size <= self.cap:
            src_batch.pop(job_idx)
            dst_batch.append(job_id)
            if not src_batch:
                sol[sm].pop(sb)
            return True
        return False

    def try_swap(self, sol):
        """Swap two jobs between different batches."""
        all_pos = self.get_all_batch_positions(sol)
        if len(all_pos) < 2:
            return False

        pos1 = random.choice(all_pos)
        pos2 = random.choice(all_pos)
        if pos1 == pos2:
            return False

        m1, b1 = pos1
        m2, b2 = pos2
        batch1 = sol[m1][b1]
        batch2 = sol[m2][b2]

        if not batch1 or not batch2:
            return False

        i1 = random.randint(0, len(batch1) - 1)
        i2 = random.randint(0, len(batch2) - 1)

        j1 = batch1[i1]
        j2 = batch2[i2]

        s1 = self.jobs[j1]['size']
        s2 = self.jobs[j2]['size']

        bs1 = self.batch_size(batch1)
        bs2 = self.batch_size(batch2)

        if bs1 - s1 + s2 <= self.cap and bs2 - s2 + s1 <= self.cap:
            batch1[i1] = j2
            batch2[i2] = j1
            return True
        return False

    def try_merge(self, sol):
        """Merge two batches on the same machine if capacity allows."""
        for _ in range(10):
            m = random.randint(0, self.num_machines - 1)
            if len(sol[m]) >= 2:
                i = random.randint(0, len(sol[m]) - 1)
                j = random.randint(0, len(sol[m]) - 1)
                if i != j:
                    b1 = sol[m][i]
                    b2 = sol[m][j]
                    if self.batch_size(b1) + self.batch_size(b2) <= self.cap:
                        merged = b1 + b2
                        hi, lo = max(i, j), min(i, j)
                        sol[m].pop(hi)
                        sol[m].pop(lo)
                        sol[m].append(merged)
                        return True
        return False

    def try_split(self, sol):
        """Split a batch with multiple jobs into two."""
        candidates = []
        for m in range(self.num_machines):
            for b_idx in range(len(sol[m])):
                if len(sol[m][b_idx]) > 1:
                    candidates.append((m, b_idx))

        if not candidates:
            return False

        m, b_idx = random.choice(candidates)
        batch = sol[m][b_idx]

        shuffled = list(batch)
        random.shuffle(shuffled)
        split_point = random.randint(1, len(shuffled) - 1)
        b1 = shuffled[:split_point]
        b2 = shuffled[split_point:]

        sol[m][b_idx] = b1
        sol[m].append(b2)
        return True

    def try_reorder(self, sol):
        """Swap two adjacent batches on a machine."""
        machines_with_multi = [m for m in range(self.num_machines) if len(sol[m]) >= 2]
        if not machines_with_multi:
            return False

        m = random.choice(machines_with_multi)
        i = random.randint(0, len(sol[m]) - 2)
        sol[m][i], sol[m][i + 1] = sol[m][i + 1], sol[m][i]
        return True

    def try_move_to_new_batch(self, sol):
        """Move a job to a new batch on the same or different machine."""
        all_pos = self.get_all_batch_positions(sol)
        if not all_pos:
            return False

        sm, sb = random.choice(all_pos)
        src_batch = sol[sm][sb]
        if len(src_batch) <= 1:
            return False

        job_idx = random.randint(0, len(src_batch) - 1)
        job_id = src_batch.pop(job_idx)

        target_m = random.randint(0, self.num_machines - 1)
        sol[target_m].append([job_id])
        return True

    def sort_machine_batches(self, sol):
        """Sort batches on each machine by (release, proc_time)."""
        for m in range(self.num_machines):
            sol[m].sort(key=lambda b: (self.batch_release(b), self.batch_proc_time(b)) if b else (0, 0))

    def format_solution(self, machines_batches):
        """Format solution according to output schema."""
        total_flow, _ = self.compute_schedule(machines_batches)

        batches_output = []
        job_assignments = {}

        for m in range(self.num_machines):
            machine_time = 0
            batch_id = 0
            for batch in machines_batches[m]:
                if not batch:
                    continue
                batch_id += 1
                pt = self.batch_proc_time(batch)
                rel = self.batch_release(batch)
                st = max(machine_time, rel)
                et = st + pt
                machine_time = et

                batches_output.append({
                    'batch_id': batch_id,
                    'machine': m,
                    'start_time': st,
                    'processing_time': pt,
                    'jobs': sorted(batch)
                })

                for j in batch:
                    job_assignments[str(j)] = {
                        'batch': batch_id,
                        'machine': m,
                        'completion_time': et
                    }

        return {
            'objective_value': float(total_flow),
            'batches': batches_output,
            'job_assignments': job_assignments
        }

    def solve(self, time_limit, logger):
        self.start_time = time.time()
        random.seed(42)

        # Phase 1: Construction - try multiple heuristics
        best_obj = float('inf')
        best_sol = None

        sort_keys = [
            lambda j: (self.jobs[j]['release_date'], self.jobs[j]['processing_time']),
            lambda j: (self.jobs[j]['release_date'], -self.jobs[j]['processing_time']),
            lambda j: (self.jobs[j]['processing_time'], self.jobs[j]['release_date']),
            lambda j: (-self.jobs[j]['processing_time'], self.jobs[j]['release_date']),
            lambda j: (self.jobs[j]['release_date'], self.jobs[j]['size']),
            lambda j: (self.jobs[j]['size'], self.jobs[j]['release_date']),
        ]

        for key in sort_keys:
            sol = self.construct_greedy(key)
            obj, _ = self.compute_schedule(sol)
            if obj < best_obj:
                best_obj = obj
                best_sol = deepcopy(sol)

            sol2 = self.construct_best_fit(key)
            obj2, _ = self.compute_schedule(sol2)
            if obj2 < best_obj:
                best_obj = obj2
                best_sol = deepcopy(sol2)

        if logger:
            sol_dict = self.format_solution(best_sol)
            logger.log_solution(best_obj, sol_dict)

        # Check if we have time for improvement
        if time.time() - self.start_time >= time_limit - 1:
            return self.format_solution(best_sol)

        # Phase 2: Simulated Annealing
        current_sol = deepcopy(best_sol)
        current_obj = best_obj

        T = max(best_obj * 0.01, 1.0)
        cooling = 0.9998

        no_improve = 0
        max_no_improve = 300

        while time.time() - self.start_time < time_limit - 1:
            move_type = random.choices(
                ['move', 'swap', 'merge', 'split', 'reorder', 'move_new'],
                weights=[25, 20, 10, 10, 25, 10],
                k=1
            )[0]

            new_sol = deepcopy(current_sol)

            if move_type == 'move':
                success = self.try_move(new_sol)
            elif move_type == 'swap':
                success = self.try_swap(new_sol)
            elif move_type == 'merge':
                success = self.try_merge(new_sol)
            elif move_type == 'split':
                success = self.try_split(new_sol)
            elif move_type == 'reorder':
                success = self.try_reorder(new_sol)
            else:
                success = self.try_move_to_new_batch(new_sol)

            if not success:
                no_improve += 1
                if no_improve > max_no_improve:
                    current_sol = deepcopy(best_sol)
                    current_obj = best_obj
                    T = max(best_obj * 0.005, 1.0)
                    no_improve = 0
                continue

            new_obj, _ = self.compute_schedule(new_sol)
            delta = new_obj - current_obj

            if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                current_obj = new_obj
                current_sol = new_sol
                no_improve = 0

                if new_obj < best_obj:
                    best_obj = new_obj
                    best_sol = deepcopy(new_sol)
                    if logger:
                        sol_dict = self.format_solution(best_sol)
                        logger.log_solution(best_obj, sol_dict)
            else:
                no_improve += 1
                if no_improve > max_no_improve:
                    current_sol = deepcopy(best_sol)
                    current_obj = best_obj
                    T = max(best_obj * 0.005, 1.0)
                    no_improve = 0

            T *= cooling

        # Phase 3: Final local search (steepest descent) from best solution
        improved = True
        while improved and time.time() - self.start_time < time_limit - 0.5:
            improved = False
            for move_func in [self.try_move, self.try_swap, self.try_reorder]:
                for _ in range(50):
                    if time.time() - self.start_time >= time_limit - 0.5:
                        break
                    new_sol = deepcopy(best_sol)
                    if move_func(new_sol):
                        new_obj, _ = self.compute_schedule(new_sol)
                        if new_obj < best_obj:
                            best_obj = new_obj
                            best_sol = new_sol
                            improved = True
                            if logger:
                                sol_dict = self.format_solution(best_sol)
                                logger.log_solution(best_obj, sol_dict)

        return self.format_solution(best_sol)


def main():
    parser = argparse.ArgumentParser(description='Batch Scheduling Solver')
    parser.add_argument('--instance_path', required=True, help='Path to instance JSON file')
    parser.add_argument('--solution_path', required=True, help='Path to write solution JSON')
    parser.add_argument('--time_limit', type=int, default=300, help='Time limit in seconds')
    parser.add_argument('--log_path', default=None, help='Path for solution log')
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    solver = BatchSchedulingSolver(instance)
    solution = solver.solve(args.time_limit, logger)

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == '__main__':
    main()