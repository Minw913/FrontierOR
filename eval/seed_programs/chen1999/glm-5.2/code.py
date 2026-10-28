import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


class IncumbentLogger:
    """Callback to log incumbent solutions found during MIP search."""
    def __init__(self, logger, problem_type, num_jobs, num_machines):
        self.logger = logger
        self.problem_type = problem_type
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.vars = {}

    def register(self, **kwargs):
        self.vars.update(kwargs)

    def __call__(self, model, where):
        if where != GRB.Callback.MIPSOL or self.logger is None:
            return
        obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)

        if self.problem_type == 'weighted_completion_time':
            x_vals = model.cbGetSolution(
                [self.vars['x'][(i, m)] for i in range(self.num_jobs) for m in range(self.num_machines)]
            )
            C_vals = model.cbGetSolution([self.vars['C'][i] for i in range(self.num_jobs)])
            x_dict = {}
            idx = 0
            for i in range(self.num_jobs):
                for m in range(self.num_machines):
                    x_dict[(i, m)] = x_vals[idx]
                    idx += 1
            C_dict = {i: C_vals[i] for i in range(self.num_jobs)}

            assignment = {}
            for i in range(self.num_jobs):
                for m in range(self.num_machines):
                    if x_dict[(i, m)] > 0.5:
                        assignment[i] = m
                        break
            schedule = {str(m): [] for m in range(self.num_machines)}
            for m in range(self.num_machines):
                jobs_on_m = [i for i in range(self.num_jobs) if assignment.get(i) == m]
                jobs_on_m.sort(key=lambda i: C_dict[i])
                schedule[str(m)] = jobs_on_m
            completion_times = {str(i): C_dict[i] for i in range(self.num_jobs)}
            solution = {
                'objective_value': obj,
                'schedule': schedule,
                'completion_times': completion_times,
                'tardy_jobs': [],
                'on_time_jobs': list(range(self.num_jobs)),
            }
        else:
            z_vals = model.cbGetSolution(
                [self.vars['z'][(i, m)] for i in range(self.num_jobs) for m in range(self.num_machines)]
            )
            U_vals = model.cbGetSolution([self.vars['U'][i] for i in range(self.num_jobs)])
            C_vals = model.cbGetSolution([self.vars['C'][i] for i in range(self.num_jobs)])
            z_dict = {}
            idx = 0
            for i in range(self.num_jobs):
                for m in range(self.num_machines):
                    z_dict[(i, m)] = z_vals[idx]
                    idx += 1
            U_dict = {i: U_vals[i] for i in range(self.num_jobs)}
            C_dict = {i: C_vals[i] for i in range(self.num_jobs)}

            assignment = {}
            on_time = []
            tardy = []
            for i in range(self.num_jobs):
                if U_dict[i] > 0.5:
                    tardy.append(i)
                else:
                    on_time.append(i)
                    for m in range(self.num_machines):
                        if z_dict[(i, m)] > 0.5:
                            assignment[i] = m
                            break
            schedule = {str(m): [] for m in range(self.num_machines)}
            for m in range(self.num_machines):
                jobs_on_m = [i for i in on_time if assignment.get(i) == m]
                jobs_on_m.sort(key=lambda i: C_dict[i])
                schedule[str(m)] = jobs_on_m
            completion_times = {str(i): C_dict[i] for i in on_time}
            solution = {
                'objective_value': obj,
                'schedule': schedule,
                'completion_times': completion_times,
                'tardy_jobs': tardy,
                'on_time_jobs': on_time,
            }

        self.logger.log_solution(obj, solution)


def solve(instance, time_limit, log_path):
    problem_type = instance['problem_type']
    num_jobs = instance['num_jobs']
    num_machines = instance['num_machines']
    weights = instance['jobs']['weights']
    processing_times = instance['jobs']['processing_times']

    logger = SolutionLogger(log_path, sense="minimize") if log_path else None

    model = gp.Model('scheduling')
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', time_limit)
    model.setParam('OutputFlag', 0)

    callback = IncumbentLogger(logger, problem_type, num_jobs, num_machines)

    if problem_type == 'weighted_completion_time':
        M = sum(max(processing_times[j]) for j in range(num_jobs)) + 1

        x = {}
        for i in range(num_jobs):
            for m in range(num_machines):
                x[i, m] = model.addVar(vtype=GRB.BINARY, name=f'x_{i}_{m}')

        C = {}
        for i in range(num_jobs):
            C[i] = model.addVar(lb=0, ub=M, vtype=GRB.CONTINUOUS, name=f'C_{i}')

        y = {}
        for i in range(num_jobs):
            for j in range(i + 1, num_jobs):
                y[i, j] = model.addVar(vtype=GRB.BINARY, name=f'y_{i}_{j}')
                y[j, i] = model.addVar(vtype=GRB.BINARY, name=f'y_{j}_{i}')

        model.update()
        callback.register(x=x, C=C, y=y)

        for i in range(num_jobs):
            model.addConstr(gp.quicksum(x[i, m] for m in range(num_machines)) == 1)

        for i in range(num_jobs):
            model.addConstr(C[i] >= gp.quicksum(processing_times[i][m] * x[i, m] for m in range(num_machines)))

        for i in range(num_jobs):
            for j in range(i + 1, num_jobs):
                model.addConstr(y[i, j] + y[j, i] <= 1)
                for m in range(num_machines):
                    model.addConstr(y[i, j] + y[j, i] >= x[i, m] + x[j, m] - 1)

        for i in range(num_jobs):
            for j in range(num_jobs):
                if i == j:
                    continue
                for m in range(num_machines):
                    model.addConstr(
                        C[j] >= C[i] + processing_times[j][m]
                        - M * (1 - x[i, m])
                        - M * (1 - x[j, m])
                        - M * (1 - y[i, j])
                    )

        model.setObjective(gp.quicksum(weights[i] * C[i] for i in range(num_jobs)), GRB.MINIMIZE)

        # ---- Warm start: assign to min-processing-time machine, Smith's rule per machine ----
        assignment = {}
        for i in range(num_jobs):
            best_m = min(range(num_machines), key=lambda m: processing_times[i][m])
            assignment[i] = best_m
        for i in range(num_jobs):
            for m in range(num_machines):
                x[i, m].Start = 1.0 if assignment[i] == m else 0.0
        for i in range(num_jobs):
            for j in range(i + 1, num_jobs):
                y[i, j].Start = 0.0
                y[j, i].Start = 0.0
        for m in range(num_machines):
            jobs_on_m = [i for i in range(num_jobs) if assignment[i] == m]
            jobs_on_m.sort(key=lambda i: processing_times[i][m] / max(weights[i], 1e-6))
            t = 0
            for idx, i in enumerate(jobs_on_m):
                t += processing_times[i][m]
                C[i].Start = t
                for idx2, j in enumerate(jobs_on_m):
                    if idx < idx2:
                        y[i, j].Start = 1.0
                        y[j, i].Start = 0.0

        model.optimize(callback)

        if model.SolCount > 0:
            obj_val = model.ObjVal
            assignment = {}
            for i in range(num_jobs):
                for m in range(num_machines):
                    if x[i, m].X > 0.5:
                        assignment[i] = m
                        break
            schedule = {str(m): [] for m in range(num_machines)}
            for m in range(num_machines):
                jobs_on_m = [i for i in range(num_jobs) if assignment.get(i) == m]
                jobs_on_m.sort(key=lambda i: C[i].X)
                schedule[str(m)] = jobs_on_m
            completion_times = {str(i): C[i].X for i in range(num_jobs)}
            solution = {
                'objective_value': obj_val,
                'schedule': schedule,
                'completion_times': completion_times,
                'tardy_jobs': [],
                'on_time_jobs': list(range(num_jobs)),
            }
            if logger:
                logger.log_solution(obj_val, solution)
            return solution

        # Fallback: heuristic solution
        assignment = {}
        for i in range(num_jobs):
            best_m = min(range(num_machines), key=lambda m: processing_times[i][m])
            assignment[i] = best_m
        schedule = {str(m): [] for m in range(num_machines)}
        completion_times = {}
        total_obj = 0
        for m in range(num_machines):
            jobs_on_m = [i for i in range(num_jobs) if assignment[i] == m]
            jobs_on_m.sort(key=lambda i: processing_times[i][m] / max(weights[i], 1e-6))
            t = 0
            for i in jobs_on_m:
                t += processing_times[i][m]
                completion_times[str(i)] = t
                total_obj += weights[i] * t
            schedule[str(m)] = jobs_on_m
        solution = {
            'objective_value': total_obj,
            'schedule': schedule,
            'completion_times': completion_times,
            'tardy_jobs': [],
            'on_time_jobs': list(range(num_jobs)),
        }
        if logger:
            logger.log_solution(total_obj, solution)
        return solution

    else:  # weighted_tardy_jobs
        due_dates = instance['jobs']['due_dates']
        M = sum(max(processing_times[j]) for j in range(num_jobs)) + max(due_dates) + 1

        z = {}
        U = {}
        for i in range(num_jobs):
            U[i] = model.addVar(vtype=GRB.BINARY, name=f'U_{i}')
            for m in range(num_machines):
                z[i, m] = model.addVar(vtype=GRB.BINARY, name=f'z_{i}_{m}')

        C = {}
        for i in range(num_jobs):
            C[i] = model.addVar(lb=0, ub=M, vtype=GRB.CONTINUOUS, name=f'C_{i}')

        y = {}
        for i in range(num_jobs):
            for j in range(i + 1, num_jobs):
                y[i, j] = model.addVar(vtype=GRB.BINARY, name=f'y_{i}_{j}')
                y[j, i] = model.addVar(vtype=GRB.BINARY, name=f'y_{j}_{i}')

        model.update()
        callback.register(z=z, U=U, C=C, y=y)

        for i in range(num_jobs):
            model.addConstr(gp.quicksum(z[i, m] for m in range(num_machines)) + U[i] == 1)

        for i in range(num_jobs):
            model.addConstr(C[i] <= due_dates[i] + M * U[i])

        for i in range(num_jobs):
            model.addConstr(C[i] >= gp.quicksum(processing_times[i][m] * z[i, m] for m in range(num_machines)))

        for i in range(num_jobs):
            for j in range(i + 1, num_jobs):
                model.addConstr(y[i, j] + y[j, i] <= 1)
                for m in range(num_machines):
                    model.addConstr(y[i, j] + y[j, i] >= z[i, m] + z[j, m] - 1)

        for i in range(num_jobs):
            for j in range(num_jobs):
                if i == j:
                    continue
                for m in range(num_machines):
                    model.addConstr(
                        C[j] >= C[i] + processing_times[j][m]
                        - M * (1 - z[i, m])
                        - M * (1 - z[j, m])
                        - M * (1 - y[i, j])
                    )

        model.setObjective(gp.quicksum(weights[i] * U[i] for i in range(num_jobs)), GRB.MINIMIZE)

        # ---- Warm start: EDD greedy ----
        sorted_jobs = sorted(range(num_jobs), key=lambda i: due_dates[i])
        machine_avail = [0.0] * num_machines
        assignment = {}
        tardy_set = set()
        on_time_list = []
        for i in sorted_jobs:
            best_m = -1
            best_finish = float('inf')
            for m in range(num_machines):
                finish = machine_avail[m] + processing_times[i][m]
                if finish <= due_dates[i] and finish < best_finish:
                    best_finish = finish
                    best_m = m
            if best_m >= 0:
                assignment[i] = best_m
                machine_avail[best_m] = best_finish
                on_time_list.append(i)
            else:
                tardy_set.add(i)

        for i in range(num_jobs):
            U[i].Start = 1.0 if i in tardy_set else 0.0
            for m in range(num_machines):
                z[i, m].Start = 1.0 if assignment.get(i) == m else 0.0
        for i in range(num_jobs):
            for j in range(i + 1, num_jobs):
                y[i, j].Start = 0.0
                y[j, i].Start = 0.0
        for m in range(num_machines):
            jobs_on_m = [i for i in on_time_list if assignment.get(i) == m]
            jobs_on_m.sort(key=lambda i: due_dates[i])
            t = 0
            for idx, i in enumerate(jobs_on_m):
                t += processing_times[i][m]
                C[i].Start = t
                for idx2, j in enumerate(jobs_on_m):
                    if idx < idx2:
                        y[i, j].Start = 1.0
                        y[j, i].Start = 0.0
        for i in tardy_set:
            C[i].Start = 0.0

        model.optimize(callback)

        if model.SolCount > 0:
            obj_val = model.ObjVal
            assignment = {}
            on_time = []
            tardy = []
            for i in range(num_jobs):
                if U[i].X > 0.5:
                    tardy.append(i)
                else:
                    on_time.append(i)
                    for m in range(num_machines):
                        if z[i, m].X > 0.5:
                            assignment[i] = m
                            break
            schedule = {str(m): [] for m in range(num_machines)}
            for m in range(num_machines):
                jobs_on_m = [i for i in on_time if assignment.get(i) == m]
                jobs_on_m.sort(key=lambda i: C[i].X)
                schedule[str(m)] = jobs_on_m
            completion_times = {str(i): C[i].X for i in on_time}
            solution = {
                'objective_value': obj_val,
                'schedule': schedule,
                'completion_times': completion_times,
                'tardy_jobs': tardy,
                'on_time_jobs': on_time,
            }
            if logger:
                logger.log_solution(obj_val, solution)
            return solution

        # Fallback: all tardy
        obj_val = sum(weights)
        solution = {
            'objective_value': obj_val,
            'schedule': {str(m): [] for m in range(num_machines)},
            'completion_times': {},
            'tardy_jobs': list(range(num_jobs)),
            'on_time_jobs': [],
        }
        if logger:
            logger.log_solution(obj_val, solution)
        return solution


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', default=None)
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    solution = solve(instance, args.time_limit, args.log_path)

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)


if __name__ == '__main__':
    main()