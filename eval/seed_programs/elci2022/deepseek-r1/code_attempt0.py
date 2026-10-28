import argparse
import json
import time
import gurobipy as gp
from gurobipy import GRB
import sys
from solution_logger import SolutionLogger

def create_fallback_assignment(n, m, release_times, mean_processing_times, capacity_requirements, facility_capacity_limits):
    assignment = {}
    for j in range(n):
        best_f = None
        best_score = float('inf')
        for f in range(m):
            if capacity_requirements[f][j] > facility_capacity_limits[f]:
                continue
            score = release_times[j] + mean_processing_times[f][j]
            if score < best_score:
                best_score = score
                best_f = f
        if best_f is None:
            for f in range(m):
                if capacity_requirements[f][j] <= facility_capacity_limits[f]:
                    best_f = f
                    break
        assignment[j] = best_f
    return assignment

def compute_expected_makespan_scheduling(assignment, scenarios, release_times, capacity_requirements, facility_capacity_limits):
    n = len(assignment)
    m = len(facility_capacity_limits)
    num_scenarios = len(scenarios)
    expected_makespan = 0.0
    for s_idx, scenario in enumerate(scenarios):
        scenario_makespan = 0.0
        jobs_by_facility = {}
        for f in range(m):
            jobs_by_facility[f] = []
        for j in range(n):
            f = assignment[j]
            jobs_by_facility[f].append(j)
        for f in range(m):
            jobs_f = jobs_by_facility[f]
            if not jobs_f:
                continue
            jobs_f_sorted = sorted(jobs_f, key=lambda j: (release_times[j], -capacity_requirements[f][j], scenario['processing_times'][f][j]))
            timeline = [(0, float('inf'), 0)]
            for j in jobs_f_sorted:
                r_j = release_times[j]
                p_j = scenario['processing_times'][f][j]
                c_j = capacity_requirements[f][j]
                candidate_start = r_j
                while True:
                    found_violation = False
                    for seg in timeline:
                        seg_start, seg_end, usage = seg
                        if seg_end <= candidate_start:
                            continue
                        if seg_start >= candidate_start + p_j:
                            break
                        overlap_start = max(seg_start, candidate_start)
                        overlap_end = min(seg_end, candidate_start + p_j)
                        if overlap_start < overlap_end and usage + c_j > facility_capacity_limits[f]:
                            candidate_start = seg_end
                            found_violation = True
                            break
                    if not found_violation:
                        break
                finish_j = candidate_start + p_j
                new_timeline = []
                for seg in timeline:
                    s, e, u = seg
                    if e <= candidate_start:
                        new_timeline.append(seg)
                    elif s >= candidate_start + p_j:
                        new_timeline.append(seg)
                    else:
                        if s < candidate_start:
                            new_timeline.append((s, candidate_start, u))
                        mid_start = max(s, candidate_start)
                        mid_end = min(e, candidate_start + p_j)
                        new_timeline.append((mid_start, mid_end, u + c_j))
                        if e > candidate_start + p_j:
                            new_timeline.append((candidate_start + p_j, e, u))
                timeline = new_timeline
                if finish_j > scenario_makespan:
                    scenario_makespan = finish_j
        expected_makespan += scenario['probability'] * scenario_makespan
    return expected_makespan

def compute_detailed_schedule(assignment, scenarios, release_times, capacity_requirements, facility_capacity_limits):
    n = len(assignment)
    m = len(facility_capacity_limits)
    num_scenarios = len(scenarios)
    schedule = {}
    for s_idx, scenario in enumerate(scenarios):
        schedule_scenario = {}
        jobs_by_facility = {}
        for f in range(m):
            jobs_by_facility[f] = []
        for j in range(n):
            f = assignment[j]
            jobs_by_facility[f].append(j)
        for f in range(m):
            jobs_f = jobs_by_facility[f]
            if not jobs_f:
                continue
            jobs_f_sorted = sorted(jobs_f, key=lambda j: (release_times[j], -capacity_requirements[f][j], scenario['processing_times'][f][j]))
            timeline = [(0, float('inf'), 0)]
            for j in jobs_f_sorted:
                r_j = release_times[j]
                p_j = scenario['processing_times'][f][j]
                c_j = capacity_requirements[f][j]
                candidate_start = r_j
                while True:
                    found_violation = False
                    for seg in timeline:
                        seg_start, seg_end, usage = seg
                        if seg_end <= candidate_start:
                            continue
                        if seg_start >= candidate_start + p_j:
                            break
                        overlap_start = max(seg_start, candidate_start)
                        overlap_end = min(seg_end, candidate_start + p_j)
                        if overlap_start < overlap_end and usage + c_j > facility_capacity_limits[f]:
                            candidate_start = seg_end
                            found_violation = True
                            break
                    if not found_violation:
                        break
                finish_j = candidate_start + p_j
                new_timeline = []
                for seg in timeline:
                    s, e, u = seg
                    if e <= candidate_start:
                        new_timeline.append(seg)
                    elif s >= candidate_start + p_j:
                        new_timeline.append(seg)
                    else:
                        if s < candidate_start:
                            new_timeline.append((s, candidate_start, u))
                        mid_start = max(s, candidate_start)
                        mid_end = min(e, candidate_start + p_j)
                        new_timeline.append((mid_start, mid_end, u + c_j))
                        if e > candidate_start + p_j:
                            new_timeline.append((candidate_start + p_j, e, u))
                timeline = new_timeline
                schedule_scenario[j] = {
                    "facility": f,
                    "start": candidate_start,
                    "finish": finish_j
                }
        schedule[str(s_idx)] = schedule_scenario
    return schedule

def main(instance_path, solution_path, time_limit, log_path=None):
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    time_start = time.time()

    with open(instance_path, 'r') as f:
        data = json.load(f)
    
    n = data['n']
    m = data['m']
    num_scenarios = data['num_scenarios']
    facility_capacity_limits = data['facility_capacity_limits']
    capacity_requirements = data['capacity_requirements']
    release_times = data['release_times']
    mean_processing_times = data['mean_processing_times']
    scenarios = data['scenarios']

    allowed = [[True] * m for _ in range(n)]
    for j in range(n):
        for f in range(m):
            if capacity_requirements[f][j] > facility_capacity_limits[f]:
                allowed[j][f] = False

    BigM = [0] * n
    for j in range(n):
        max_val = 0
        for f in range(m):
            for scenario in scenarios:
                p_jsf = scenario['processing_times'][f][j]
                val = release_times[j] + p_jsf
                if val > max_val:
                    max_val = val
        BigM[j] = max_val

    best_actual_objective = float('inf')
    best_assignment = None

    def extract_assignment_from_model(model, x, n, m, allowed):
        assignment = {}
        for j in range(n):
            for f in range(m):
                if allowed[j][f]:
                    x_val = model.cbGetSolution(x[j, f])
                    if x_val > 0.5:
                        assignment[j] = f
                        break
            if j not in assignment:
                for f in range(m):
                    if allowed[j][f]:
                        assignment[j] = f
                        break
        return assignment

    def callback_wrapper(model, where, x, n, m, allowed, scenarios, release_times, capacity_requirements, facility_capacity_limits, time_start, time_limit, logger):
        nonlocal best_actual_objective, best_assignment
        if where == GRB.Callback.MIPSOL:
            time_elapsed = time.time() - time_start
            if time_elapsed >= time_limit:
                model.terminate()
                return
            try:
                assignment_incumbent = extract_assignment_from_model(model, x, n, m, allowed)
                expected_makespan_actual = compute_expected_makespan_scheduling(
                    assignment_incumbent, scenarios, release_times, capacity_requirements, facility_capacity_limits
                )
                if expected_makespan_actual < best_actual_objective:
                    best_actual_objective = expected_makespan_actual
                    best_assignment = assignment_incumbent
                    if logger:
                        logger.log(best_actual_objective)
            except Exception as e:
                print(f"Error in callback: {e}", file=sys.stderr)

    try:
        model = gp.Model("assignment")
        model.setParam('OutputFlag', 0)

        x = {}
        for j in range(n):
            for f in range(m):
                if allowed[j][f]:
                    x[j, f] = model.addVar(vtype=GRB.BINARY, name=f"x_{j}_{f}")
                else:
                    x[j, f] = None

        M_fs = {}
        M_s = {}
        for s_idx in range(num_scenarios):
            M_s[s_idx] = model.addVar(vtype=GRB.CONTINUOUS, name=f"M_s_{s_idx}")
            for f in range(m):
                M_fs[f, s_idx] = model.addVar(vtype=GRB.CONTINUOUS, name=f"M_fs_{f}_{s_idx}")

        for j in range(n):
            model.addConstr(gp.quicksum(x[j, f] for f in range(m) if allowed[j][f]) == 1, name=f"assign_job_{j}")

        for s_idx, scenario in enumerate(scenarios):
            for f in range(m):
                for j in range(n):
                    if allowed[j][f]:
                        p_jsf = scenario['processing_times'][f][j]
                        rhs = release_times[j] + p_jsf
                        model.addConstr(M_fs[f, s_idx] >= rhs - BigM[j] * (1 - x[j, f]), 
                                      name=f"constr1_j{j}_f{f}_s{s_idx}")

        for s_idx, scenario in enumerate(scenarios):
            for f in range(m):
                expr = gp.LinExpr()
                for j in range(n):
                    if allowed[j][f]:
                        p_jsf = scenario['processing_times'][f][j]
                        expr += x[j, f] * p_jsf * capacity_requirements[f][j]
                model.addConstr(M_fs[f, s_idx] >= expr / facility_capacity_limits[f], 
                              name=f"constr2_f{f}_s{s_idx}")

        for s_idx in range(num_scenarios):
            for f in range(m):
                model.addConstr(M_s[s_idx] >= M_fs[f, s_idx], name=f"max_constr_s{s_idx}_f{f}")

        obj = gp.LinExpr()
        for s_idx, scenario in enumerate(scenarios):
            prob = scenario['probability']
            obj += prob * M_s[s_idx]
        model.setObjective(obj, GRB.MINIMIZE)

        time_elapsed_so_far = time.time() - time_start
        time_remaining_total = time_limit - time_elapsed_so_far
        time_for_mip = max(1, time_remaining_total - 10)
        model.setParam('TimeLimit', time_for_mip)

        if logger:
            model.optimize(lambda model, where: callback_wrapper(
                model, where, x, n, m, allowed, scenarios, release_times, capacity_requirements, 
                facility_capacity_limits, time_start, time_limit, logger
            ))
        else:
            model.optimize()

    except gp.GurobiError as e:
        print(f"Gurobi error: {e}", file=sys.stderr)
    except Exception as e:
        print(f"Error during MIP: {e}", file=sys.stderr)

    time_after_mip = time.time()
    time_elapsed_after_mip = time_after_mip - time_start
    time_remaining_after_mip = time_limit - time_elapsed_after_mip

    if best_assignment is None:
        if model.status == GRB.INFEASIBLE or model.status == GRB.INF_OR_UNBD:
            fallback_assignment = create_fallback_assignment(
                n, m, release_times, mean_processing_times, capacity_requirements, facility_capacity_limits
            )
            fallback_actual = compute_expected_makespan_scheduling(
                fallback_assignment, scenarios, release_times, capacity_requirements, facility_capacity_limits
            )
            if fallback_actual < best_actual_objective:
                best_actual_objective = fallback_actual
                best_assignment = fallback_assignment
                if logger:
                    logger.log(best_actual_objective)
        elif model.status == GRB.TIME_LIMIT or model.status == GRB.INTERRUPTED or model.status == GRB.OPTIMAL:
            try:
                assignment_final = {}
                for j in range(n):
                    for f in range(m):
                        if allowed[j][f]:
                            if x[j, f].X > 0.5:
                                assignment_final[j] = f
                                break
                    if j not in assignment_final:
                        for f in range(m):
                            if allowed[j][f]:
                                assignment_final[j] = f
                                break
                final_actual = compute_expected_makespan_scheduling(
                    assignment_final, scenarios, release_times, capacity_requirements, facility_capacity_limits
                )
                if final_actual < best_actual_objective:
                    best_actual_objective = final_actual
                    best_assignment = assignment_final
                    if logger:
                        logger.log(best_actual_objective)
            except:
                fallback_assignment = create_fallback_assignment(
                    n, m, release_times, mean_processing_times, capacity_requirements, facility_capacity_limits
                )
                fallback_actual = compute_expected_makespan_scheduling(
                    fallback_assignment, scenarios, release_times, capacity_requirements, facility_capacity_limits
                )
                if fallback_actual < best_actual_objective:
                    best_actual_objective = fallback_actual
                    best_assignment = fallback_assignment
                    if logger:
                        logger.log(best_actual_objective)
        else:
            fallback_assignment = create_fallback_assignment(
                n, m, release_times, mean_processing_times, capacity_requirements, facility_capacity_limits
            )
            fallback_actual = compute_expected_makespan_scheduling(
                fallback_assignment, scenarios, release_times, capacity_requirements, facility_capacity_limits
            )
            if fallback_actual < best_actual_objective:
                best_actual_objective = fallback_actual
                best_assignment = fallback_assignment
                if logger:
                    logger.log(best_actual_objective)
    else:
        pass

    if best_assignment is None:
        fallback_assignment = create_fallback_assignment(
            n, m, release_times, mean_processing_times, capacity_requirements, facility_capacity_limits
        )
        best_assignment = fallback_assignment
        best_actual_objective = compute_expected_makespan_scheduling(
            best_assignment, scenarios, release_times, capacity_requirements, facility_capacity_limits
        )
        if logger:
            logger.log(best_actual_objective)

    detailed_schedule = compute_detailed_schedule(
        best_assignment, scenarios, release_times, capacity_requirements, facility_capacity_limits
    )

    assignment_output = {str(j): best_assignment[j] for j in range(n)}
    schedule_output = {}
    for s_idx in range(num_scenarios):
        scenario_key = str(s_idx)
        schedule_scenario = detailed_schedule.get(scenario_key, {})
        schedule_output[scenario_key] = {str(j): schedule_scenario[j] for j in schedule_scenario}

    solution = {
        "objective_value": best_actual_objective,
        "assignment": assignment_output,
        "schedule": schedule_output
    }

    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    main(args.instance_path, args.solution_path, args.time_limit, args.log_path)