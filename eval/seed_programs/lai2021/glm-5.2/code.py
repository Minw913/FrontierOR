def local_search(assignment, D, N, m, L, U, start_time, time_limit):
    ...
    def update_dist_sum_for_element(i):
        for g in range(m):
            s = 0.0
            for k in groups[g]:
                s += D[i][k]
            dist_sum[i][g] = s
    ...
    return assignment, current_obj, groups