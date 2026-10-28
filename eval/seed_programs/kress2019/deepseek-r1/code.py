import argparse
import json
import time
from solution_logger import SolutionLogger

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    return parser.parse_args()

def load_instance(instance_path):
    with open(instance_path, 'r') as f:
        data = json.load(f)
    return data

def get_position_at_time(timeline, t):
    if not timeline:
        return None
    if t < timeline[0][0]:
        return timeline[0][2]
    for seg in timeline:
        start_time, end_time, start_slot, end_slot, activity_type, container_id = seg
        if start_time <= t <= end_time:
            if activity_type == 'move':
                if end_slot >= start_slot:
                    return start_slot + (t - start_time)
                else:
                    return start_slot - (t - start_time)
            else:
                return start_slot
    last_seg = timeline[-1]
    return last_seg[3]

def check_activity(crane, new_seg, other_crane_timeline):
    start_time, end_time, start_slot, end_slot, activity_type, container_id = new_seg
    for t in range(start_time, end_time + 1):
        if activity_type == 'move':
            if end_slot >= start_slot:
                pos = start_slot + (t - start_time)
            else:
                pos = start_slot - (t - start_time)
        else:
            pos = start_slot
        other_pos = get_position_at_time(other_crane_timeline, t)
        if other_pos is None:
            return False, t
        if crane == 'W':
            if not (pos < other_pos):
                return False, t
        else:
            if not (other_pos < pos):
                return False, t
    return True, None

def schedule_initial_move_and_lift(crane_timeline, available_time, current_slot, p, container_id, target_slot, is_seaside=True):
    segments = []
    if current_slot != 0:
        move_duration = abs(current_slot - 0)
        move_start = available_time
        move_end = move_start + move_duration - 1
        move_seg = (move_start, move_end, current_slot, 0, 'move', container_id)
        segments.append(move_seg)
        lift_start = move_end + 1
        current_slot_after_move = 0
    else:
        lift_start = available_time
        current_slot_after_move = 0
    lift_end = lift_start + p - 1
    lift_seg = (lift_start, lift_end, 0, 0, 'lift', container_id)
    segments.append(lift_seg)
    end_time = lift_end
    new_current_slot = 0
    return segments, lift_end + 1, new_current_slot

def schedule_move_and_drop(available_time, current_slot, target_slot, p, container_id):
    segments = []
    if current_slot != target_slot:
        move_duration = abs(current_slot - target_slot)
        move_start = available_time
        move_end = move_start + move_duration - 1
        move_seg = (move_start, move_end, current_slot, target_slot, 'move', container_id)
        segments.append(move_seg)
        drop_start = move_end + 1
        current_slot_after_move = target_slot
    else:
        drop_start = available_time
        current_slot_after_move = current_slot
    drop_end = drop_start + p - 1
    drop_seg = (drop_start, drop_end, target_slot, target_slot, 'drop', container_id)
    segments.append(drop_seg)
    return segments, drop_end + 1, current_slot_after_move

def schedule_landside_for_handover(available_time_l, current_slot_l, s_i, target_i, drop_start_w, p, container_id, craneL_timeline, craneW_timeline):
    move_start_l = available_time_l
    move_duration1 = abs(current_slot_l - s_i)
    move_end_l = move_start_l + move_duration1 - 1
    move_seg1 = (move_start_l, move_end_l, current_slot_l, s_i, 'move', container_id)
    lift_start_l = max(move_end_l + 1, drop_start_w)
    lift_end_l = lift_start_l + p - 1
    lift_seg = (lift_start_l, lift_end_l, s_i, s_i, 'lift', container_id)
    move_duration2 = abs(s_i - target_i)
    move_start_l2 = lift_end_l + 1
    move_end_l2 = move_start_l2 + move_duration2 - 1
    move_seg2 = (move_start_l2, move_end_l2, s_i, target_i, 'move', container_id)
    drop_start_l = move_end_l2 + 1
    drop_end_l = drop_start_l + p - 1
    drop_seg = (drop_start_l, drop_end_l, target_i, target_i, 'drop', container_id)
    new_segments_l = [move_seg1, lift_seg, move_seg2, drop_seg]
    for seg in new_segments_l:
        valid, conflict_t = check_activity('L', seg, craneW_timeline)
        if not valid:
            return None, None, None, None
        valid_self, _ = check_activity('L', seg, craneL_timeline)
        if not valid_self:
            return None, None, None, None
    return new_segments_l, drop_end_l + 1, target_i, drop_start_l

def main():
    args = parse_arguments()
    start_time_global = time.time()
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    instance = load_instance(args.instance_path)
    S = instance['S']
    n = instance['n']
    m = instance['m']
    p_val = instance['p']
    sigma_w = instance['sigma_w']
    sigma_l = instance['sigma_l']
    seaside_containers = sorted(instance['seaside_containers'], key=lambda x: x['id'])
    landside_containers = sorted(instance['landside_containers'], key=lambda x: (x['deadline'], x['id']))
    T_max = 10000
    craneW_timeline = []
    craneL_timeline = []
    craneW_available_time = 0
    craneW_current_slot = sigma_w
    craneL_available_time = 0
    craneL_current_slot = sigma_l
    if craneW_available_time == 0 and craneW_current_slot == sigma_w:
        seg0 = (0, 0, sigma_w, sigma_w, 'wait', None)
        craneW_timeline.append(seg0)
    if craneL_available_time == 0 and craneL_current_slot == sigma_l:
        seg0 = (0, 0, sigma_l, sigma_l, 'wait', None)
        craneL_timeline.append(seg0)
    seaside_events = []
    landside_events = []
    scheduled_landside = [False] * len(landside_containers)
    def check_time():
        return time.time() - start_time_global < args.time_limit
    for i, sc in enumerate(seaside_containers):
        if not check_time():
            break
        container_id = sc['id']
        target_i = sc['target_slot']
        segments_w_step1, lift_end_time, new_slot_w = schedule_initial_move_and_lift(
            craneW_timeline, craneW_available_time, craneW_current_slot, p_val, container_id, target_i
        )
        temp_craneW_timeline = craneW_timeline + segments_w_step1
        candidate_slots = []
        candidate_slots.append(craneL_current_slot)
        candidate_slots.append(1)
        candidate_slots.append(S)
        candidate_slots.append(target_i // 2)
        candidate_slots.append((craneL_current_slot + target_i) // 2)
        candidate_slots = [s for s in candidate_slots if 0 <= s <= S]
        candidate_slots = list(set(candidate_slots))
        best_option = None
        best_delivery_time = None
        best_segments_w = None
        best_segments_l = None
        best_s_i = None
        best_end_time_w = None
        best_end_slot_w = None
        best_end_time_l = None
        best_end_slot_l = None
        direct_segments_w, end_time_w_direct, end_slot_w_direct = schedule_move_and_drop(
            lift_end_time, 0, target_i, p_val, container_id
        )
        all_segments_w_direct = segments_w_step1 + direct_segments_w
        valid_direct = True
        for seg in all_segments_w_direct:
            valid, _ = check_activity('W', seg, craneL_timeline)
            if not valid:
                valid_direct = False
                break
        if valid_direct:
            delivery_time_direct = direct_segments_w[-1][0]
            best_option = 'direct'
            best_delivery_time = delivery_time_direct
            best_segments_w = all_segments_w_direct
            best_s_i = None
            best_end_time_w = end_time_w_direct
            best_end_slot_w = end_slot_w_direct
            best_segments_l = []
            best_end_time_l = craneL_available_time
            best_end_slot_l = craneL_current_slot
        for s_i in candidate_slots:
            if s_i == target_i:
                continue
            segments_w_handover, end_time_w_handover, end_slot_w_handover = schedule_move_and_drop(
                lift_end_time, 0, s_i, p_val, container_id
            )
            all_segments_w_handover = segments_w_step1 + segments_w_handover
            valid_handover_w = True
            for seg in all_segments_w_handover:
                valid, _ = check_activity('W', seg, craneL_timeline)
                if not valid:
                    valid_handover_w = False
                    break
            if not valid_handover_w:
                continue
            drop_start_w = segments_w_handover[-1][0]
            new_segments_l, end_time_l_handover, end_slot_l_handover, delivery_time_handover = schedule_landside_for_handover(
                craneL_available_time, craneL_current_slot, s_i, target_i, drop_start_w, p_val, container_id, craneL_timeline, temp_craneW_timeline + segments_w_handover
            )
            if new_segments_l is None:
                continue
            if best_delivery_time is None or delivery_time_handover < best_delivery_time:
                best_option = 'handover'
                best_delivery_time = delivery_time_handover
                best_segments_w = all_segments_w_handover
                best_s_i = s_i
                best_end_time_w = end_time_w_handover
                best_end_slot_w = end_slot_w_handover
                best_segments_l = new_segments_l
                best_end_time_l = end_time_l_handover
                best_end_slot_l = end_slot_l_handover
        if best_option is None:
            continue
        craneW_timeline.extend(best_segments_w)
        craneW_available_time = best_end_time_w
        craneW_current_slot = best_end_slot_w
        if best_option == 'handover':
            craneL_timeline.extend(best_segments_l)
            craneL_available_time = best_end_time_l
            craneL_current_slot = best_end_slot_l
            event = {
                'container_id': container_id,
                'target_slot': target_i,
                'seaside_drop_time': best_segments_w[-1][0],
                'seaside_drop_slot': best_s_i,
                'landside_lift_time': best_segments_l[1][0],
                'landside_lift_slot': best_s_i,
                'landside_drop_time': best_segments_l[-1][0]
            }
        else:
            event = {
                'container_id': container_id,
                'target_slot': target_i,
                'seaside_drop_time': best_segments_w[-1][0],
                'seaside_drop_slot': target_i
            }
        seaside_events.append(event)
        if logger and seaside_events:
            delivery_times = [e['seaside_drop_time'] if 'landside_drop_time' not in e else e['landside_drop_time'] for e in seaside_events]
            makespan = max(delivery_times) if delivery_times else 0
            logger.log(makespan)
        for idx, lsc in enumerate(landside_containers):
            if scheduled_landside[idx]:
                continue
            if lsc['deadline'] > T_max:
                continue
            source_slot = lsc['source_slot']
            eft = lsc['earliest_finish_time']
            deadline = lsc['deadline']
            segments = []
            if craneL_current_slot != source_slot:
                move_duration1 = abs(craneL_current_slot - source_slot)
                move_start = craneL_available_time
                move_end = move_start + move_duration1 - 1
                move_seg1 = (move_start, move_end, craneL_current_slot, source_slot, 'move', lsc['id'])
                segments.append(move_seg1)
                lift_start = move_end + 1
            else:
                lift_start = craneL_available_time
            lift_end = lift_start + p_val - 1
            lift_seg = (lift_start, lift_end, source_slot, source_slot, 'lift', lsc['id'])
            segments.append(lift_seg)
            move_duration2 = abs(source_slot - (S+1))
            move_start2 = lift_end + 1
            move_end2 = move_start2 + move_duration2 - 1
            move_seg2 = (move_start2, move_end2, source_slot, S+1, 'move', lsc['id'])
            segments.append(move_seg2)
            drop_start = move_end2 + 1
            drop_end = drop_start + p_val - 1
            drop_seg = (drop_start, drop_end, S+1, S+1, 'drop', lsc['id'])
            segments.append(drop_seg)
            if drop_start < eft:
                wait_start = drop_start
                wait_end = eft - 1
                wait_seg = (wait_start, wait_end, S+1, S+1, 'wait', lsc['id'])
                segments.insert(-1, wait_seg)
                drop_seg = (eft, eft + p_val - 1, S+1, S+1, 'drop', lsc['id'])
                segments[-1] = drop_seg
                drop_start_actual = eft
            else:
                drop_start_actual = drop_start
            if drop_start_actual > deadline:
                continue
            valid_all = True
            for seg in segments:
                valid, _ = check_activity('L', seg, craneW_timeline)
                if not valid:
                    valid_all = False
                    break
                valid_self, _ = check_activity('L', seg, craneL_timeline)
                if not valid_self:
                    valid_all = False
                    break
            if not valid_all:
                continue
            craneL_timeline.extend(segments)
            last_seg = segments[-1]
            craneL_available_time = last_seg[1] + 1
            craneL_current_slot = S+1
            scheduled_landside[idx] = True
            landside_events.append({
                'container_id': lsc['id'],
                'source_slot': source_slot,
                'lift_time': lift_start,
                'drop_time': drop_start_actual
            })
    delivery_times = []
    for event in seaside_events:
        if 'landside_drop_time' in event:
            delivery_times.append(event['landside_drop_time'])
        else:
            delivery_times.append(event['seaside_drop_time'])
    makespan = max(delivery_times) if delivery_times else 0
    all_times = set()
    for seg in craneW_timeline:
        all_times.add(seg[0])
        all_times.add(seg[1])
    for seg in craneL_timeline:
        all_times.add(seg[0])
        all_times.add(seg[1])
    if all_times:
        T_end = max(all_times)
    else:
        T_end = 0
    crane_w_positions = {}
    crane_l_positions = {}
    for t in range(0, T_end + 1):
        pos_w = get_position_at_time(craneW_timeline, t)
        pos_l = get_position_at_time(craneL_timeline, t)
        crane_w_positions[str(t)] = pos_w
        crane_l_positions[str(t)] = pos_l
    output = {
        "objective_value": makespan,
        "crane_w_positions": crane_w_positions,
        "crane_l_positions": crane_l_positions,
        "seaside_events": seaside_events,
        "landside_events": landside_events
    }
    with open(args.solution_path, 'w') as f:
        json.dump(output, f, indent=2)

if __name__ == "__main__":
    main()