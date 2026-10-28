import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from solution_logger import SolutionLogger


class ScheduleBuilder:
    def __init__(self, instance: Dict[str, Any]):
        self.S = int(instance["S"])
        self.p = int(instance["p"])
        self.max_slot = self.S + 1

        self.w = int(instance["sigma_w"])
        self.l = int(instance["sigma_l"])
        if not (0 <= self.w < self.l <= self.max_slot):
            raise ValueError("Initial crane positions must satisfy 0 <= sigma_w < sigma_l <= S+1")

        self.t = 0
        self.w_positions: List[int] = [self.w]
        self.l_positions: List[int] = [self.l]

        self.seaside_events: List[Dict[str, Any]] = []
        self.landside_events: List[Dict[str, Any]] = []

        self.deadline_violation = 0
        self.valid = True

    def step(self, dw: int = 0, dl: int = 0) -> None:
        nw = self.w + dw
        nl = self.l + dl

        if abs(dw) > 1 or abs(dl) > 1:
            raise ValueError("A crane may move by at most one slot per time unit")
        if not (0 <= nw <= self.max_slot and 0 <= nl <= self.max_slot):
            raise ValueError("Crane movement outside block bounds")
        if nw >= nl:
            raise ValueError(
                f"Crane separation violated at time {self.t + 1}: W={nw}, L={nl}"
            )

        self.w = nw
        self.l = nl
        self.t += 1
        self.w_positions.append(self.w)
        self.l_positions.append(self.l)

    def hold(self, duration: int) -> None:
        for _ in range(duration):
            self.step(0, 0)

    def move_w_to(self, target: int) -> None:
        while self.w != target:
            direction = 1 if target > self.w else -1
            self.step(direction, 0)

    def move_l_to(self, target: int) -> None:
        while self.l != target:
            direction = 1 if target > self.l else -1
            self.step(0, direction)

    def park_w_at_zero(self) -> None:
        self.move_w_to(0)

    def park_l_at_landside(self) -> None:
        self.move_l_to(self.max_slot)

    def process_landside(self, container: Dict[str, Any]) -> None:
        cid = int(container["id"])
        source = int(container["source_slot"])
        earliest = int(container["earliest_finish_time"])
        deadline = int(container["deadline"])

        self.move_l_to(source)
        lift_time = self.t
        self.hold(self.p)

        self.move_l_to(self.max_slot)

        # Interpret the stated time window as a window on completion of the
        # drop operation.
        required_wait = earliest - (self.t + self.p)
        if required_wait > 0:
            self.hold(required_wait)

        drop_time = self.t
        self.hold(self.p)
        completion = self.t

        if completion < earliest:
            self.valid = False
            self.deadline_violation += earliest - completion
        if completion > deadline:
            self.valid = False
            self.deadline_violation += completion - deadline

        self.landside_events.append(
            {
                "container_id": cid,
                "source_slot": source,
                "lift_time": lift_time,
                "drop_time": drop_time,
            }
        )

    def process_seaside_direct(self, container: Dict[str, Any]) -> int:
        cid = int(container["id"])
        target = int(container["target_slot"])

        self.move_w_to(0)
        self.hold(self.p)  # Lift at slot zero.
        self.move_w_to(target)

        drop_time = self.t
        self.hold(self.p)

        self.seaside_events.append(
            {
                "container_id": cid,
                "target_slot": target,
                "seaside_drop_time": drop_time,
                "seaside_drop_slot": target,
            }
        )
        return self.t

    def process_seaside_cooperative(self, container: Dict[str, Any]) -> int:
        cid = int(container["id"])
        target = int(container["target_slot"])

        if self.S < 1:
            raise ValueError("Cooperative handling requires at least one storage slot")

        # For a target at S+1, use slot S as the intermediate handover slot.
        handover = self.S

        self.move_w_to(0)
        self.hold(self.p)  # Seaside lift at slot zero.
        self.move_w_to(handover)

        seaside_drop_time = self.t
        self.hold(self.p)

        # W retreats as L enters the handover slot.
        self.step(-1, -1)
        landside_lift_time = self.t

        # L lifts while W continues retreating when possible.
        for _ in range(self.p):
            self.step(-1 if self.w > 0 else 0, 0)

        # L moves to the landside handover point.
        self.step(-1 if self.w > 0 else 0, 1)
        landside_drop_time = self.t

        for _ in range(self.p):
            self.step(-1 if self.w > 0 else 0, 0)

        self.seaside_events.append(
            {
                "container_id": cid,
                "target_slot": target,
                "seaside_drop_time": seaside_drop_time,
                "seaside_drop_slot": handover,
                "landside_lift_time": landside_lift_time,
                "landside_lift_slot": handover,
                "landside_drop_time": landside_drop_time,
            }
        )
        return self.t

    def process_all_seaside(self, containers: List[Dict[str, Any]]) -> int:
        if not containers:
            return 0

        completion = self.t
        for container in containers:
            target = int(container["target_slot"])
            if target == self.max_slot:
                completion = self.process_seaside_cooperative(container)
            else:
                completion = self.process_seaside_direct(container)
        return completion

    def solution(self, objective: int) -> Dict[str, Any]:
        return {
            "objective_value": float(objective),
            "crane_w_positions": {
                str(t): int(position) for t, position in enumerate(self.w_positions)
            },
            "crane_l_positions": {
                str(t): int(position) for t, position in enumerate(self.l_positions)
            },
            "seaside_events": self.seaside_events,
            "landside_events": self.landside_events,
        }


def required_landside_count(
    landside: List[Dict[str, Any]], makespan: int
) -> int:
    """
    With containers sorted by non-decreasing deadline, return a prefix size
    that includes every container due by the makespan and one container from
    the next deadline group when such a group exists.
    """
    due_count = 0
    for container in landside:
        if int(container["deadline"]) <= makespan:
            due_count += 1
        else:
            break

    if due_count < len(landside):
        return due_count + 1
    return due_count


def build_candidate(
    instance: Dict[str, Any],
    pre_count: int,
) -> Tuple[Dict[str, Any], bool, int]:
    seaside = sorted(
        instance.get("seaside_containers", []),
        key=lambda c: int(c["id"]),
    )
    landside = sorted(
        instance.get("landside_containers", []),
        key=lambda c: (int(c["deadline"]), int(c["id"])),
    )

    builder = ScheduleBuilder(instance)
    builder.park_w_at_zero()

    # Process a prefix before seaside operations. Increasing this prefix can
    # rescue tight landside deadlines, at the cost of a larger makespan.
    actual_pre_count = min(pre_count, len(landside))
    for container in landside[:actual_pre_count]:
        builder.process_landside(container)

    builder.park_l_at_landside()
    makespan = builder.process_all_seaside(seaside)

    required_count = required_landside_count(landside, makespan)
    processed_count = max(actual_pre_count, required_count)
    processed_count = min(processed_count, len(landside))

    # The remaining required landside work may occur after the seaside
    # makespan, provided its individual time windows are respected.
    if processed_count > actual_pre_count:
        builder.park_w_at_zero()
        for container in landside[actual_pre_count:processed_count]:
            builder.process_landside(container)

    solution = builder.solution(makespan)
    return solution, builder.valid, builder.deadline_violation


def write_solution(path: str, solution: Dict[str, Any]) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(solution, handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", required=True, type=int)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    start_time = time.monotonic()
    deadline = start_time + max(1, args.time_limit)

    with open(args.instance_path, "r", encoding="utf-8") as handle:
        instance = json.load(handle)

    instance["seaside_containers"] = sorted(
        instance.get("seaside_containers", []),
        key=lambda c: int(c["id"]),
    )
    instance["landside_containers"] = sorted(
        instance.get("landside_containers", []),
        key=lambda c: (int(c["deadline"]), int(c["id"])),
    )

    m = len(instance["landside_containers"])

    best_solution: Optional[Dict[str, Any]] = None
    best_objective = float("inf")

    fallback_solution: Optional[Dict[str, Any]] = None
    fallback_score: Optional[Tuple[int, float]] = None

    # First construct a robust incumbent with all landside jobs before the
    # seaside sequence. It is often feasible even under tight deadlines.
    initial_counts = [m]
    if m == 0:
        initial_counts = [0]

    tried = set()
    for pre_count in initial_counts:
        if time.monotonic() >= deadline:
            break
        tried.add(pre_count)
        try:
            solution, valid, violation = build_candidate(instance, pre_count)
        except (ValueError, RuntimeError):
            continue

        objective = float(solution["objective_value"])
        score = (violation, objective)
        if fallback_score is None or score < fallback_score:
            fallback_score = score
            fallback_solution = solution

        if valid and objective < best_objective:
            best_objective = objective
            best_solution = solution
            if logger:
                logger.log_solution(objective, solution)

    # Search from zero upward. Since every additional pre-makespan landside
    # job delays the seaside sequence, the first feasible candidate in this
    # order is normally the best candidate under this scheduling policy.
    for pre_count in range(0, m + 1):
        if pre_count in tried:
            continue
        if time.monotonic() >= deadline:
            break

        try:
            solution, valid, violation = build_candidate(instance, pre_count)
        except (ValueError, RuntimeError):
            continue

        objective = float(solution["objective_value"])
        score = (violation, objective)
        if fallback_score is None or score < fallback_score:
            fallback_score = score
            fallback_solution = solution

        if valid and objective < best_objective:
            best_objective = objective
            best_solution = solution
            if logger:
                logger.log_solution(objective, solution)

            # Subsequent candidates perform more work before the seaside
            # sequence and therefore cannot improve this policy's makespan.
            break

    if best_solution is None:
        if fallback_solution is None:
            # Last-resort construction, also covering extremely short limits.
            fallback_solution, _, _ = build_candidate(instance, m)
        best_solution = fallback_solution
        if logger:
            logger.log_solution(
                float(best_solution["objective_value"]),
                best_solution,
            )

    write_solution(args.solution_path, best_solution)


if __name__ == "__main__":
    main()