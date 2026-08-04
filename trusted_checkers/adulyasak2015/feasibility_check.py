#!/usr/bin/env python3
"""Trusted objective wrapper for the Adulyasak 2-SPRP checker.

The public solution contract omits the scenario-dependent recourse variables.
The dataset checker can therefore validate the first-stage routing decisions
but only bounds the reported objective. This wrapper delegates those structural
checks, then fixes the submitted first-stage decisions and solves the continuous
recourse problem to certify the exact expected cost.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from typing import Any

import gurobipy as gp
from gurobipy import GRB


ABS_TOL = 1e-4
REL_TOL = 1e-8


def _load_object(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("JSON payload must be an object")
    return value


def _write_result(path: str, *, feasible: bool, violations: list[str]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "feasible": feasible,
                "violated_constraints": [] if feasible else ["trusted_objective"],
                "violations": violations,
                "violation_magnitudes": [],
            },
            handle,
            indent=2,
        )


def _run_dataset_checker(
    instance_path: str,
    solution_path: str,
) -> dict[str, Any]:
    paper_dir = os.path.dirname(os.path.dirname(os.path.abspath(instance_path)))
    checker = os.path.join(paper_dir, "feasibility_check.py")
    if not os.path.isfile(checker):
        raise ValueError("dataset feasibility checker is missing")
    with tempfile.TemporaryDirectory(prefix="frontieror_adulyasak_base_") as tmp:
        result_path = os.path.join(tmp, "result.json")
        completed = subprocess.run(
            [
                sys.executable,
                checker,
                "--instance_path",
                instance_path,
                "--solution_path",
                solution_path,
                "--result_path",
                result_path,
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if completed.returncode != 0:
            raise ValueError("dataset feasibility checker failed")
        return _load_object(result_path)


def _binary(mapping: dict[str, Any], key: str) -> int:
    value = mapping.get(key, 0)
    if isinstance(value, bool):
        return int(value)
    number = float(value)
    rounded = round(number)
    if not math.isfinite(number) or abs(number - rounded) > 1e-6 or rounded not in (0, 1):
        raise ValueError(f"{key} must be binary")
    return int(rounded)


def _edge_value(mapping: dict[str, Any], key: str, depot_edge: bool) -> int:
    value = mapping.get(key, 0)
    if isinstance(value, bool):
        value = int(value)
    number = float(value)
    rounded = round(number)
    upper = 2 if depot_edge else 1
    if (
        not math.isfinite(number)
        or abs(number - rounded) > 1e-6
        or rounded < 0
        or rounded > upper
    ):
        raise ValueError(f"{key} must be an integer in [0, {upper}]")
    return int(rounded)


def _certified_objective(
    instance: dict[str, Any],
    solution: dict[str, Any],
) -> float:
    n = int(instance["n"])
    horizon = int(instance["T"])
    vehicle_count = int(instance["m"])
    vehicle_capacity = float(instance["Q"])
    production_capacity = float(instance["C"])
    setup_unit_cost = float(instance["f"])
    production_unit_cost = float(instance["u"])
    holding_costs = [float(value) for value in instance["h"]]
    inventory_capacities = [float(value) for value in instance["L"]]
    initial_inventory = [float(value) for value in instance["I0"]]
    penalty_costs = [float(value) for value in instance["sigma"]]
    transportation_costs = instance["transportation_costs"]
    probabilities = [float(value) for value in instance["scenario_probabilities"]]
    demands = instance["demand_scenarios"]
    scenario_count = int(instance["n_scenarios"])

    customers = range(1, n + 1)
    nodes = range(0, n + 1)
    periods = range(1, horizon + 1)
    vehicles = range(1, vehicle_count + 1)
    scenarios = range(scenario_count)
    edges = [(i, j) for i in nodes for j in nodes if i < j]

    y_raw = solution.get("y")
    z_raw = solution.get("z")
    x_raw = solution.get("x")
    if not isinstance(y_raw, dict) or not isinstance(z_raw, dict) or not isinstance(x_raw, dict):
        raise ValueError("solution y, z, and x must be objects")
    y = {t: _binary(y_raw, str(t)) for t in periods}
    z = {
        (i, k, t): _binary(z_raw, f"{i}_{k}_{t}")
        for i in nodes
        for k in vehicles
        for t in periods
    }
    x = {
        (i, j, k, t): _edge_value(x_raw, f"{i}_{j}_{k}_{t}", i == 0)
        for i, j in edges
        for k in vehicles
        for t in periods
    }

    setup_cost = setup_unit_cost * sum(y.values())
    routing_cost = sum(
        float(transportation_costs[i][j]) * x[i, j, k, t]
        for i, j in edges
        for k in vehicles
        for t in periods
    )

    model = gp.Model("trusted_adulyasak_recourse")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    model.Params.Method = 1

    production = model.addVars(periods, scenarios, lb=0.0, name="p")
    inventory = model.addVars(nodes, periods, scenarios, lb=0.0, name="I")
    unmet = model.addVars(customers, periods, scenarios, lb=0.0, name="e")
    delivery = model.addVars(customers, vehicles, periods, scenarios, lb=0.0, name="q")

    def demand(omega: int, customer: int, period: int) -> float:
        return float(demands[omega][customer - 1][period - 1])

    for omega in scenarios:
        for t in periods:
            previous = initial_inventory[0] if t == 1 else inventory[0, t - 1, omega]
            model.addConstr(
                previous + production[t, omega]
                == gp.quicksum(
                    delivery[i, k, t, omega] for i in customers for k in vehicles
                )
                + inventory[0, t, omega]
            )
            remaining_demand = sum(
                demand(omega, i, future)
                for i in customers
                for future in range(t, horizon + 1)
            )
            production_bound = min(
                production_capacity,
                vehicle_capacity,
                remaining_demand,
            )
            model.addConstr(production[t, omega] <= production_bound * y[t])
            model.addConstr(inventory[0, t, omega] <= inventory_capacities[0])

        for i in customers:
            for t in periods:
                previous = (
                    initial_inventory[i]
                    if t == 1
                    else inventory[i, t - 1, omega]
                )
                model.addConstr(
                    previous
                    + gp.quicksum(delivery[i, k, t, omega] for k in vehicles)
                    + unmet[i, t, omega]
                    == demand(omega, i, t) + inventory[i, t, omega]
                )
                model.addConstr(
                    inventory[i, t, omega] + demand(omega, i, t)
                    <= inventory_capacities[i]
                )
                future_demand = sum(
                    demand(omega, i, future)
                    for future in range(t, horizon + 1)
                )
                delivery_bound = min(
                    inventory_capacities[i],
                    vehicle_capacity,
                    future_demand,
                )
                for k in vehicles:
                    model.addConstr(
                        delivery[i, k, t, omega]
                        <= delivery_bound * z[i, k, t]
                    )

        for k in vehicles:
            for t in periods:
                model.addConstr(
                    gp.quicksum(
                        delivery[i, k, t, omega] for i in customers
                    )
                    <= vehicle_capacity * z[0, k, t]
                )

    recourse_cost = gp.quicksum(
        probabilities[omega]
        * (
            gp.quicksum(production_unit_cost * production[t, omega] for t in periods)
            + gp.quicksum(
                holding_costs[i] * inventory[i, t, omega]
                for i in nodes
                for t in periods
            )
            + gp.quicksum(
                penalty_costs[i - 1] * unmet[i, t, omega]
                for i in customers
                for t in periods
            )
        )
        for omega in scenarios
    )
    model.setObjective(recourse_cost, GRB.MINIMIZE)
    model.optimize()
    if model.Status != GRB.OPTIMAL:
        raise ValueError(f"trusted recourse solve did not reach optimality: {model.Status}")
    value = setup_cost + routing_cost + float(model.ObjVal)
    model.dispose()
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--result_path", required=True)
    args = parser.parse_args()

    try:
        base_result = _run_dataset_checker(args.instance_path, args.solution_path)
        if base_result.get("feasible") is not True:
            with open(args.result_path, "w", encoding="utf-8") as handle:
                json.dump(base_result, handle, indent=2)
            return
        instance = _load_object(args.instance_path)
        solution = _load_object(args.solution_path)
        reported = float(solution["objective_value"])
        if not math.isfinite(reported):
            raise ValueError("objective_value must be finite")
        certified = _certified_objective(instance, solution)
        tolerance = max(ABS_TOL, REL_TOL * max(abs(reported), abs(certified)))
        if abs(reported - certified) > tolerance:
            _write_result(
                args.result_path,
                feasible=False,
                violations=[
                    "objective_value does not match the trusted fixed-decision recourse solve"
                ],
            )
            return
        _write_result(args.result_path, feasible=True, violations=[])
    except Exception as exc:
        _write_result(
            args.result_path,
            feasible=False,
            violations=[f"trusted checker rejected solution: {type(exc).__name__}: {exc}"],
        )


if __name__ == "__main__":
    main()
