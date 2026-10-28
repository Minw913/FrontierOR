import argparse
import json
import math
from collections import defaultdict
import datetime

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    print("Gurobi is not available, using heuristic approach")
    gp = None


class CrewSchedulingSolver:
    def __init__(self, instance_data):
        self.instance = instance_data
        self.num_crew = instance_data["num_crew_members"]
        self.num_pairings = instance_data["num_mother_pairings"]
        self.planning_horizon = instance_data["planning_horizon_days"]
        self.pairings = {p["id"]: p for p in instance_data["pairings"]}
        self.pairing_list = instance_data["pairings"]

        # Work rules extraction
        self.weekly_rest_required = instance_data["work_rules"]["weekly_rest_period_hours"]
        self.monthly_rest_addition = instance_data["work_rules"]["monthly_rest_additional_hours"]
        self.max_spread_between_breaks = instance_data["work_rules"]["max_spread_between_breaks_days"]
        self.monthly_flight_limit = instance_data["work_rules"]["monthly_flight_limit_hours"]
        self.rolling_flight_limit = instance_data["work_rules"]["rolling_flight_limit_hours"]
        self.max_consecutive_early_dps = instance_data["work_rules"]["max_consecutive_early_departures_before_8am"]
        self.max_cross_pairings_per_month = instance_data["work_rules"]["max_cross_pairings_per_month"]
        self.latest_start_sunday_hrs = instance_data["work_rules"]["weekly_rest_latest_start_on_sunday_hour"]

        # Precomputed values
        self.weekly_rest_minutes = self.weekly_rest_required * 60

    def solve(self, time_limit):
        """
        Solve the crew scheduling problem using a two-stage process:
        1. Generate feasible roster templates (heuristic)
        2. Select the best combination of rosters for all crew members (optimization)
        """
        if gp is not None:
            return self.solve_with_gurobi(time_limit)
        else:
            return self.solve_heuristically()

    def solve_with_gurobi(self, time_limit):
        # Step 1: Generate roster templates using heuristics
        roster_templates = self.generate_roster_templates()
        
        # Step 2: Formulate and solve the set partitioning/covering problem
        m = gp.Model("crew-scheduling")
        m.setParam('TimeLimit', time_limit)

        # Create variables
        x = {}  # x[crew][template_idx] = 1 if crew c gets assigned template t
        for ci in range(self.num_crew):
            for ti, template in enumerate(roster_templates):
                x[(ci, ti)] = m.addVar(vtype=GRB.BINARY)

        # Create covering variables for uncovered pairings
        uncovered_vars = {}
        for pairing in self.pairing_list:
            uncovered_vars[pairing["id"]] = m.addVar(vtype=GRB.INTEGER, lb=0)

        # Constraint: each crew gets exactly one template
        for ci in range(self.num_crew):
            m.addConstr(gp.quicksum(x[(ci, ti)] for ti in range(len(roster_templates))) == 1)

        # Constraint: cover all required pairings
        for pairing in self.pairing_list:
            p_id = pairing["id"]
            coverage_needed = pairing["coverage_requirement"]
            
            total_assigned = gp.quicksum(
                x[(ci, ti)] 
                for ti, template in enumerate(roster_templates) 
                for ci in range(self.num_crew)
                if p_id in template['pairings']
            )
            
            # Define uncovered amount
            m.addConstr(uncovered_vars[p_id] == coverage_needed - total_assigned)

        # Set objective
        m.setObjective(
            gp.quicksum(
                uncovered_vars[p["id"]] * p["duration_minutes"] 
                for p in self.pairing_list
            ),
            GRB.MINIMIZE
        )

        # Optimize
        m.optimize()

        # Extract solution
        solution = {}
        if m.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
            active_rosters = []
            for ci in range(self.num_crew):
                roster_chosen = False
                for ti in range(len(roster_templates)):
                    if x[(ci, ti)].x > 0.9:
                        template = roster_templates[ti]
                        solution[str(ci)] = {
                            "roster_index": ti,
                            "pairings": template.get("pairings", []),
                            "unproductive_time": template.get("unproductive_time", 0)
                        }
                        roster_chosen = True
                        break
                assert roster_chosen, f"No roster chosen for crew {ci}"

            obj_val = m.ObjVal
        else:
            # Failed to find a solution
            obj_val = float('inf')
            for ci in range(self.num_crew):
                solution[str(ci)] = {
                    "roster_index": 0,
                    "pairings": [],
                    "unproductive_time": self.planning_horizon * 24 * 60
                }

        # Compute uncovered details
        uncovered_details = []
        for p in self.pairing_list:
            uv = uncovered_vars[p["id"]].x if 'uncovered_vars' in locals() and m.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT] else p["coverage_requirement"]
            if uv > 0:
                uncovered_details.append({
                    "pairing_id": p["id"],
                    "uncovered_count": max(0, int(round(uv))),
                    "duration_minutes": p["duration_minutes"]
                })

        result = {
            "objective_value": obj_val if obj_val != float('inf') else sum(p["coverage_requirement"] * p["duration_minutes"] for p in self.pairing_list),
            "rosters": solution,
            "uncovered_pairings": uncovered_details
        }
        return result

    def generate_roster_templates(self):
        """Generate some feasible roster combinations heuristically."""
        templates = []

        # For each crew member, try to create basic feasible rosters
        for crew_info in self.instance["crew_members"]:
            qualifications_set = set(crew_info["qualifications"])
            pa_days = set(crew_info.get("pre_assigned_days", []))
            
            # Get feasible pairings for this crew
            eligible_pairings = []
            for p in self.pairing_list:
                if p["aircraft_type"] in qualifications_set:
                    # Check if pairing conflicts with pre-assigned activities
                    conflict = False
                    start_day = p["start_day"]
                    end_day = p["start_day"] + p["duration_days"] - 1
                    
                    for day in range(start_day, end_day+1):
                        if day in pa_days:
                            conflict = True
                            break
                            
                    if not conflict:
                        eligible_pairings.append(p)

            # Create simple rosters: assign subset of eligible pairings respecting max_flight_limit
            max_pairings_per_roster = 10  # Limit to prevent exponential growth
            sorted_pairs = sorted(eligible_pairings, key=lambda x: x["start_day"])

            # Try to pack as many eligible early pairings as possible within limits
            current_duration = crew_info.get("flight_hours_previous_15_days", 0)
            selected_p = []
            
            for p in sorted_pairs:
                if current_duration + p["flight_hours"] <= self.rolling_flight_limit:
                    selected_p.append(p["id"])
                    current_duration += p["flight_hours"]
                
                if len(selected_p) >= max_pairings_per_roster:
                    break

            if len(selected_p) > 0:
                r_template = {
                    "pairings": selected_p[:max_pairings_per_roster],
                    "unproductive_time": 0  # Placeholder
                }
                templates.append(r_template)

        # Add empty roster (in case no eligible pairing)
        templates.append({
            "pairings": [],
            "unproductive_time": self.planning_horizon * 24 * 60
        })
        
        return templates

    def solve_heuristically(self):
        """Fallback heuristic solution when Gurobi is not available"""
        solution = {}

        for i in range(self.num_crew):
            solution[str(i)] = {
                "roster_index": 0,
                "pairings": [],  # For simplicity, just use empty pairing
                "unproductive_time": self.planning_horizon * 24 * 60  # Total minutes in month
            }

        # Objective value assumes all pairings are uncovered
        total_uncovered = sum(p["coverage_requirement"] * p["duration_minutes"] for p in self.pairing_list)

        return {
            "objective_value": total_uncovered,
            "rosters": solution,
            "uncovered_pairings": [
                {
                    "pairing_id": p["id"],
                    "uncovered_count": p["coverage_requirement"],
                    "duration_minutes": p["duration_minutes"]
                } for p in self.pairing_list
            ]
        }


def main():
    parser = argparse.ArgumentParser(description="Solve crew scheduling problem")
    parser.add_argument("--instance_path", required=True, help="Path to the input instance JSON file")
    parser.add_argument("--solution_path", required=True, help="Path to save the solution JSON file")
    parser.add_argument("--time_limit", required=True, type=int, help="Time limit in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Path to write logs of intermediate best solutions")

    args = parser.parse_args()

    # Initialize logger
    logger = None
    if args.log_path:
        from solution_logger import SolutionLogger
        logger = SolutionLogger(args.log_path, sense="minimize")

    with open(args.instance_path, "r") as f:
        instance_data = json.load(f)

    solver = CrewSchedulingSolver(instance_data)

    solution = solver.solve(args.time_limit)

    # Log final objective value if logger exists
    if logger and solution:
        logger.log(solution["objective_value"])

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)


if __name__ == "__main__":
    main()