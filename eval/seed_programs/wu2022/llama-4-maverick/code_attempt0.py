import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def read_instance(instance_path):
    with open(instance_path, 'r') as f:
        return json.load(f)

def solve(instance):
    num_items = instance['dimensions']['num_items']
    num_periods = instance['dimensions']['num_periods']
    num_plants = instance['dimensions']['num_plants']
    num_customers = instance['dimensions']['num_customers']
    
    plant_capabilities = instance['plant_capabilities']
    customer_item_demands = instance['customer_item_demands']
    demand = instance['demand']
    setup_cost = instance['setup_cost']
    production_cost = instance['production_cost']
    holding_cost = instance['holding_cost']
    setup_time = instance['setup_time']
    production_time = instance['production_time']
    capacity = instance['capacity']
    plant_opening_cost = instance['plant_opening_cost']
    transportation_cost = instance['transportation_cost']

    model = gp.Model("SupplyChainOptimization")
    model.modelSense = GRB.MINIMIZE

    # Decision Variables
    Z = model.addVars(num_plants, vtype=GRB.BINARY, name="Z")
    U = model.addVars(num_items, num_plants, vtype=GRB.BINARY, name="U")
    Y = model.addVars(num_items, num_plants, num_periods, vtype=GRB.BINARY, name="Y")
    X = model.addVars(num_items, num_plants, num_periods, vtype=GRB.CONTINUOUS, name="X", lb=0)
    I = model.addVars(num_items, num_plants, num_periods, vtype=GRB.CONTINUOUS, name="I", lb=0)

    # Objective Function
    model.setObjective(
        gp.quicksum(plant_opening_cost[p] * Z[p] for p in range(num_plants)) +
        gp.quicksum(setup_cost[i][p][t] * Y[i, p, t] for i in range(num_items) for p in range(num_plants) for t in range(num_periods)) +
        gp.quicksum(production_cost[i][p][t] * X[i, p, t] for i in range(num_items) for p in range(num_plants) for t in range(num_periods)) +
        gp.quicksum(holding_cost[i][p][t] * I[i, p, t] for i in range(num_items) for p in range(num_plants) for t in range(num_periods)) +
        gp.quicksum(transportation_cost[i][c][p][t] * demand[i][c][t] * U[i, p] for i in range(num_items) for c in range(num_customers) for p in range(num_plants) for t in range(num_periods)),
        GRB.MINIMIZE
    )

    # Constraints
    for i in range(num_items):
        model.addConstr(gp.quicksum(U[i, p] for p in range(num_plants)) == 1, f"SingleSourcing_{i}")

    for i in range(num_items):
        for p in range(num_plants):
            model.addConstr(U[i, p] <= Z[p], f"PlantOpen_{i}_{p}")
            model.addConstr(U[i, p] <= 1 if p in plant_capabilities[i] else 0, f"Capability_{i}_{p}")

    for i in range(num_items):
        for p in range(num_plants):
            for t in range(num_periods):
                model.addConstr(X[i, p, t] <= (1 << 30) * Y[i, p, t], f"ProductionSetup_{i}_{p}_{t}")
                model.addConstr(X[i, p, t] <= (1 << 30) * U[i, p], f"ProductionAssignment_{i}_{p}_{t}")

    for p in range(num_plants):
        for t in range(num_periods):
            model.addConstr(gp.quicksum(setup_time[i][p][t] * Y[i, p, t] + production_time[i][p][t] * X[i, p, t] for i in range(num_items)) <= capacity[p][t] + (1 << 30) * (1 - Z[p]), f"Capacity_{p}_{t}")

    for i in range(num_items):
        for p in range(num_plants):
            for t in range(num_periods):
                if t == 0:
                    model.addConstr(I[i, p, t] == X[i, p, t] - gp.quicksum(demand[i][c][t] for c in range(num_customers)) * U[i, p], f"InventoryBalanceInitial_{i}_{p}_{t}")
                else:
                    model.addConstr(I[i, p, t] == I[i, p, t-1] + X[i, p, t] - gp.quicksum(demand[i][c][t] for c in range(num_customers)) * U[i, p], f"InventoryBalance_{i}_{p}_{t}")

    for i in range(num_items):
        for t in range(num_periods):
            model.addConstr(gp.quicksum(X[i, p, t] for p in range(num_plants)) >= gp.quicksum(demand[i][c][t] for c in range(num_customers)), f"DemandSatisfaction_{i}_{t}")

    return model, Z, U, Y

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str)
    args = parser.parse_args()

    instance = read_instance(args.instance_path)
    model, Z, U, Y = solve(instance)

    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None

    model.Params.TimeLimit = args.time_limit
    model.Params.LogToConsole = 0  # Disable Gurobi console output
    model.optimize()

    if model.SolCount > 0:
        objective_value = model.getObjective().getValue()
        Z_sol = {str(p): Z[p].X for p in Z}
        U_sol = {f"{i}_{p}": U[i, p].X for i in range(instance['dimensions']['num_items']) for p in range(instance['dimensions']['num_plants'])}
        Y_sol = {f"{i}_{p}_{t}": Y[i, p, t].X for i in range(instance['dimensions']['num_items']) for p in range(instance['dimensions']['num_plants']) for t in range(instance['dimensions']['num_periods'])}

        solution = {
            "objective_value": objective_value,
            "Z": Z_sol,
            "U": U_sol,
            "Y": Y_sol
        }

        with open(args.solution_path, 'w') as f:
            json.dump(solution, f, indent=4)

        if logger:
            logger.log(objective_value)

if __name__ == "__main__":
    main()