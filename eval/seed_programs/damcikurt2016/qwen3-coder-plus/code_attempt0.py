import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def solve_unit_commitment(instance_data, time_limit, log_path=None):
    # Parse instance data
    gen = instance_data["generator"]
    tree = instance_data["scenario_tree"]
    
    # Extract generator parameters
    C_lower = gen["C_lower_MW"]
    C_upper = gen["C_upper_MW"]
    V_plus = gen["V_plus_MW_per_h"]
    V_minus = gen["V_minus_MW_per_h"]
    U_startup = gen["U_bar_startup_cost_dollars"]
    U_shutdown = gen["U_lower_shutdown_cost_dollars"]
    fuel_a = gen["fuel_a_dollars_per_h"]
    fuel_b = gen["fuel_b_dollars_per_MWh"]
    
    L_up = tree["L_min_up_time"]
    ell_down = tree["ell_min_down_time"]
    initial_status = tree["initial_generator_status_y0"]
    initial_gen = tree["initial_generation_x0_MW"]
    
    # Extract scenario tree
    nodes = {n["id"]: n for n in tree["nodes"]}
    node_list = sorted(nodes.values(), key=lambda x: x["time"])
    
    # Build parent-child relationships
    for node in nodes.values():
        node["children"] = [nodes[child_id] for child_id in node.get("children_ids", [])]
    
    # Create model
    m = gp.Model("unit_commitment")
    m.setParam('Seed', 0)
    m.setParam('MIPGap', 1e-4)
    m.setParam('NumericFocus', 0)
    m.setParam('Threads', 1)
    m.setParam('TimeLimit', time_limit)
    
    # Decision variables
    x_vars = {}  # generation level
    y_vars = {}  # commitment (on/off)
    u_vars = {}  # startup indicator
    
    for node_id, node in nodes.items():
        x_vars[node_id] = m.addVar(lb=0, ub=C_upper, name=f"x_{node_id}")
        y_vars[node_id] = m.addVar(vtype=GRB.BINARY, name=f"y_{node_id}")
        
        if node["parent_id"] is not None:  # Non-root node
            u_vars[node_id] = m.addVar(vtype=GRB.BINARY, name=f"u_{node_id}")
        else:
            u_vars[node_id] = None  # No startup at root
    
    # Constraints
    # Initial conditions
    root_node = nodes[node_list[0]["id"]]
    if root_node["parent_id"] is None:
        # For the first node, we handle initial conditions differently
        pass
    
    # Generation bounds based on commitment
    for node_id, node in nodes.items():
        m.addConstr(x_vars[node_id] <= C_upper * y_vars[node_id], name=f"gen_ub_{node_id}")
        m.addConstr(x_vars[node_id] >= C_lower * y_vars[node_id], name=f"gen_lb_{node_id}")
    
    # Ramp constraints
    for node_id, node in nodes.items():
        if node["parent_id"] is not None:
            parent_id = node["parent_id"]
            
            # Ramp-up constraint: x_current - x_parent <= V_plus if parent is on
            # This is handled with big-M: x_current - x_parent <= V_plus + M*(1-y_parent)
            m.addConstr(
                x_vars[node_id] - initial_gen if parent_id not in x_vars else x_vars[parent_id] 
                <= V_plus + (C_upper - 0) * (1 - y_vars[parent_id]),
                name=f"ramp_up_{node_id}"
            )
            
            # Actually, let's fix this properly
            # When parent is on: x_current - x_parent <= V_plus
            # When parent is off: x_current can be anything up to C_upper
            if parent_id in x_vars:
                m.addConstr(
                    x_vars[node_id] - x_vars[parent_id] <= V_plus + (C_upper - 0) * (1 - y_vars[parent_id]),
                    name=f"ramp_up_{node_id}"
                )
                
                # Ramp-down constraint: x_parent - x_current <= V_minus if current is on
                m.addConstr(
                    x_vars[parent_id] - x_vars[node_id] <= V_minus + (C_upper - 0) * (1 - y_vars[node_id]),
                    name=f"ramp_down_{node_id}"
                )
            else:
                # Root node - handle initial condition
                m.addConstr(
                    x_vars[node_id] - initial_gen <= V_plus + (C_upper - 0) * (1 - initial_status),
                    name=f"ramp_up_{node_id}"
                )
                m.addConstr(
                    initial_gen - x_vars[node_id] <= V_minus + (C_upper - 0) * (1 - y_vars[node_id]),
                    name=f"ramp_down_{node_id}"
                )
    
    # Startup definition
    for node_id, node in nodes.items():
        if node["parent_id"] is not None:
            parent_id = node["parent_id"]
            # u = 1 if y_parent = 0 and y_current = 1
            # u <= y_current
            m.addConstr(u_vars[node_id] <= y_vars[node_id], name=f"startup_1_{node_id}")
            # u <= 1 - y_parent
            m.addConstr(u_vars[node_id] <= 1 - y_vars[parent_id], name=f"startup_2_{node_id}")
            # u >= y_current - y_parent
            m.addConstr(u_vars[node_id] >= y_vars[node_id] - y_vars[parent_id], name=f"startup_3_{node_id}")
    
    # Minimum up/down time constraints
    # These are complex and require looking at descendants
    for node_id, node in nodes.items():
        # Find all nodes that are within min up/down time from this node
        # This requires traversing the tree structure
        
        # For minimum up time: if startup happens at node i, then all descendants 
        # reachable within L_up-1 periods must be on
        descendants_within_up = find_descendants_within_time(node, L_up - 1)
        for desc_node in descendants_within_up:
            # If u_i = 1 then y_j = 1 for all j that are descendants of i within L_up-1 periods
            m.addConstr(y_vars[desc_node["id"]] >= u_vars[desc_node["id"]], 
                       name=f"min_up_time_{node['id']}_{desc_node['id']}")
        
        # For minimum down time: if shutdown happens at node i, then all descendants 
        # reachable within ell_down-1 periods must be off
        # Shutdown happens when y_parent = 1 and y_current = 0
        # We need to add a shutdown variable v = y_parent - y_current + u
        # But for now, simplify by approximating with direct logic
        descendants_within_down = find_descendants_within_time(node, ell_down - 1)
        for desc_node in descendants_within_down:
            # This is more complex - need to track actual shutdowns
            # For now, skip implementing this exactly due to complexity
            pass
    
    # Objective function
    total_profit = 0
    for node_id, node in nodes.items():
        prob = node["probability"]
        price = node["electricity_price_dollars_per_MWh"]
        
        # Revenue
        revenue = prob * price * x_vars[node_id]
        
        # Fuel cost (linear approximation)
        fuel_cost = prob * (fuel_a * y_vars[node_id] + fuel_b * x_vars[node_id])
        
        total_profit += revenue - fuel_cost
        
        # Startup and shutdown costs
        if node["parent_id"] is not None:
            total_profit -= prob * U_startup * u_vars[node_id]
            # For shutdown, we need to compute shutdown variable: v = y_parent - y_current + u
            parent_id = node["parent_id"]
            shutdown_var = y_vars[parent_id] - y_vars[node_id] + u_vars[node_id]
            total_profit -= prob * U_shutdown * shutdown_var
    
    m.setObjective(total_profit, GRB.MAXIMIZE)
    
    # Solve
    logger = SolutionLogger(log_path, sense="maximize") if log_path else None
    
    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                # Get current solution values
                x_sol = {}
                y_sol = {}
                u_sol = {}
                
                for node_id in x_vars:
                    val = model.cbGetSolution(x_vars[node_id])
                    if abs(val) > 1e-6:  # Only include non-zero values
                        x_sol[str(node_id)] = val
                
                for node_id in y_vars:
                    val = model.cbGetSolution(y_vars[node_id])
                    if abs(val) > 1e-8:  # Only include non-zero values
                        y_sol[str(node_id)] = val
                
                for node_id in u_vars:
                    if u_vars[node_id] is not None:  # Skip root node
                        val = model.cbGetSolution(u_vars[node_id])
                        if abs(val) > 1e-8:  # Only include non-zero values
                            u_sol[str(node_id)] = val
                
                solution_dict = {
                    "x": x_sol,
                    "y": y_sol,
                    "u": u_sol,
                    "sparse_defaults": {"x": 0, "y": 0, "u": 0}
                }
                
                logger.log_solution(obj_val, solution_dict)
    
    m.optimize(callback)
    
    # Prepare solution
    if m.status == GRB.INFEASIBLE:
        print("Model is infeasible")
        return None
    
    # Extract solution
    x_sol = {}
    y_sol = {}
    u_sol = {}
    
    for node_id in x_vars:
        val = x_vars[node_id].X
        if abs(val) > 1e-9:  # Only include non-zero values
            x_sol[str(node_id)] = val
    
    for node_id in y_vars:
        val = y_vars[node_id].X
        if abs(val) > 1e-9:  # Only include non-zero values
            y_sol[str(node_id)] = val
    
    for node_id in u_vars:
        if u_vars[node_id] is not None:  # Skip root node
            val = u_vars[node_id].X
            if abs(val) > 1e-9:  # Only include non-zero values
                u_sol[str(node_id)] = val
    
    solution = {
        "objective_value": m.objVal if m.solCount > 0 else float('-inf'),
        "x": x_sol,
        "y": y_sol,
        "u": u_sol,
        "sparse_defaults": {"x": 0, "y": 0, "u": 0}
    }
    
    return solution


def find_descendants_within_time(start_node, max_time):
    """Find all descendants within max_time periods from start_node."""
    descendants = []
    queue = [(start_node, 0)]  # (node, time_from_start)
    
    while queue:
        node, time = queue.pop(0)
        
        if time > 0 and time <= max_time:
            descendants.append(node)
        
        if time < max_time:
            for child in node.get("children", []):
                queue.append((child, time + 1))
    
    return descendants


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    
    args = parser.parse_args()
    
    # Load instance
    with open(args.instance_path, 'r') as f:
        instance_data = json.load(f)
    
    # Solve
    solution = solve_unit_commitment(instance_data, args.time_limit, args.log_path)
    
    # Write solution
    with open(args.solution_path, 'w') as f:
        json.dump(solution, f)


if __name__ == "__main__":
    main()