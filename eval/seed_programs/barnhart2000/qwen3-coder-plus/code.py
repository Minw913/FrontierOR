import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def solve_network_routing(instance_path, solution_path, time_limit, log_path):
    # Load the instance
    with open(instance_path, 'r') as f:
        instance = json.load(f)

    # Parse the data
    num_nodes = instance['network']['num_nodes']
    num_arcs = instance['network']['num_arcs']
    nodes = instance['network']['nodes']
    arcs = instance['network']['arcs']
    num_commodities = instance['commodities']['num_commodities']
    commodity_list = instance['commodities']['commodity_list']
    
    # Create a mapping from arc (from, to) to arc_id for easier processing
    arc_to_id = {}
    for arc in arcs:
        arc_to_id[(arc['from_node'], arc['to_node'])] = arc['arc_id']

    # Create model
    model = gp.Model("NetworkRouting")
    model.setParam("Threads", 1)
    
    # Set parameters
    model.setParam('TimeLimit', time_limit)
    model.setParam('OutputFlag', 0)
    
    # Callback for logging solutions
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    def mycallback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj_val = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if logger:
                logger.log(obj_val)
    
    # Prepare variables and constraints
    # For each commodity and each arc, we have x_k_ij binary variables indicating whether commodity k uses arc (i,j)
    # Also binary variables y_k indicating whether commodity k is rejected
    x_vars = {}  # x[k][arc_id]
    y_vars = {}  # y[k]
    
    # Arc index mapping
    arc_ids = [arc['arc_id'] for arc in arcs]
    arcs_map = {arc['arc_id']: arc for arc in arcs}
    
    for k_idx, commodity in enumerate(commodity_list):
        commodity_id = commodity['commodity_id']
        y_vars[k_idx] = model.addVar(vtype=GRB.BINARY, name=f"y_{commodity_id}")
        
        # Map each arc to variable for this commodity
        x_vars[k_idx] = {}
        for arc in arcs:
            arc_id = arc['arc_id']
            x_vars[k_idx][arc_id] = model.addVar(vtype=GRB.BINARY, name=f"x_{commodity_id}_{arc_id}")
    
    model.update()
    
    # Add flow conservation constraints for each commodity
    for k_idx, commodity in enumerate(commodity_list):
        origin = commodity['origin']
        dest = commodity['destination']
        demand = commodity['demand']
        
        # For every node in the network, add flow conservation
        for node in nodes:
            # Incoming arcs to this node for commodity k
            incoming_arcs = [(a['arc_id'], a['from_node']) for a in arcs 
                             if a['to_node'] == node and a['from_node'] != node]
            # Outgoing arcs from this node for commodity k
            outgoing_arcs = [(a['arc_id'], a['to_node']) for a in arcs 
                             if a['from_node'] == node and a['to_node'] != node]
            
            in_flow = gp.quicksum(x_vars[k_idx][arc_id] for arc_id, _ in incoming_arcs)
            out_flow = gp.quicksum(x_vars[k_idx][arc_id] for arc_id, _ in outgoing_arcs)
            
            if node == origin:
                # At origin node: net outflow is 1 if not rejected
                model.addConstr(out_flow - in_flow == 1 - y_vars[k_idx],
                                name=f"flow_conservation_{commodity['commodity_id']}_out_{node}")
            elif node == dest:
                # At destination node: net inflow is 1 if not rejected
                model.addConstr(in_flow - out_flow == 1 - y_vars[k_idx],
                                name=f"flow_conservation_{commodity['commodity_id']}_in_{node}")
            else:
                # At intermediate nodes, no net flow
                model.addConstr(in_flow == out_flow,
                                name=f"flow_conservation_{commodity['commodity_id']}_intermediate_{node}")

    # Add capacity constraints for each arc
    for arc in arcs:
        arc_id = arc['arc_id']
        capacity = arc['capacity']
        # The sum of demands for all commodities using this arc must not exceed its capacity
        used_demand = gp.quicksum(
            commodity_list[k]['demand'] * x_vars[k][arc_id]
            for k in range(len(commodity_list))
        )
        model.addConstr(used_demand <= capacity,
                        name=f"capacity_{arc_id}")

    # Define the objective: minimize total cost of rejections
    obj_expr = gp.quicksum(
        commodity_list[k]['artificial_arc_cost'] * y_vars[k]
        for k in range(len(commodity_list))
    )
    
    model.setObjective(obj_expr, GRB.MINIMIZE)
    
    # Optimize the model
    model.optimize(mycallback)
    
    # Process the solution
    objective_value = model.objVal if model.solCount > 0 else float('inf')
    
    solution_output = {
        "objective_value": objective_value,
        "commodities": []
    }
    
    # Extract solution
    if model.solCount > 0:
        for k_idx, commodity in enumerate(commodity_list):
            commodity_id = commodity['commodity_id']
            rejected = bool(round(y_vars[k_idx].x))
            
            path = []
            if not rejected:
                for arc in arcs:
                    arc_id = arc['arc_id']
                    if round(x_vars[k_idx][arc_id].x) > 0:
                        path.append({
                            "from": arc['from_node'],
                            "to": arc['to_node'],
                            "arc_id": arc['arc_id']
                        })
            
            solution_output["commodities"].append({
                "commodity_id": commodity_id,
                "rejected": rejected,
                "path_arcs": path
            })
    
    # Write the solution
    with open(solution_path, 'w') as f:
        json.dump(solution_output, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True, help="Path to instance file")
    parser.add_argument("--solution_path", required=True, help="Path to write solution")
    parser.add_argument("--time_limit", type=int, required=True, help="Time limit in seconds")
    parser.add_argument("--log_path", required=False, help="Path for log output")

    args = parser.parse_args()
    
    solve_network_routing(args.instance_path, args.solution_path, args.time_limit, args.log_path)


if __name__ == "__main__":
    main()