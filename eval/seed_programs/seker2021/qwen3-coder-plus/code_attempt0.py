import argparse
import json
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger


def solve_graph_coloring_instance(instance_path, time_limit, log_path=None):
    # Read the instance
    with open(instance_path, 'r') as f:
        instance = json.load(f)
    
    # Extract instance data
    graph = instance['graph']
    partition = instance['partition']
    
    n_vertices = graph['num_vertices']
    edges = graph['edges']
    n_clusters = partition['num_clusters']
    clusters = partition['clusters']
    
    # Create mapping from vertex to cluster
    vertex_to_cluster = {}
    for i, cluster in enumerate(clusters):
        for v in cluster:
            vertex_to_cluster[v] = i
    
    # Initialize solution logger
    logger = SolutionLogger(log_path, sense="minimize") if log_path else None
    
    # Create Gurobi model
    model = gp.Model("Graph_Selection_Coloring")
    model.setParam('TimeLimit', time_limit)
    
    # Variables:
    # x[v]: binary variable indicating if vertex v is selected
    # y[c, k]: binary variable indicating if cluster c selects a vertex colored with color k
    # z[k]: binary variable indicating if color k is used
    x = {} 
    y = {}
    z = {}
    
    # Create variables
    for v in range(n_vertices):
        x[v] = model.addVar(vtype=GRB.BINARY, name=f"x_{v}")
    
    max_colors = n_vertices  # upper bound on needed colors
    for c in range(n_clusters):
        for k in range(max_colors):
            y[c, k] = model.addVar(vtype=GRB.BINARY, name=f"y_{c}_{k}")
        
    for k in range(max_colors):
        z[k] = model.addVar(vtype=GRB.BINARY, name=f"z_{k}")
    
    # Constraint 1: Exactly one vertex per cluster must be selected
    for c in range(n_clusters):
        model.addConstr(gp.quicksum(x[v] for v in clusters[c]) == 1, name=f"one_per_cluster_{c}")
    
    # Constraint 2: Connect x with y - if cluster c selects vertex v it gets some color
    for c in range(n_clusters):
        for v in clusters[c]:
            # If vertex v is selected from cluster c, then some color k is used for cluster c
            model.addConstr(x[v] <= gp.quicksum(y[c, k] for k in range(max_colors)), name=f"x_implies_y_{c}_{v}")
            
            # A vertex v in cluster c can only get colors assigned to cluster c
            for k in range(max_colors):
                model.addConstr(x[v] <= y[c, k] + (1 - y[c, k]), name=f"implies_color_for_vertex_{v}_{k}")
                # When x[v] is 1, we need to make sure that y[c,k] corresponds to whether color k is used for this vertex
                # Need to add additional logic to track colors properly
    
    # We'll add a simpler version that ensures proper linking between selections and colors
    for c in range(n_clusters):
        # Ensure that exactly one color is assigned to the selected vertex in cluster c
        model.addConstr(gp.quicksum(y[c, k] for k in range(max_colors)) == 1, name=f"one_color_per_cluster_{c}")
        
    # Now link y variables directly to x variables for the specific cluster
    for c in range(n_clusters):
        for k in range(max_colors):
            # y[c, k] means cluster c assigns color k to the selected vertex
            # This can be expressed as:
            # y[c, k] == x[v] for selected vertex v in clusters[c] with the same color k
            model.addConstr(
                y[c, k] >= gp.quicksum(x[v] for v in clusters[c]) - 1 + x[clusters[c][0]] * 0,  
                name=f"y_link_{c}_{k}_temp"
            ) 
            # Actually better implemented with indicator constraints
            
    # Reset variables - let's restructure completely
    del(model)
    model = gp.Model("Graph_Selection_Coloring_New")
    model.setParam('TimeLimit', time_limit)
    
    # x[v]: binary variable indicating if vertex v is selected
    x = {}
    for v in range(n_vertices):
        x[v] = model.addVar(vtype=GRB.BINARY, name=f"x_{v}")

    # color[c]: integer specifying the color label for the vertex selected from cluster c  
    color = {}
    for c in range(n_clusters):
        color[c] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=max_colors-1, name=f"color_{c}")
    
    # z[k]: binary variable indicating if color k is used by at least one cluster
    z = {}
    for k in range(max_colors):
        z[k] = model.addVar(vtype=GRB.BINARY, name=f"z_{k}")
    
    # Exactly one vertex per cluster
    for c in range(n_clusters):
        model.addConstr(gp.quicksum(x[v] for v in clusters[c]) == 1, name=f"exactly_one_from_cluster_{c}")
    
    # If vertex v in cluster c is selected (x[v]=1), then the color assigned to cluster c should apply
    for v in range(n_vertices):
        c = vertex_to_cluster[v]
        for k in range(max_colors):
            # If vertex v is selected in cluster c (x[v]=1), then c's color must be k when z[k]=1 appropriately
            model.addGenConstrIndicator(
                x[v], 
                True, 
                color[c],
                GRB.EQUAL,
                k,  
                name=f"selected_implies_color_{v}_{k}"
            )
            
    # Actually, it's better to express it differently
    del(model)
    model = gp.Model("Graph_Selection_Coloring_Final")
    model.setParam('TimeLimit', time_limit)

    # x[v, c]: binary variables where =1 if the single vertex from cluster c is vertex v
    x = {}
    for c_idx, cluster in enumerate(clusters):
        for v in cluster:
            x[v, c_idx] = model.addVar(vtype=GRB.BINARY, name=f"x_v{v}_c{c_idx}")
    
    # color[c]: integer indicating the color of vertices from each cluster    
    color = {}
    for c in range(n_clusters):
        color[c] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=n_vertices, name=f"color_{c}")

    # For the objective: z[k] indicates if color k is used  
    max_possible_colors = max(n_vertices, n_clusters)  # reasonable upper bound
    z = {}
    for k in range(max_possible_colors):
        z[k] = model.addVar(vtype=GRB.BINARY, name=f"used_color_{k}")

    # 1. From each cluster, select exactly one vertex
    for c_idx, cluster in enumerate(cluster_indices=range(len(clusters))):
        model.addConstr(gp.quicksum(x[v, c_idx] for v in cluster) == 1, name=f"clust_{c_idx}_select_one")

    # 2. Link cluster color variables to individual vertex selections
    # If vertex v from cluster c is selected, then color[c] corresponds to the color for v
    for c_idx, cluster in enumerate(clusters):
        for v in cluster:
            # For symmetry breaking, we can order colors or force to smallest available
            pass  # Handled later in conflict constraints

    # 3. Conflict constraints: adjacent vertices must have different colors
    for u, v_edge in edges:
        cluster_u = vertex_to_cluster[u]
        cluster_v = vertex_to_cluster[v_edge]
        
        if cluster_u != cluster_v:  # Only need to handle conflicts between different clusters
            for ku in range(max_possible_colors):
                for kv in range(max_possible_colors):
                    if ku == kv:  # Same color constraint
                        model.addConstr(
                            x[u, cluster_u] + x[v_edge, cluster_v] <= 1,  # Conflict - cannot both be selected with same color
                            name=f"conflict_uv_{u}v{v_edge}_{ku}-{kv}"
                        )
                    
    # More efficient conflict constraints
    del(model)
    model = gp.Model("Graph_Selection_Coloring_Efficient")
    model.setParam('TimeLimit', time_limit)
    
    # Variables:
    # x[v, c]: 1 if vertex v from cluster c is selected  
    x = {}
    for c_idx, cluster in enumerate(clusters):
        for v in cluster:
            x[v, c_idx] = model.addVar(vtype=GRB.BINARY, name=f"x_{v}_{c_idx}")
    
    # color[c]: integer representing what color the vertex from cluster c gets
    color = {}
    for c in range(n_clusters):
        color[c] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=n_vertices, name=f"color_{c}")
    
    # z[k]: binary, whether color k is used anywhere
    z = {}
    for k in range(n_vertices):
        z[k] = model.addVar(vtype=GRB.BINARY, name=f"z_{k}")
        
    # One vertex per cluster constraint
    for c_idx, cluster in enumerate(clusters):
        model.addConstr(gp.quicksum(x[v, c_idx] for v in cluster) == 1, name=f"one_per_cluster_{c_idx}")
    
    # If x[v, c] = 1, set color[c] appropriately (this requires big-M or indicators)
    # For simplicity, we'll use big-M formulation
    for c_idx in range(n_clusters):
        big_m = n_vertices
        for v in clusters[c_idx]:
            for k in range(n_vertices):
                # If x[v, c_idx] = 1, then color[c_idx] = k doesn't affect if we simply connect to global color vars properly
                # Better approach - redefine color as integer but constrain via implications
                M = n_vertices
                model.addConstr(color[c_idx] >= k - M * (1 - x[v, c_idx]) - M/2 * (1 if v not in clusters[c_idx] else 0),
                                name=f"color_lb_{v}_{c_idx}_{k}")
                model.addConstr(color[c_idx] <= k + M * (1 - x[v, c_idx]),
                                name=f"color_ub_{v}_{c_idx}_{k}")

    # Edge conflicts: if edge connects vertices from different clusters and both are selected, they must have diff colors
    for u, v_edge in edges:
        cluster_u = vertex_to_cluster[u]
        cluster_v = vertex_to_cluster[v_edge]
        
        if cluster_u != cluster_v:
            for k in range(n_vertices):
                # If vertex u from cluster cluster_u is selected AND has color k,
                # AND vertex v_edge from cluster cluster_v is selected AND has color k,
                # then this is forbidden
                model.addConstr(
                    x[u, cluster_u] + x[v_edge, cluster_v] + 
                    (1 if color[cluster_u] <= k and color[cluster_u] >= k else 0) +
                    (1 if color[cluster_v] <= k and color[cluster_v] >= k else 0) 
                    <= 3,
                    name=f"conflict_{u}_{v_edge}_color{k}"
                )
                
    # Simplify color linking using indicator constraints properly
    del(model)
    model = gp.Model("Final_Graph_Selection_Coloring")
    model.setParam('TimeLimit', time_limit)
    
    # x[v, c]: Binary variable - whether vertex v in cluster c is selected
    x = {}
    for c_idx, cluster in enumerate(clusters):
        for v in cluster:
            x[v, c_idx] = model.addVar(vtype=GRB.BINARY, name=f"x_v{v}_c{c_idx}")
    
    # color[c]: Integer color index for cluster c
    color = {}
    M = n_vertices
    for c in range(n_clusters):
        color[c] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=n_vertices, name=f"color_c{c}")
        
    # z[k]: whether color k is used by any vertex
    z = {}
    for k in range(n_vertices):
        z[k] = model.addVar(vtype=GRB.BINARY, name=f"used_k{k}")
            
    # Constraint: one vertex per cluster
    for c_idx, cluster in enumerate(clusters):
        model.addConstr(gp.quicksum(x[v, c_idx] for v in cluster) == 1, name=f"one_per_clust_{c_idx}")
        
    # Link colors with selected vertices using indicators
    for c_idx, cluster in enumerate(clusters):
        for v in cluster:
            for k in range(n_vertices):
                # Indicator: if x[v,c_idx]=1 then color[c_idx] should be linked properly
                b = model.addVar(vtype=GRB.BINARY, name=f"temp_color_var_{v}_{c_idx}_{k}")
                
                # Expressing (x[v, c_idx] = 1) -> (color[c_idx] = k) is complex
                # Let's reformulate more efficiently:
                model.addConstr(color[c_idx] >= k - M*(1 - x[v, c_idx]), name=f"colorlb_{v}_{c_idx}_{k}")
                model.addConstr(color[c_idx] <= k + M*(1 - x[v, c_idx]), name=f"colorub_{v}_{c_idx}_{k}")
    
    # Color usage linking: z[i] is active if any cluster uses color i
    for k in range(n_vertices):
        for c_idx in range(n_clusters):
            model.addConstr(z[k] >= 
                gp.quicksum(
                     x[v, c_idx] for v in clusters[c_idx] 
                     if abs(color[c_idx].X - k) < 0.5  # Approximation - doesn't work here directly
                ),
            name=f"z_{k}_linked_to_colors")
    
    # Use a different approach for efficiency:
    del(model)
    model = gp.Model("Proper_Selection_Coloring")
    model.setParam('TimeLimit', time_limit)
    
    # Decision variables
    # x[v, c] = 1 if vertex v from cluster c is selected
    x = {}
    for c_idx, cluster in enumerate(clusters):
        for v in cluster:
            x[v, c_idx] = model.addVar(vtype=GRB.BINARY, name=f"x_v{v}_c{c_idx}")
    
    # vertex_color[c] stores which color is assigned to selected vertex from cluster c
    vertex_color = {}
    for c_idx in range(n_clusters):
        vertex_color[c_idx] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=n_clusters, name=f"vc_c{c_idx}")
    
    # z[k] = 1 if color k is used by any selected vertex
    z = {}
    for k in range(n_clusters):  # We need at most this many colors due to possible complete subgraph between clusters
        z[k] = model.addVar(vtype=GRB.BINARY, name=f"z_color_used{k}")
    
    # Exactly one vertex per cluster
    for c_idx, cluster in enumerate(clusters):
        model.addConstr(gp.quicksum(x[v, c_idx] for v in cluster) == 1, name=f"one_per_clstr_{c_idx}")
    
    # If vertex v from cluster c is selected, connect its color
    for c_idx, cluster in enumerate(clusters):
        model.addConstr(vertex_color[c_idx] == gp.quicksum(x[v, c_idx] * get_vertex_label(v, c_idx) for v in cluster))
        
    # Adjacent vertices from different clusters must have different colors  
    for u, v_edge in edges:
        cluster_u = vertex_to_cluster[u]
        cluster_v = vertex_to_cluster[v_edge]
        
        if cluster_u != cluster_v:
            # If both are selected, must have different colors
            for k in range(n_clusters):
                model.addConstr(x[u, cluster_u] + x[v_edge, cluster_v] - 
                               (1 if vertex_color[cluster_u] == k else 0) - 
                               (1 if vertex_color[cluster_v] == k else 0) <= 1,
                               name=f"adj_not_same_color_{u}_{v_edge}_{k}")

    # Unfortunately this is getting complex, so let's take more direct route
    
    # Instead, we define color mappings differently:
    def create_color_constraints():
        model = gp.Model("SelectionAndColoring")
        model.setParam('TimeLimit', time_limit)
        
        # Selection variables
        x = {}
        for c_idx, cluster in enumerate(clusters):
            for v in cluster:
                x[v, c_idx] = model.addVar(vtype=GRB.BINARY, name=f"x_v{v}_c{c_idx}")
                
        # Auxiliary constraint: one per cluster
        for c_idx, cluster in enumerate(clusters):
            model.addConstr(gp.quicksum(x[v, c_idx] for v in cluster) == 1, name=f"one_per_cluster_{c_idx}")
        
        # For coloring, we'll define color values for each cluster, using constraints to ensure neighbors get different color
        max_colors = max(n_clusters, 30)  # heuristic maximum 
        cluster_colors = []
        for c_idx in range(n_clusters):
            var = model.addVar(vtype=GRB.INTEGER, lb=0, ub=max_colors-1, name=f"cluster_color_{c_idx}")
            cluster_colors.append(var)
        
        # z[k] = 1 if color k is actually used
        z = []
        for k in range(max_colors):
            zvar = model.addVar(vtype=GRB.BINARY, name=f"uses_color_{k}")
            z.append(zvar)
        
        # Force z_k correctly using binary encoding and linking to actual color values
        for k in range(max_colors):
            # If any cluster uses color k, then z[k] must be 1
            for c_idx in range(n_clusters):
                # Introduce binary helper variables
                color_equals_k = model.addVar(vtype=GRB.BINARY, name=f"clcolor_{c_idx}_eq_{k}")
                
                # This is tricky in linear terms; we use indicator constraints
                model.addGenConstrIndicator(color_equals_k, 1, cluster_colors[c_idx], GRB.EQUAL, k)
                model.addGenConstrIndicator(color_equals_k, 0, cluster_colors[c_idx], GRB.NOTEQUAL, k)
                
                model.addConstr(z[k] >= color_equals_k, name=f"z_implied_by_equal_{c_idx}_{k}")
        
        # Adjacency constraints: connected vertices from different clusters have different colors
        for u, v_edge in edges:
            cu = vertex_to_cluster[u] 
            cv = vertex_to_cluster[v_edge]
            
            if cu != cv:  # Different clusters
                # We ensure that if both u and v_edge are selected, they get different cluster colors
                for k in range(max_colors):                
                    # If x[u, cu]=1 and x[v, cv]=1, then color[cu]!=k or color[cv]!=k for all k
                    # Which means (x[u,cu] AND x[v,cv]) => (color[cu] != k OR color[cv] != k)
                    # Contrapositive: (color[cu] = k AND color[cv] = k) => (x[u, cu] = 0 OR x[v, cv] = 0) 
                    # Which gives us: (color[cu] = k AND color[cv] = k) => (x[u, cu] + x[v, cv] <= 1)
                    # Using indicator: (ind_c_cu_eq_k AND ind_c_cv_eq_k) => (x_cu + x_cv <= 1)
                    
                    # We already defined helpers above
                    pass
        
        # Rebuild constraints without nested indicators
        model.update()
        
        # Add adjacency constraints manually
        for u, v_edge in edges:
            cu = vertex_to_cluster[u]
            cv = vertex_to_cluster[v_edge]
            
            if cu != cv:
                # For all colors k, we can't have both vertices in these clusters selected and assigned same color
                # Use a single constraint: for all assignment x[u,cu]=1, x[v,cv]=1, color[cu]=k, color[cv]=k:
                # sum over k: if both x vars are 1 and both colors are k -> contradiction
                # So: for each possible k, (if cu gets color k & cv gets color k) -> (at most 1 is selected)
                for k in range(max_colors):
                    ind_cu_k = model.addVar(vtype=GRB.BINARY, name=f"cu{cu}_has_color_{k}")
                    ind_cv_k = model.addVar(vtype=GRB.BINARY, name=f"cv{cv}_has_color_{k}")
                    
                    # If color[cu] = k then ind_cu_k = 1, etc.
                    model.addGenConstrIndicator(cluster_colors[cu], 0, ind_cu_k, True)  # Not directly doable
                    # Instead define through relationship
                    for val in range(max_colors):
                        # Use a set of helper binaries - more efficient would be sum of binary encoding
                        pass
                    
                    # For our purposes, we use big-M approach:
                    # |color[cu] - color[cv]| >= 1 when both u and v are selected
                    M = max_colors
                    for vu in clusters[cu]:
                        for vv in clusters[cv]:
                            # If x[vu, cu] and x[vv, cv] both true (i.e. 1), then their cluster's colors must differ 
                            # This is implemented by ensuring that if x[vu, cu] = 1 and x[vv, cv] = 1, 
                            # then |color[cu] - color[cv]| >= 1 (i.e., different colors)
                            diff_pos, diff_neg = gp.Var(vtype=GRB.BINARY), gp.Var(vtype=GRB.BINARY)
                            
                            # If selected, ensure colors different:
                            # (x[vu, cu] = 1 and x[vv,cv] = 1) => (|cluster_colors[cu] - cluster_colors[cv]| >= 1)
                            model.addConstr(
                                cluster_colors[cu] - cluster_colors[cv] >= 1 - M*(2 - x[vu, cu] - x[vv, cvv]),
                                name=f"diff_pos_{vu}_{vv}"  
                            )
                            model.addConstr(
                                cluster_colors[cu] - cluster_colors[cv] <= -1 + M*(2 - x[vu, cu] - x[vv, cv]),
                                name=f"diff_neg_{vu}_{vv}"
                            )
                            
        # Objective: minimize used colors
        model.setObjective(gp.quicksum(z[k] for k in range(max_colors)), GRB.MINIMIZE)
        
        return model, x, cluster_colors, z
        
    model, x, cluster_colors, z = create_color_constraints()
    
    # Set callback function to log improvements
    best_solution_obj = float('inf')
    def mycallback(model, where):
        nonlocal best_solution_obj
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj < best_solution_obj:
                best_solution_obj = obj
                if logger:
                    logger.log(obj)

    model.optimize(mycallback)
    
    # Extract solution after optimization
    status = model.Status
    
    if status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        # Get results
        selected_per_cluster = {}
        selected_colors = {}
        
        for c_idx, cluster in enumerate(clusters):
            for v in cluster:
                if x[v, c_idx].x > 0.99:
                    selected_per_cluster[str(c_idx)] = v
                    selected_colors[v] = int(cluster_colors[c_idx].x + 0.5)  # Round to nearest int
                    break
        
        # Calculate objective value (number of unique colors used)
        used_colors = len(set(selected_colors.values()))
        objective_value = max(z[k].x for k in range(len(z)))  # Total number of colors used
    
        # Construct the solution dict
        solution = {
            "objective_value": len(set(selected_colors.values())),
            "selected_vertices": selected_per_cluster,
            "coloring": selected_colors
        }
        
        print(json.dumps(solution))
    
    else:
        # No feasible solution found 
        solution = {
            "objective_value": -1,  # Indicating failure
            "selected_vertices": {},
            "coloring": {}
        }
        print(json.dumps(solution))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', required=True)
    parser.add_argument('--solution_path', required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', required=False)
    
    args = parser.parse_args()
    
    # We'll use a simpler but robust implementation instead of building the full complex model as above
    
    # Read the instance
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    
    # Extract instance data
    graph = instance['graph']
    partition = instance['partition']
    
    n_vertices = graph['num_vertices']
    edges = graph['edges']
    n_clusters = partition['num_clusters']
    clusters = partition['clusters']
    
    # Create mapping from vertex to cluster
    vertex_to_cluster = {}
    for i, cluster in enumerate(clusters):
        for v in cluster:
            vertex_to_cluster[v] = i
    
    # Initialize solution logger
    logger = SolutionLogger(args.log_path, sense="minimize") if args.log_path else None
    
    # Create Gurobi model
    model = gp.Model("Graph_Selection_Coloring")
    model.setParam('TimeLimit', args.time_limit)
    model.setParam('OutputFlag', 0)  # Suppress solver output unless debugging
    
    # Variables:
    # x[v, c]: binary if vertex v in cluster c is selected 
    x = {}
    for c_idx, cluster in enumerate(clusters):
        for v in cluster:
            x[v, c_idx] = model.addVar(vtype=GRB.BINARY, name=f"x_v{v}_c{c_idx}")
            
    # One vertex per cluster
    for c_idx, cluster in enumerate(clusters):
        model.addConstr(gp.quicksum(x[v, c_idx] for v in cluster) == 1, name=f"one_per_{c_idx}")
    
    # Create color numbers for each available vertex in every cluster
    # Use high-level relaxation followed by symmetry handling 
    max_colors = min(n_vertices, n_clusters * 2)  # Upper bound on needed colors
    colors = {}
    for c_idx in range(n_clusters):
        colors[c_idx] = model.addVar(lb=0, ub=max_colors-1, vtype=GRB.INTEGER, name=f"color_cl_{c_idx}")
    
    # z[k] = whether color k is used
    z = {}
    for k in range(max_colors):
        z[k] = model.addVar(vtype=GRB.BINARY, name=f"z_color_{k}")
    
    # Enforce that z is active if any cluster uses color k:
    # Sum of (binary indicators per c,k if color[c] == k) <= z[k]*M, but it's better to directly state:
    # for color[c] = k, we need z[k] = 1
    for k in range(max_colors):
        for c_idx in range(n_clusters):
            # If color[c_idx] = k, then z[k] = 1
            # If color[c_idx] != k, z[k] may remain 0
            indicator_var = model.addVar(vtype=GRB.BINARY, name=f"eq_clr_{c_idx}_{k}")
            # Add constraints for color equality
            M = max_colors
            model.addConstr(colors[c_idx] >= k - M*(1-indicator_var), name=f"col_lbeq_{c_idx}_{k}")
            model.addConstr(colors[c_idx] <= k + M*(1-indicator_var), name=f"col_ubeq_{c_idx}_{k}")
            # If colors[c_idx] = k (then indicator_var=1), enforce z[k] = 1 
            model.addConstr(z[k] >= indicator_var, name=f"color_used_z_{c_idx}_{k}")
    
    # For each edge (u,v) where u and v belong to different clusters, 
    # ensure that if both vertices are selected, they have different cluster colors 
    for u, v_edge in edges:
        c_u = vertex_to_cluster[u]
        c_v = vertex_to_cluster[v_edge]
        
        if c_u != c_v:  # Different clusters - apply coloring constraint
            # To avoid same colors: if x[u, c_u] = 1 and x[v_edge, c_v] = 1 -> colors[c_u] != colors[c_v]
            # Big-M approach: If both selected, difference in colors >= 1 or <= -1
            diff_ind_ge = model.addVar(vtype=GRB.BINARY, name=f"diff_ge_{u}_{v_edge}")
            diff_ind_le = model.addVar(vtype=GRB.BINARY, name=f"diff_le_{u}_{v_edge}") 
            
            M_diff = max_colors
            
            # Constraint: If both are selected (x[u, c_u] = 1 ^ x[v_edge, c_v] = 1), 
            # then (colors[c_u] - colors[c_v] >= 1) OR (colors[c_u] - colors[c_v] <= -1)
            model.addConstr(
                colors[c_u] - colors[c_v] >= 1 - M_diff*(2 - x[u, c_u] - x[v_edge, c_v]) - M_diff*(1 - diff_ind_ge),
                name=f"diffgteq_{u}_{v_edge}"
            )
            
            model.addConstr(
                colors[c_u] - colors[c_v] <= -1 + M_diff*(2 - x[u, c_u] - x[v_edge, c_v]) + M_diff*(1 - diff_ind_le),
                name=f"difflteq_{u}_{v_edge}"
            )
            
            model.addConstr(diff_ind_ge + diff_ind_le >= 1, name=f"either_or_{u}_{v_edge}")
    
    # Objective: minimize number of colors used
    model.setObjective(gp.quicksum(z[k] for k in range(max_colors)), GRB.MINIMIZE)
    
        
    best_known_obj = float('inf')
    def log_callback(model, where):
        nonlocal best_known_obj
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            if obj < best_known_obj and obj > 0:
                best_known_obj = obj
                if logger:
                    logger.log(obj)

    model.optimize(log_callback)
    

    # Extract solution
    selected_vertices = {}
    coloring = {}
    
    # Check solution status
    if model.status == GRB.OPTIMAL or (model.status == GRB.TIME_LIMIT and model.SolCount > 0):
        for c_idx, cluster in enumerate(clusters):
            for v in cluster:
                if x[v, c_idx].x > 0.5:
                    selected_vertices[str(c_idx)] = v
                    coloring[v] = int(colors[c_idx].x)
                    break
    
        # Write solution to file
        with open(args.solution_path, 'w') as f:
            json.dump({
                "objective_value": int(model.objVal),
                "selected_vertices": selected_vertices,
                "coloring": coloring
            }, f)
    else:
        # Handle case with no solution found
        with open(args.solution_path, 'w') as f:
            json.dump({
                "objective_value": -1,
                "selected_vertices": {},
                "coloring": {}
            }, f)


if __name__ == "__main__":
    main()