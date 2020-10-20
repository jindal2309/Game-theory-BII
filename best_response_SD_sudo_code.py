# Sudo code

def best_response():
	for t in range(T):
        for u in G.nodes():
        	# conn_edge_group: connected edge component group; <list> of edges grouped by components and their component id
        	# not_conn_edge_group: edge which are not connected grouped by components; <list> 
        	# conn_comp_len: dictionary containing length of components belonging to connected edges
        	conn_edge_group, not_conn_edge_group, conn_comp_len = get_SD_components()
        	conn_edge_group = subset_lists(conn_edge_group)
            not_conn_edge_group = subset_lists(not_conn_edge_group)

        	# TODO: combination of all components
        	for conn_edge_list in conn_edge_group:
                for not_conn_edge_list in not_conn_edge_group:
	        		if reduction_in_cost(conn_edge_list, not_conn_edge_list)  > 0:
	        			update_strategy(conn_edge_list, not_conn_edge_list)
	        			if check_Nash():
	        				return x, check_Nash


def subset_lists(my_list):
	'''Generate 2^len(my_list) combinations'''
    subs = []
    for i in range(0, len(my_list)+1):
      temp = [list(x) for x in combinations(my_list, i)]
      if len(temp)>0:
        subs.extend(temp)
    return subs

def get_SD_components(u):
    c <- component[u]
    H = Graph(); H.node = c.node; H.edge = c.edge - edge from node u
    comp <- connected_component(H)
    conn_comp_d <- dict of {component_id: list of nodes belonging to component_id}
    conn_comp_id <- dict of {node: component_id}

    conn_edge_group <- list of [(edge_list1, component_id1), (edge_list2, component_id2)]
    not_conn_edge_group -< list; [edge_list1, edge_list2]
    return conn_edge_group, not_conn_edge_group



def update_strategy(x, edge_comb_list):

    initial_infection_cost = comp_len[comp_id[u]]*Cinf[u]/(len(x)+0.0)
    social_dist_cost = 0
    for (edge_list, conn_component_id) in conn_edge_list:
        for edge in edge_list:
            if x[edge] == 0 and x[(edge[1], edge[0])] != 1: # x[edge] 0-> 1
                comp_d, comp_id, comp_len, comp_max_id = remove_edge([edge])
                x[edge] = 1
                social_dist_cost += Csd[edge]

    for edge_list in not_conn_edge_list:
        for edge in edge_list:
            if x[edge] == 1 and x[(edge[1], edge[0])] != 1: #x[edge] 1-> 0
                comp_d, comp_id, comp_len, comp_max_id = add_edge([edge])
                x[edge] = 0
                social_dist_cost -= Csd[edge]

    new_infection_cost = comp_len[comp_id[u]]*Cinf[u]/(len(x)+0.0)
    cost[u] = cost[u] + social_dist_cost + new_infection_cost - initial_infection_cost
    return x, comp_d, comp_id, comp_len, cost, comp_max_id


def reduction_in_cost(conn_edge_list, not_conn_edge_list):
	cost_reduction = 0; z = 0; nbr_comp = {};
    for (edge_list, conn_component_id) in conn_edge_list:
        for edge in edge_list:
            if x[(edge[1], edge[0])] != 1:
                cost_reduction -= Csd[edge]
                nbr_comp[conn_component_id] = 1


        for conn_component_id in nbr_comp:
        z += conn_comp_len[conn_component_id]

    # Socially distant edges: current x[edge_list[0]] == 1
    # Cost reduced by adding component back
    nbr_comp = {}
    for edge_list in not_conn_edge_list:
        for edge in edge_list:
            if x[(edge[1], edge[0])] != 1:
                cost_reduction += Csd[edge]
                nbr_comp[comp_id[edge[1]]] = 1

        
    for j in nbr_comp: 
        z -= comp_len[j]
    cost_reduction += z*Cinf[u]/(len(x)+0.0)
    return cost_reduction




def add_edge(edge_list):
	#add edges from edge_list and combine components
	#use ids starting from comp_max_id + 1

def remove_edge(edge_list):
	#remove edges from edge_list and split its component
	#use ids starting from comp_max_id + 1

