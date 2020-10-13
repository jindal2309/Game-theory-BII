# Sudo code

def best_response():
	for t in range(T):
        for u in G.nodes():
        	# Group edges having same component
        	edge_group = dict of grouped edges of node u # {component_id: list(edges)}
        	for edge_list in edge_group.values:
        		if reduction_in_cost(edge_list)  > 0:
        			update_strategy(edge_list)
        			if check_Nash():
        				return x, check_Nash


def update_strategy(x, edge_list):
	social_dist_cost = 0
	if x[edge_list[0]] == 0:  #No social distance
		remove_edge(edge_list)
		for edge in edge_list:
            x[(edge[0], edge[1])] = 1
            x[(edge[1], edge[0])] = 1
            social_dist_cost += Csd[edge]
	else:
		add_edge(edge)
		for edge in edge_list:
            x[(edge[0], edge[1])] = 0
            x[(edge[1], edge[0])] = 0

	infection_cost = comp_len[comp_id[u]]*Cinf[u]/(len(x)+0.0)
	cost[u] = social_dist_cost + infection_cost
	return cost[u]

def add_edge(edge_list):
	#add edges from edge_list and combine components
	#use ids starting from comp_max_id + 1

def remove_edge(edge_list):
	#remove edges from edge_list and split its component
	#use ids starting from comp_max_id + 1

