'''
Csd[i] = cost of vaccination for node i
Cinf[i]: cost of infection for node i
x[i]: current strategy of node i
    x[i] = 1 ==> i is vaccinated
S(x): set of vaccinated nodes
comp(x): components formed by residual nodes
cost[i]: cost of node i

#Evaluating reduction in cost for node i
#return old cost - new cost
def reduction_in_cost(x, comp, cost, Csd, Cinf, i)
    if x[i] == 0, then return  cost[i] - Csd[i]
    if x[i] == 1
        A = {comp(j, x): j is a nbr of i }
        N = \sum_{X in A} |X|
        return  Csd[i] - N^2 Cinf[i]/n

#best response
def best_respose(Csd, Cinf)
    xinit: random strategy
    initialize comp, cost
    for t = 1.. T:
        for i in V:
            if reduction_in_cost(x, i) > 0:
                flip x[i]
                update comp

#possible efficiencies
    #uniform Csd, Cinf setting
        if there is a large comp X: check benefit of vaccination
        if all comp are small: check benefit of not vaccinating
'''
import numpy as np
import networkx as nx
#import EoN
import matplotlib.pyplot as plt
import csv, random, pdb, sys
from IPython.core.debugger import set_trace
import collections
from itertools import combinations

#each line: id1, id2
def read_graph(fname):
    G = nx.Graph()
    fp_reader = csv.reader(open(fname), delimiter = ',')
    for line in fp_reader:
        G.add_edge(line[0], line[1])
    return G

#create components
#x: strategy vector where x[i] = 1 means i is vaccinated
def init_comp(G, x):
    
    # comp_id: {node u: component_id i}; mapping of each node to it's current component id
    # comp_len: {component_id i: length(int)}; mapping of component id to its length
    # comp_d: {component_id i: list of node in ith component}
    # max_comp_id: integer; each time we create a new component id so it will be helpful.
    comp_id = {}; comp_len = {}; comp_d = {}; max_comp_id = 0
    
    # first create a graph H which is a copy of G and then get connected component of H
    H = nx.Graph()
    for u in G.nodes(): H.add_node(u)
    for e in G.edges():
        u = e[0]; v = e[1]
        if x[(u,v)] == 0: #edge is not social distanced
            H.add_edge(u, v)
    comp = nx.connected_components(H)

    # comp is a list of list; for each component we assign a component id and 
    for c in list(comp):
        for u in c: 
            comp_id[u] = max_comp_id
        comp_len[max_comp_id] = len(list(c))
        comp_d[max_comp_id] = list(c)
        max_comp_id += 1
#     for u in x:
#         if x[u] == 1: comp_id[u] = -1
        
    return H, comp_d, comp_id, comp_len, max_comp_id

def comp_cost(x, comp_id, comp_len, Csd, Cinf):
    cost = {}
    # Assumption: cost of Social distancing is added to both nodes
    for edge in x:
        if x[edge] == 1: 
            cost[edge[0]] = cost.get(edge[0],0) + Csd[edge]
    
    #print("len of comp_len", len(comp_len))
    for i in Cinf:
        cost[i] = cost.get(i,0) + comp_len[comp_id[i]]*Cinf[i]/(len(x)+0.0)
    return cost


def check_NE(G, x, comp_d, comp_id, comp_len, cost, Csd, Cinf):
    num_violated = 0
    for u in G.nodes():
        conn_edge_group, not_conn_edge_group, conn_comp_len = get_SD_components(G, x, comp_id, comp_len, comp_d, u)
        conn_edge_group = subset_lists(conn_edge_group)
        not_conn_edge_group = subset_lists(not_conn_edge_group)

        violated_flag = False
        for conn_edge_list in conn_edge_group:
                for not_conn_edge_list in not_conn_edge_group:
                    if reduction_in_cost(G, x, comp_id, comp_len, cost, Csd, Cinf, u, conn_edge_list, not_conn_edge_list, conn_comp_len) > 0: 
                        violated_flag = True
                        break
            
        if violated_flag == True:
            num_violated += 1

    return num_violated

#remove edges from edge_list and split its comp
#use ids starting from comp_max_id + 1
def remove_edge(G, x, comp_d, comp_id, comp_len, comp_max_id, u, edge_list):
    
    C = set(comp_d[comp_id[u]])
    edge_list = set(edge_list)
    comp_max_id += 1;
    #print('ff', u, 'comp_d=', comp_d, 'comp_id=', comp_id, 'C=', C, 'comp_max_id=', comp_max_id)
    H = nx.Graph()
    for v in C: 
        H.add_node(v)

    # Remove edges that are in edge_list    
    for v1 in C: 
        for v2 in G.neighbors(v1):
            if (v1,v2) not in edge_list: 
                H.add_edge(v1, v2)
    comp1 = nx.connected_components(H)
    comp = list(comp1).copy()
    #print('fff', H.nodes(), H.edges(), list(comp))
    
    for c in list(comp):
        comp_max_id += 1
        comp_d[comp_max_id] = list(c); 
        comp_len[comp_max_id] = len(c)
        for v in list(c): 
            comp_id[v] = comp_max_id
    #print('gg', u,  'comp_d=', comp_d, 'comp_id=', comp_id, 'comp_max_id=', comp_max_id)    
    return comp_d, comp_id, comp_len, comp_max_id

#add edges from edge_list and create comp
#use ids starting from comp_max_id + 1
def add_edge(G, x, comp_d, comp_id, comp_len, comp_max_id, u, edge_list):
    Tu = comp_d[comp_id[u]].copy()
    del comp_d[comp_id[u]]
    del comp_len[comp_id[u]]
    comp_max_id += 1
    S = Tu

    for vprime in Tu: 
        comp_id[vprime] = comp_max_id
    
    for edge in edge_list:
        u1,v = edge
        T = []
        if comp_id[v] != comp_id[u] and comp_id[v] in comp_d:
            T = comp_d[comp_id[v]].copy()
        elif comp_id[v] == comp_id[u]:
            T = Tu
        S = S + T
        if comp_id[v] in comp_d: 
            del comp_d[comp_id[v]]
            del comp_len[comp_id[v]]
        #comp_id[v] = comp_max_id
        for vprime in T: 
            comp_id[vprime] = comp_max_id
#             for vprime in x:
#                 if comp_id[vprime] not in comp_len: print('err3', vprime, u, v)
    #merge the components containing S into one
    comp_id[u] = comp_max_id
    comp_d[comp_max_id] = S
    comp_len[comp_max_id] = len(S)

    return comp_d, comp_id, comp_len, comp_max_id

# return reduction in cost if edge_list flips its strategy
def reduction_in_cost(G, x, comp_id, comp_len, cost, Csd, Cinf, u, conn_edge_list, not_conn_edge_list, conn_comp_len):

    # If current state of edge_list is not socially distant; 
    # cost increase = cost of making all edge to SD - cost of infection reduced from SD
    # conn_edge_list: currently connected to components that we want to seperate
    cost_reduction = 0
    # z: nodes reduced from the component; nodes seperated - nodes added
    z = 0
    nbr_comp = {}
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
   
#flip strategy of node u
def update_strategy(x, G, H, comp_d, comp_id, comp_len, cost, Csd, Cinf, comp_max_id, u, conn_edge_list, not_conn_edge_list):

    social_dist_cost = 0
    initial_infection_cost = comp_len[comp_id[u]]*Cinf[u]/(len(x)+0.0)
    for (edge_list, conn_component_id) in conn_edge_list:
        for edge in edge_list:
            if x[edge] == 0 and x[(edge[1], edge[0])] != 1:
                # x[edge] 0-> 1
                comp_d, comp_id, comp_len, comp_max_id = remove_edge(G, x, comp_d, 
                                                                             comp_id, comp_len, comp_max_id, u, [edge])
                x[edge] = 1
                social_dist_cost += Csd[edge]


    for edge_list in not_conn_edge_list:
        for edge in edge_list:
            if x[edge] == 1 and x[(edge[1], edge[0])] != 1:
                #x[edge] 1-> 0
                comp_d, comp_id, comp_len, comp_max_id = add_edge(G, x,
                                                                  comp_d, comp_id, comp_len, comp_max_id, u, [edge])
                x[edge] = 0
                social_dist_cost -= Csd[edge]

    new_infection_cost = comp_len[comp_id[u]]*Cinf[u]/(len(x)+0.0)
    cost[u] = cost[u] + social_dist_cost + new_infection_cost - initial_infection_cost
    return x, comp_d, comp_id, comp_len, cost, comp_max_id
    


def get_SD_components(G, x, comp_id, comp_len, comp_d, u):
    curr_comp = list(comp_d[comp_id[u]])
    H = nx.Graph()
    for v in curr_comp: 
        H.add_node(v)

    # Remove all incident edges from node u
    # Build a graph of nodes/edges only from curr_comp with no edges on u
    for v1 in curr_comp: 
        for v2 in G.neighbors(v1):
            if not (v1 == u or v2 == u) and (v2 in curr_comp): 
                H.add_edge(v1, v2)
                    
    comp1 = nx.connected_components(H)
    comp = list(comp1).copy()
    # print(comp)
    # print(H.edges)
    # print(nx.draw(H))

    conn_comp_max_id = 0
    conn_comp_d = {}
    conn_comp_id = {}
    conn_comp_len = {}
    for c in list(comp):
        conn_comp_d[conn_comp_max_id] = list(c); 
        conn_comp_len[conn_comp_max_id] = len(c)
        for v in list(c): 
            conn_comp_id[v] = conn_comp_max_id
        conn_comp_max_id += 1

    conn_edge_group = collections.defaultdict(list)
    not_conn_edge_group = collections.defaultdict(list)

    for v in G.neighbors(u):
        if v in curr_comp:
            neighbour_comp_id = conn_comp_id[v]
            conn_edge_group[neighbour_comp_id].append((u,v))
        else:
            neighbour_comp_id = comp_id[v]
            not_conn_edge_group[neighbour_comp_id].append((u,v))
            
    not_conn_edge_group = list(not_conn_edge_group.values())
    conn_edge_group1 = []
    for key, val in conn_edge_group.items():
        conn_edge_group1.append((val, key))
    # print(conn_edge_group)
    # print(not_conn_edge_group)
    return conn_edge_group1, not_conn_edge_group, conn_comp_len


def subset_lists(my_list):
    subs = []
    for i in range(0, len(my_list)+1):
      temp = [list(x) for x in combinations(my_list, i)]
      if len(temp)>0:
        subs.extend(temp)
    return subs


#start at strategy x and run for T steps
def best_response(G, Csd, Cinf, x, T, epsilon=0.05):
    if len(x) == 0:
        for u in G.nodes(): x[u] = np.random.randint(0, 2)
    
    H, comp_d, comp_id, comp_len, comp_max_id = init_comp(G, x)
    cost = comp_cost(x, comp_id, comp_len, Csd, Cinf)
    for t in range(T):
        #u = random.choice(list(V)); 
        for u in G.nodes():
            conn_edge_group, not_conn_edge_group, conn_comp_len = get_SD_components(G, x, comp_id, comp_len, comp_d, u)
            conn_edge_group = subset_lists(conn_edge_group)
            not_conn_edge_group = subset_lists(not_conn_edge_group)

            for conn_edge_list in conn_edge_group:
                for not_conn_edge_list in not_conn_edge_group:
                    if reduction_in_cost(G, x, comp_id, comp_len, cost, Csd, Cinf, u, conn_edge_list, not_conn_edge_list, conn_comp_len) > 0:
                        x, comp_d, comp_id, comp_len, cost, comp_max_id = update_strategy(x, 
                                            G, H, comp_d, comp_id, comp_len, cost, Csd, Cinf, comp_max_id, u, conn_edge_list, not_conn_edge_list)

                        if check_NE(G, x, comp_d, comp_id, comp_len, cost, Csd, Cinf) < epsilon*len(x):
                            return x, check_NE(G, x, comp_d, comp_id, comp_len, cost, Csd, Cinf)
    return x, check_NE(G, x, comp_d, comp_id, comp_len, cost, Csd, Cinf)



if __name__ == '__main__':
### run for a fixed network and fixed alpha
###########################################
    
    
    T = int(sys.argv[1])
    epsilon = float(sys.argv[2])
    alphavals = sys.argv[3].split(',')

    #### read from a fixed graph
#     fname = sys.argv[4]
#     G = read_graph(fname)

    ## random graphs
    n = int(sys.argv[4]); 
    m = int(sys.argv[5])
    # n: Number of nodes; m: Number of edges to attach from a new node to existing nodes
    G = nx.barabasi_albert_graph(n, m)

      
    for alpha in alphavals:
        #print("Started for alpha: ", alpha)
        x = {}; Csd = {}; Cinf = {}; #alpha = 10
        for u in G.nodes():
            #print(u, x[u])
            Cinf[u] = 1*float(alpha)

        for edge in G.edges():
            # TODO: use np.random.randint(0, 2) or constant
            # 0: no social distance; 1: social distance
            u,v = edge
            x[(u,v)] = np.random.randint(0, 2);
            x[(v,u)] = np.random.randint(0, 2);
            Csd[(u,v)] = 1;
            Csd[(v,u)] = 1;
        
        #T = 500
        x, nviol = best_response(G, Csd, Cinf, x, T, epsilon)
        print("alpha: ", alpha, "Num violated: ", nviol/len(x), "Social distanced edge: ", len([i for i in x if x[i] == 1]), "Len of x", len(x))
        #print("x", x)
        sys.stdout.flush()