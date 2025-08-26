
from functions_0729 import *
import time
from datetime import datetime
from tqdm import tqdm
import pickle
date_str = datetime.now().strftime("%Y%m%d")

N_list = [2000, 5000, 10000]
K_list = [50, 100, 500] 
ind_max_degree_list = [2, 5, 5]
nx.set_random_state(42)
for N, K, ind_max_degree in zip(N_list, K_list, ind_max_degree_list):
    dist = 'normal'
    G = gen_graph(N, K, ind_max_degree, dist)
    deg_max = 500
    G = reduce_degree_to_max_replace_general(G, deg_max)
    ind_nodes = [n for n, d in G.nodes(data=True) if d['bipartite']==0]
    grp_nodes = [n for n, d in G.nodes(data=True) if d['bipartite']==1]

    adj = generate_adjacency_matrix(G, ind_nodes, grp_nodes)
    degree_ind = np.sum(adj, axis=1).astype(int)
    pair_matrix = np.dot(adj, adj.T)

    deg_matrix = np.broadcast_to(degree_ind[:, None], (degree_ind.size, degree_ind.size))
    weight_union = deg_matrix + deg_matrix.T - pair_matrix

    p = 0.5
    p_x = 2

    outcome = gen_outcome_covariate1(N, p_x, adj)
    tau = np.mean(outcome[0] - outcome[1])
    result_dict = {"N": N, "K": K, "p":p, "outcome": outcome, "adj": adj, "degree_ind": degree_ind, "pair_matrix": pair_matrix, "weight_union": weight_union, "tau": tau}
    with open(f'/home/ls/remote/bipartite/Bipartite_experiment/code/results/simulation_data1_{N}_{K}_{date_str}.pkl', 'wb') as file:
       pickle.dump(result_dict, file)


    outcome = gen_outcome_covariate2(N, p_x, adj)
    tau = np.mean(outcome[0] - outcome[1])
    result_dict = {"N": N, "K": K, "p":p, "outcome": outcome, "adj": adj, "degree_ind": degree_ind, "pair_matrix": pair_matrix, "weight_union": weight_union, "tau": tau}
    with open(f'/home/ls/remote/bipartite/Bipartite_experiment/code/results/simulation_data2_{N}_{K}_{date_str}.pkl', 'wb') as file:
       pickle.dump(result_dict, file)


    outcome = gen_outcome_covariate3(N, p_x, adj)
    tau = np.mean(outcome[0] - outcome[1])
    result_dict = {"N": N, "K": K, "p":p, "outcome": outcome, "adj": adj, "degree_ind": degree_ind, "pair_matrix": pair_matrix, "weight_union": weight_union, "tau": tau}
    with open(f'/home/ls/remote/bipartite/Bipartite_experiment/code/results/simulation_data3_{N}_{K}_{date_str}.pkl', 'wb') as file:
        pickle.dump(result_dict, file)


