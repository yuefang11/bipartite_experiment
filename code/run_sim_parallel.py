import multiprocessing
# from functions_0729 import *
from functions_wrap_up_1208 import *
import time
from datetime import datetime
from tqdm import tqdm
import pickle
import collections


with open('/home/ls/remote/bipartite/code/results/real_data_sim1_1220.pkl', 'rb') as file:
    data = pickle.load(file)

N = data['N']
K = data['K']
p = data['p']
outcome = data['outcome']

adj = data['adj']
degree_ind = data['degree_ind']
pair_matrix = data['pair_matrix']
weight_union = data['weight_union']
tau = data['tau']


def simulate_iter(index):
    random.seed(index)
    np.random.seed(index)
    A = np.random.binomial(1, p, K)
    result = model_beta_est(p, outcome, adj, A, degree_ind, pair_matrix, weight_union)

    if index % 100 == 0:
        print(index, time.time() - start)
    return result

result_list = []
MC = 1000
start = time.time()


if __name__ == '__main__':    
    
    # Create a pool of workers and map the simulate function to each index
    pool_obj = multiprocessing.Pool(10)
    results =  pool_obj.map(simulate_iter,range(MC))
    pool_obj.close()

    col_name = ["tau_est",  "tau_est_adj_center_optimal" , "var_est",  "var_est_adj_center_optimal"]
    df = pd.DataFrame(results, columns=col_name)

    df['tau'] = tau * np.ones(MC)

    date_str = datetime.now().strftime("%Y%m%d")
    df.to_csv(f"/home/ls/remote/bipartite/code/results/results_1218/real_data_result1_{date_str}.csv", index=False)
