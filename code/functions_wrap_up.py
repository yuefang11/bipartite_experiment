
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
from scipy.stats import norm
from math import comb
import networkx as nx
from networkx.algorithms import bipartite
import random
import statsmodels.api as sm
import time
from scipy.optimize import fsolve
from tqdm import tqdm
import collections
# from functions import *
import gurobipy as gp
from gurobipy import GRB
import cvxpy as cp
from cvxpy import MOSEK


def gen_graph(N, K, max_degree, dist):
    ind_set = [f'ind_{i}' for i in range(N)]
    grp_set = [f'grp_{i}' for i in range(K)]
    B = nx.Graph()
    B.add_nodes_from(ind_set, bipartite=0)  # Add nodes for set 1
    B.add_nodes_from(grp_set, bipartite=1)  # Add nodes for set 2
    degree_list = list(range(1, max_degree+1))
    
    if dist == 'uniform':
        node_degree = np.random.choice(degree_list, size=N, replace=True)
    elif dist == 'normal':
        mean = (max_degree + 1) / 2
        std_dev = (max_degree - 1) / 6  # Adjust this for your desired spread
        samples = np.random.normal(loc=mean, scale=std_dev, size=N)
        mapped_samples = np.clip(samples, 1, max_degree)
        node_degree = np.round(mapped_samples).astype(int)
    # elif dist == 'bimodal':

    elif dist == 'chi2': 
        samples = np.random.chisquare(4, N)
        normalized_samples = (samples - np.min(samples)) / (np.max(samples) - np.min(samples))
        scaled_samples = 1 + normalized_samples * (max_degree - 1)
        discrete_samples = np.floor(scaled_samples).astype(int)        
        node_degree = np.clip(discrete_samples, 1, max_degree)

    for node1 in ind_set:
        group_i = random.sample(grp_set, node_degree[ind_set.index(node1)])
        for node2 in group_i:
            B.add_edge(node1, node2)
    return B

def generate_adjacency_matrix(graph, set1, set2):
    # Get the number of nodes
    num_nodes1 = len(set1)
    num_nodes2 = len(set2)

    adjacency_matrix = np.zeros((num_nodes1, num_nodes2))
    for node in set1:
        neighbors = list(graph.neighbors(node))
        for neighbor in neighbors:
            adjacency_matrix[set1.index(node), set2.index(neighbor)] = 1
    return adjacency_matrix
def reduce_degree_to_max_replace_general(graph, deg_max):
    ind_nodes = [n for n, d in graph.nodes(data=True) if d['bipartite']==0]
    grp_nodes = [n for n, d in graph.nodes(data=True) if d['bipartite']==1]
    ind_degree_max = np.max([graph.degree(node) for node in ind_nodes])
    dic_degree = collections.defaultdict(set)
    for s in range(ind_degree_max, 1, -1):
        # dic_degree[s] = collections.defaultdict(set)
        pair_dict = collections.defaultdict(set)
        for type1_node in ind_nodes:
            if graph.degree(type1_node) == s:
                neighbors = set(graph.neighbors(type1_node))
                dic_degree[s].add(tuple(sorted(neighbors)))
                pair_dict[tuple(sorted(neighbors))].add(type1_node)
        # number of pairs of groups that share common members
        total_deg = len(dic_degree[s])
        print('S is', s, 'and number is', total_deg)
        # if the total degree is greater than the max_degree, randomly remove edges
        if total_deg > deg_max:
            num_to_remove = total_deg - deg_max
            to_remove = random.sample(list(dic_degree[s]), num_to_remove)
            remaining_sets = set(dic_degree[s]) - set(to_remove)
            
            for group_set in to_remove:
                for node in pair_dict[group_set]:
                    grp_to_join = random.sample(list(remaining_sets), 1)[0]

                    for grp_node in group_set:
                        graph.remove_edge(node, grp_node)
                    for grp_node in grp_to_join:
                        graph.add_edge(node, grp_node)
                    # print(node, grp_to_remove)
    return graph


def gen_outcome_covariate1(N, p, adj):
    beta1 = np.ones(p)
    beta0 = np.ones(p)
    degree = np.sum(adj, axis=1)
    X = np.random.rand(N, p)
    error1 = np.random.normal(0, 1, N) 
    error0 = np.random.normal(0, 1, N) 
    Y1 = 0.25 + np.dot(X, beta1) + error1
    Y0 = 0 + np.dot(X, beta0) + error0
    X = np.hstack((X, degree[:, None]))
    return (Y1, Y0, X)

def gen_outcome_covariate2(N, p, adj):
    beta = np.ones(p) * 0.1
    beta1 = np.ones(p) 
    beta0 = np.ones(p)
    # beta0 = np.random.uniform(0, 1, p)
    degree = np.sum(adj, axis=1)
    X = np.random.rand(N, p)
    error1 = np.random.normal(0, 1.5, N) 
    error0 = np.random.normal(0, 1.5, N) 
    Y1 = 0 + 0.1 * degree + np.dot(X, beta) + np.dot(X, beta1) + error1
    Y0 = 0 + np.dot(X, beta0) + error0
    X = np.hstack((X, degree[:, None]))
    return (Y1, Y0, X)

# def gen_outcome_covariate_alt2(N, p, adj):
#     beta1 = np.random.uniform(0, 1, p)
#     beta0 = np.random.uniform(0, 1, p)
#     degree = np.sum(adj, axis=1)
#     X = np.random.rand(N, p)
#     error1 = np.random.normal(0, 1, N) * 0.1
#     error0 = np.random.normal(0, 1, N) * 0.1
#     Y1 = 0.5 + np.dot(X, beta1) + error1
#     Y0 =  error0
#     return (Y1, Y0, X)

def gen_outcome_covariate3(N, p, adj):
    # beta = np.random.uniform(0, 1, p)
    beta1 = np.ones(p)
    beta0 = np.ones(p)
    alpha = np.random.uniform(0, 0.5, N)
    degree = np.sum(adj, axis=1)
    X = np.random.rand(N, p)
    error1 = np.random.normal(0, 1, N) 
    error0 = np.random.normal(0, 1, N) 
    Y1 =  alpha + np.dot(X, beta1) + error1
    Y0 = np.dot(X, beta0) + error0
    X = np.hstack((X, degree[:, None]))
    return (Y1, Y0, X)



def model_beta_est(p, outcome, adj, A, degree_ind, pair_matrix, weight_union):
    Y1, Y0, X = outcome
    N = len(Y1)
    
    dim = X.shape[1]
    
    # Calculate exposure and treatment indicators
    exposure = np.dot(adj, A)
    cov = exposure/degree_ind
    
    # Get indices for treated and control groups
    treat_idx = np.where(cov==1)[0]
    control_idx = np.where(cov==0)[0]
    N1 = len(treat_idx)
    N0 = len(control_idx)

    # Calculate weights and matrices
    weight_intersect1 = (1/p)**(pair_matrix)
    weight_intersect0 = (1/(1-p))**(pair_matrix)
    lambda1 = weight_intersect1 - 1
    lambda0 = weight_intersect0 - 1

    weight_union1 = (1/p)**(weight_union)
    weight_union0 = (1/(1-p))**(weight_union)

    lambda1_est = lambda1[cov==1][:,cov==1]
    lambda0_est = lambda0[cov==0][:,cov==0]

    weight_union1_est = weight_union1[cov==1][:,cov==1]
    weight_union0_est = weight_union0[cov==0][:,cov==0]

    lambda_tau = 1 * (pair_matrix >= 1)
    lambda_tau1 = lambda_tau[cov==1][:,cov==1]
    lambda_tau0 = lambda_tau[cov==0][:,cov==0]

    # Calculate basic estimates
    mu1_hat = np.sum(Y1[cov==1] / p**(degree_ind[cov==1])) / np.sum(1/p**(degree_ind[cov==1]))
    mu0_hat = np.sum(Y0[cov==0] / (1-p)**(degree_ind[cov==0])) / np.sum(1/(1-p)**(degree_ind[cov==0]))
    tau_est = mu1_hat - mu0_hat

    # Center the data
    X_demean = X - np.mean(X, axis=0)

    Y1_demean_est = Y1 - mu1_hat
    Y0_demean_est = Y0 - mu0_hat

    # Calculate naive variance estimates
    v1_est = ((Y1[cov==1] - mu1_hat)) @ (lambda1_est*weight_union1_est) @ (Y1[cov==1] - mu1_hat) / (np.sum(1/p**(degree_ind[cov==1])))**2
    v0_est = ((Y0[cov==0] - mu0_hat)) @ (lambda0_est*weight_union0_est) @ (Y0[cov==0] - mu0_hat) / (np.sum(1/(1-p)**(degree_ind[cov==0])))**2
    var_est = (np.sqrt(v1_est) + np.sqrt(v0_est))**2

    # calulcute variance estimates under covariate adjustment, without constraints
    # beta1_est_center = np.linalg.inv(X_demean[cov==1].T @ (lambda1_est*weight_union1_est) @ X_demean[cov==1]) @ (X_demean[cov==1].T @ ((lambda1_est*weight_union1_est) @ Y1_demean_est[cov==1])) 
    # beta0_est_center = np.linalg.inv(X_demean[cov==0].T @ (lambda0_est*weight_union0_est) @ X_demean[cov==0]) @ (X_demean[cov==0].T @ ((lambda0_est*weight_union0_est) @ Y0_demean_est[cov==0])) 

    # y1_pred_center = np.dot(X_demean, beta1_est_center)
    # y0_pred_center = np.dot(X_demean, beta0_est_center)
    # mu1_hat_adj = np.sum((Y1-y1_pred_center)[cov==1] / p**degree_ind[cov==1]) / np.sum(1/p**(degree_ind[cov==1]))
    # mu0_hat_adj = np.sum((Y0-y0_pred_center)[cov==0] / (1-p)**degree_ind[cov==0]) / np.sum(1/(1-p)**(degree_ind[cov==0]))
    # tau_est_adj_center = mu1_hat_adj - mu0_hat_adj

      
    # v1_adj_center = (Y1_demean_est[cov==1] - np.dot(X_demean[cov==1], beta1_est_center)) @ (lambda1_est * weight_union1_est) @ (Y1_demean_est[cov==1] - np.dot(X_demean[cov==1], beta1_est_center)) / (np.sum(1/p**(degree_ind[cov==1])))**2
    # v0_adj_center = (Y0_demean_est[cov==0] - np.dot(X_demean[cov==0], beta0_est_center)) @ (lambda0_est * weight_union0_est) @ (Y0_demean_est[cov==0] - np.dot(X_demean[cov==0], beta0_est_center)) / (np.sum(1/(1-p)**(degree_ind[cov==0])))**2
    # var_est_adj_center = (np.sqrt(v1_adj_center) + np.sqrt(v0_adj_center))**2

    block11 = X_demean.T @ lambda1 @ X_demean
    block12 = X_demean.T @ lambda_tau @ X_demean
    block21 = X_demean.T @ lambda_tau @ X_demean
    block22 = X_demean.T @ lambda0 @ X_demean


    block_matrix = np.block([[block11, block12],
                           [block21, block22]])
    b1 = X_demean[cov==1].T @ ((lambda1_est * weight_union1_est) @ Y1_demean_est[cov==1])/(np.sum(1/p**(degree_ind[cov==1]))/N)**2 + X_demean[cov==0].T @ ((lambda_tau0 * weight_union0_est) @ Y0_demean_est[cov==0])/(np.sum(1/(1-p)**(degree_ind[cov==0]))/N)**2
    b2 = X_demean[cov==1].T @ ((lambda_tau1 * weight_union1_est) @ Y1_demean_est[cov==1])/(np.sum(1/p**(degree_ind[cov==1]))/N)**2 + X_demean[cov==0].T @ ((lambda0_est * weight_union0_est) @ Y0_demean_est[cov==0])/(np.sum(1/(1-p)**(degree_ind[cov==0]))/N)**2

    # Stack b1 and b2 vertically to create b matrix
    b = np.concatenate([b1, b2])
    # print(b.shape)
    beta_adj_alt = np.linalg.inv(block_matrix) @ b

    # print(beta_adj_alt)
    # Split beta_adj_alt into beta1 and beta2 components
    dim = X_demean.shape[1]  # Get dimension of X
    beta1_adj_alt = beta_adj_alt[:dim]  # First half is beta1
    beta0_adj_alt = beta_adj_alt[dim:]  # Second half is beta2

    y1_pred_center_alt = np.dot(X_demean, beta1_adj_alt)
    y0_pred_center_alt = np.dot(X_demean, beta0_adj_alt)
    mu1_hat_adj_alt = np.sum((Y1-y1_pred_center_alt)[cov==1] / p**degree_ind[cov==1]) / np.sum(1/p**(degree_ind[cov==1]))
    mu0_hat_adj_alt = np.sum((Y0-y0_pred_center_alt)[cov==0] / (1-p)**degree_ind[cov==0]) / np.sum(1/(1-p)**(degree_ind[cov==0]))
    tau_est_adj_alt = mu1_hat_adj_alt - mu0_hat_adj_alt

    v1_adj_center_alt = (Y1_demean_est[cov==1] - np.dot(X_demean[cov==1], beta1_adj_alt)) @ (lambda1_est * weight_union1_est) @ (Y1_demean_est[cov==1] - np.dot(X_demean[cov==1], beta1_adj_alt)) / (np.sum(1/p**(degree_ind[cov==1])))**2
    v0_adj_center_alt = (Y0_demean_est[cov==0] - np.dot(X_demean[cov==0], beta0_adj_alt)) @ (lambda0_est * weight_union0_est) @ (Y0_demean_est[cov==0] - np.dot(X_demean[cov==0], beta0_adj_alt)) / (np.sum(1/(1-p)**(degree_ind[cov==0])))**2
    var_est_adj_alt = (np.sqrt(v1_adj_center_alt) + np.sqrt(v0_adj_center_alt))**2

    # calculate variance estimates under covariate adjustment, with constraints, conic optimization
    try:
        # Define optimization variables
        # Define optimization variables with explicit dimensions
        b1 = cp.Variable(dim) 
        b0 = cp.Variable(dim)  
        a1 = cp.Variable(1)  # scalar
        a0 = cp.Variable(1)  # scalar

        # Extract relevant submatrices
        X1 = X_demean[treat_idx]    # Shape: (N1, dim)
        X0 = X_demean[control_idx]   # Shape: (N0, dim)
        Y1_res = Y1_demean_est[treat_idx]  # Shape: (N1,)
        Y0_res = Y0_demean_est[control_idx]  # Shape: (N0,)

        # Weight matrices
        W1 = lambda1_est * weight_union1_est  # Shape: (N1, N1)
        W0 = lambda0_est * weight_union0_est  # Shape: (N0, N0)



        # Compute Cholesky decompositions for SOC constraints
        def ensure_psd(matrix, epsilon=1e-6):
            matrix = (matrix + matrix.T) / 2
            eigvals = np.linalg.eigvalsh(matrix)
            if np.min(eigvals) < epsilon:
                # print("Matrix is not PSD, adding a small multiple of the identity matrix")
                matrix += (epsilon - np.min(eigvals)) * np.eye(matrix.shape[0])
            return matrix

        W1_psd = ensure_psd(W1)
        W0_psd = ensure_psd(W0)
        L1 = np.linalg.cholesky(W1_psd)
        L0 = np.linalg.cholesky(W0_psd)

        
        constraints = []
        
        # First SOC constraint
        residual1 = Y1_res - X1 @ b1  # (N1,)
        constraints.append(
            cp.SOC(a1, L1 @ residual1)
        )
        
        # Second SOC constraint
        residual0 = Y0_res - X0 @ b0  # (N0,)
        constraints.append(
            cp.SOC(a0, L0 @ residual0)
        )

        # Cross term using block matrices
        X1_lambda1 = X1.T @ lambda1_est @ X1  
        X1_lambda_tau = X1.T @ lambda_tau1 @ X1  
        X0_lambda0 = X0.T @ lambda0_est @ X0  
        X0_lambda_tau = X0.T @ lambda_tau0 @ X0  

        Q = np.block([[X1_lambda1, X1_lambda_tau],
                     [X0_lambda_tau, X0_lambda0]])  
        Q = ensure_psd(Q)
        
        b = cp.hstack([b1, b0])  
        
        # Linear terms
        linear_term1 = X1.T @ (lambda1_est*weight_union1_est) @ Y1_res  
        linear_term2 = X0.T @ (lambda_tau0*weight_union0_est) @ Y0_res  
        linear_term3 = X1.T @ (lambda_tau1*weight_union1_est) @ Y1_res 
        linear_term4 = X0.T @ (lambda_tau0*weight_union0_est) @ Y0_res  
        
        linear_terms = np.concatenate([
            linear_term1 + linear_term2,
            linear_term3 + linear_term4
        ])  
        
        # Combined constraint
        constraints.append(cp.quad_form(b, Q) - 2 * linear_terms @ b <= 0)

        # Objective
        objective = cp.Minimize(a1 + a0)

        # Create and solve problem
        prob = cp.Problem(objective, constraints)
        
        if not prob.is_dcp():
            print("Problem is not DCP!")
            return (tau_est, tau_est_adj_alt, np.nan, var_est, var_est_adj_alt, np.nan)
        mosek_params = {
            'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': 1e-8,   # Relative gap tolerance
            'MSK_DPAR_INTPNT_CO_TOL_PFEAS': 1e-8,     # Primal feasibility tolerance
            'MSK_DPAR_INTPNT_CO_TOL_DFEAS': 1e-8,     # Dual feasibility tolerance
            'MSK_DPAR_INTPNT_CO_TOL_MU_RED': 1e-8,    # Relative complementarity gap tolerance
            'MSK_DPAR_INTPNT_CO_TOL_NEAR_REL': 3,     # Tolerance for numerical robustness
            'MSK_IPAR_INTPNT_MAX_ITERATIONS': 400      # Maximum number of iterations
        }
        
        result = prob.solve(solver=cp.MOSEK, mosek_params=mosek_params)
      

        if prob.status == cp.OPTIMAL:
            # Extract optimal coefficients
            beta1_optimal = b1.value
            beta0_optimal = b0.value

            # Calculate predictions
            y1_pred = X_demean @ beta1_optimal
            y0_pred = X_demean @ beta0_optimal

            # Calculate adjusted estimates
            mu1_hat_adj = np.sum((Y1-y1_pred)[treat_idx] / p**degree_ind[treat_idx]) / np.sum(1/p**(degree_ind[treat_idx]))
            mu0_hat_adj = np.sum((Y0-y0_pred)[control_idx] / (1-p)**degree_ind[control_idx]) / np.sum(1/(1-p)**(degree_ind[control_idx]))
            tau_est_adj = mu1_hat_adj - mu0_hat_adj

            # Calculate adjusted variances
            v1_adj = ((Y1_demean_est[treat_idx]-y1_pred[treat_idx]).T @ W1 @ (Y1_demean_est[treat_idx]-y1_pred[treat_idx])) / (np.sum(1/p**(degree_ind[treat_idx])))**2
            v0_adj = ((Y0_demean_est[control_idx]-y0_pred[control_idx]).T @ W0 @ (Y0_demean_est[control_idx]-y0_pred[control_idx])) / (np.sum(1/(1-p)**(degree_ind[control_idx])))**2
            var_est_adj = (np.sqrt(v1_adj) + np.sqrt(v0_adj))**2

            return (tau_est, tau_est_adj_alt, tau_est_adj, 
                    var_est, var_est_adj_alt, var_est_adj)
        else:
            print(f"Optimization failed with status: {prob.status}")
            return (tau_est, tau_est_adj_alt, tau_est_adj, 
                    var_est, var_est_adj_alt, var_est_adj)

    except Exception as e:
        print(f"Error in optimization: {str(e)}")
        return (tau_est, tau_est_adj_alt, tau_est_adj, 
                var_est, var_est_adj_alt, var_est_adj)