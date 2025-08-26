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
import rdata
import os
from math import radians, sin, cos, acos
import pickle
from datetime import datetime

date_str = datetime.now().strftime("%m%d")

path = '/home/ls/remote/bipartite/dataset/'

data = rdata.parser.parse_file(path+'analysis_dat.RData')
converted_data = rdata.conversion.convert(data)

df_plant = pd.DataFrame(converted_data['analysis_dat'])
df_aqi_m = pd.read_csv(path + 'annual_conc_by_monitor_2004.csv')
df_geo = pd.read_csv(path + 'geo_county.csv')
cnty_postal = pd.read_csv(path + 'state_postal.csv')


df_aqi_m = df_aqi_m[df_aqi_m['State Code'] != 'CC']
df_aqi_m['State Code'] = pd.to_numeric(df_aqi_m['State Code'], errors='coerce')
state_code = df_aqi_m['State Code']
cnty_code = df_aqi_m['County Code']
state_code_str = state_code.apply(lambda x: f"{int(x):02d}")
cnty_code_str = cnty_code.apply(lambda x: f"{int(x):03d}")
combined_code = state_code_str + cnty_code_str
df_aqi_m['FIPS'] = pd.to_numeric(combined_code)

merged_df = pd.merge(df_geo, cnty_postal, on='Postal')
merged_df_aqi = pd.merge(merged_df, df_aqi_m, on=['FIPS'], how='inner')

columns = ['FIPS', 'Population', 'Land Area', 'Water Area', 'Latitude', 'Longitude','Arithmetic Mean',
       'Arithmetic Standard Dev', '1st Max Value', '99th Percentile', '90th Percentile']
merged_df_aqi = merged_df_aqi[columns]
merged_df_aqi = merged_df_aqi[merged_df_aqi['Arithmetic Mean'] >=0]
merged_df_aqi['Population'] = merged_df_aqi['Population'].str.replace(',', '').astype(int)
merged_df_aqi = merged_df_aqi[merged_df_aqi['Population'] <= 1e6]

merged_df_aqi.dropna(inplace=True)
random.seed(121)
np.random.seed(121)
merged_df_aqi = merged_df_aqi.sample(frac=0.2) 
merged_df_aqi.reset_index(drop=True, inplace=True)
df_plant.dropna(inplace=True)
df_plant.reset_index(drop=True, inplace=True)

latitudes_aqi = merged_df_aqi['Latitude'].values
longitudes_aqi = merged_df_aqi['Longitude'].values
latitudes_plant = df_plant['Fac.Latitude'].values
longitudes_plant = df_plant['Fac.Longitude'].values

latitudes_aqi_rad = np.radians(latitudes_aqi)
longitudes_aqi_rad = np.radians(longitudes_aqi)
latitudes_plant_rad = np.radians(latitudes_plant)
longitudes_plant_rad = np.radians(longitudes_plant)

# Calculate differences and apply the Haversine formula
dlat = latitudes_plant_rad[:, np.newaxis] - latitudes_aqi_rad
dlon = longitudes_plant_rad[:, np.newaxis] - longitudes_aqi_rad

a = np.sin(dlat/2)**2 + np.cos(latitudes_aqi_rad) * np.cos(latitudes_plant_rad[:, np.newaxis]) * np.sin(dlon/2)**2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

# Earth radius in kilometers (6371.01 km)
dist_matrix_m = (6371.01 * c).T
nearest_dist = np.min(dist_matrix_m, axis=1)
merged_df_aqi['nearest_dist'] = (nearest_dist)

adjacency_matrix = 1 * (dist_matrix_m <= 30)
idx_to_remove_i = np.where((np.sum(adjacency_matrix, axis=1) == 0) | (np.sum(adjacency_matrix, axis=1) > 5))[0]
idx_to_remove_j = np.where(np.sum(adjacency_matrix, axis=0) == 0)[0]
data_left = merged_df_aqi[~merged_df_aqi.index.isin(idx_to_remove_i)]
data_left = data_left.reset_index(drop=True)
data_right = df_plant[~df_plant.index.isin(idx_to_remove_j)]
data_right = data_right.reset_index(drop=True)
adj = np.delete(adjacency_matrix, idx_to_remove_i, axis=0)
adj = np.delete(adj, idx_to_remove_j, axis=1)
degree_left = np.sum(adj, axis=1)
data_left['degree'] = degree_left

degree_right = np.sum(adj, axis=0)
# treat = data_right['SnCR']*1


X1 = data_left[['Population']]/1000000
X1 = X1.replace(',', '', regex=True).astype(float)
X1_log = np.log(X1)
X2 = data_left[['nearest_dist']]/30
# X2_log = np.log(X2)
X = np.concatenate([X1, X2], axis=1)

N = adj.shape[0]
N_plant = adj.shape[1]
p = 0.5
print(adj.shape)
pair_matrix = np.dot(adj, adj.T)


deg_matrix = np.broadcast_to(degree_left[:, None], (degree_left.size, degree_left.size))
weight_union = deg_matrix + deg_matrix.T - pair_matrix

random.seed(121)
np.random.seed(121)
beta = np.array([100, -30]) 
tau = -10 * np.ones(X.shape[0])
Y1 = 50 + X @ beta + tau + np.random.normal(0, 1, X.shape[0])
Y0 = 50 + X @ beta + np.random.normal(0, 1, X.shape[0])
tau_true1 = np.mean(Y1) - np.mean(Y0)

X_all = np.concatenate([X, degree_left[:, None]], axis=1)
data1 = (Y1, Y0, X_all)
result_dict1 = {"N": N, "K": N_plant, "p": p, "outcome": data1, "adj": adj, "degree_ind": degree_left,
                "pair_matrix": pair_matrix, "weight_union": weight_union, "tau": tau_true1}
with open(f'/home/ls/remote/bipartite/code/results/real_data_sim1_{date_str}.pkl', 'wb') as f:
    pickle.dump(result_dict1, f)


random.seed(121)
np.random.seed(121)
tau_1 = np.array([1, -6])  # Reduced coefficient for second component
tau = X @ tau_1  -  degree_left  # Reduced coefficient for degree term
Y1 = 65 + X @ beta + tau + np.random.normal(0, 10, X.shape[0])
Y0 = 70 + X @ beta + np.random.normal(0, 10, X.shape[0]) 
tau_true2 = np.mean(Y1) - np.mean(Y0)
X_all = np.concatenate([X, degree_left[:, None]], axis=1)
data2 = (Y1, Y0, X_all)
result_dict2 = {"N": N, "K": N_plant, "p": p, "outcome": data2, "adj": adj, "degree_ind": degree_left,
                "pair_matrix": pair_matrix, "weight_union": weight_union, "tau": tau_true2}
with open(f'/home/ls/remote/bipartite/code/results/real_data_sim2_{date_str}.pkl', 'wb') as f:
    pickle.dump(result_dict2, f)


random.seed(121)
np.random.seed(121)
tau = np.random.uniform(-10, 0, X.shape[0])
Y1 = 65 + X @ beta + tau + np.random.normal(0, 10, X.shape[0])
Y0 = 70 + X @ beta + np.random.normal(0, 10, X.shape[0]) 
tau_true3 = np.mean(Y1) - np.mean(Y0)     
X_all = np.concatenate([X, degree_left[:, None]], axis=1)
data3 = (Y1, Y0, X_all)
result_dict3 = {"N": N, "K": N_plant, "p": p, "outcome": data3, "adj": adj, "degree_ind": degree_left, "pair_matrix": pair_matrix, "weight_union": weight_union, "tau": tau_true3}
with open(f'/home/ls/remote/bipartite/code/results/real_data_sim3_{date_str}.pkl', 'wb') as f:
    pickle.dump(result_dict3, f)



# Y = data_left['Arithmetic Mean']
# X = data_left[['Population', 'Land Area' ,'Water Area']]
# X = X.replace(',', '', regex=True).astype(float)
# X_log = np.log(X)  
# X_log['Degree'] = degree_left


