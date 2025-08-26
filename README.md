# bipartite_experiment
Data and code for "Design-based causal inference in bipartite experiments".

## data
- *`analysis_dat.RData`*: this dataset is originally from [dataverse](https://dataverse.harvard.edu/dataverse/dapsm) of paper "Adjusting for Unmeasured Spatial Confounding with Distance Adjusted Propensity Score Matching", by Papadogeorgou, Choirat, and Zigler
- *`annual_conc_by_monitor_2005.csv`*: this dataset is downloaded from [EPA website](https://aqs.epa.gov/aqsweb/airdata/download_files.html)
- *`geo_county.csv`*: this dataset provides the demographic information of each county, which is from the census.gov
- *`state_postal.csv`*: this dataset provides the two-letter postal abbreviation for each state

## code
- *`functions_wrap_up.py`*: provides all the functions needed for further analysis
- *`run_sim_gen_data.py`*: provide code of generating datasets based on simulated bipartite graphs
- *`real_data_gen.py`*: generate datasets based on real bipartite graphs
- *`run_sim_parallel.py`*: obtain results based on different datasets
