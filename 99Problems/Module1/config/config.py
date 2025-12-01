# Load

MAX_DATE = "2024-01-31"
MIN_DATE = "2024-01-01"

artifacts_dir = "./artifacts"
data_path = "./artifacts/raw_data.csv"
date_limits_path = ""#./artifacts/date_limits.json

data_load_path = f"./{artifacts_dir}/01_data_load.csv"

# Feature Selection

data_feat_select_path = f"./{artifacts_dir}/02_data_feat_select.csv"

# Clean Seperate

outlier_summary_path = './artifacts/outlier_summary.csv'
cat_missing_impute_path = "./artifacts/cat_missing_impute.csv"

data_clean_seperate_path = f"./{artifacts_dir}/03_data_clean_seperate.csv"
cont_vars_clean_seperate_path = f"./{artifacts_dir}/01_cont_vars_clean_seperate.csv"
cat_vars_clean_seperate_path = f"./{artifacts_dir}/01_cat_vars_clean_seperate.csv"

# Combine Bin Save

scaler_path = "./artifacts/scaler.pkl"

column_drift_path = './artifacts/columns_drift.json'
training_data_path = './artifacts/training_data.csv'

values_list = ['li', 'organic','signup','fb']

mapping = {'li' : 'socials', 
           'fb' : 'socials', 
           'organic': 'group1', 
           'signup': 'group1'
           }

data_gold_path = ""#./artifacts/train_data_gold.csv

# Train

import datetime
import datetime
from scipy.stats import uniform, randint

current_date = datetime.datetime.now().strftime("%Y_%B_%d")
data_version = "00000"
experiment_name = current_date
mlruns_dir = "" #mlruns
ml_runs_trash_dir = "" #mlruns/.trash

cat_cols = ["customer_group", "onboarding", "bin_source", "source"]

params_xgbrf = {
    "learning_rate": uniform(1e-2, 3e-1),
    "min_split_loss": uniform(0, 10),
    "max_depth": randint(3, 10),
    "subsample": uniform(0, 1),
    "objective": ["reg:squarederror", "binary:logistic", "reg:logistic"],
    "eval_metric": ["aucpr", "error"]
}
xgboost_model_path = "" #./artifacts/lead_model_xgboost.json

lr_model_path = "" # ./artifacts/lead_model_lr.pkl
params_lr = {
            'solver': ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
            'penalty':  ["none", "l1", "l2", "elasticnet"],
            'C' : [100, 10, 1.0, 0.1, 0.01]
}


column_list_path = "" # ./artifacts/columns_list.json
 
model_results_path = "" #./artifacts/model_results.json

# Model Selection

artifact_path = "model"
model_name = "lead_model"


# Deploy
model_version = 1
