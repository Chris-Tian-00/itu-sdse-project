# Train

import datetime
import datetime
from scipy.stats import uniform, randint

current_date = datetime.datetime.now().strftime("%Y_%B_%d")
data_gold_path = ""#./artifacts/train_data_gold.csv
data_version = "00000"
experiment_name = current_date
artifacts_dir = "" #artifacts
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
