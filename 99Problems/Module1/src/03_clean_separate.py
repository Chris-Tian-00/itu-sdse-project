#Cleaning
import numpy as np
import pandas as pd
from pprint import pprint

# from Module1.config import config as cfg
# from Module1.src.utils import load_csv_to_df, save_to_csv, describe_numeric_col, impute_missing_values

from config import config as cfg
from src.utils import load_csv_to_df, save_to_csv, describe_numeric_col, impute_missing_values



data = load_csv_to_df(cfg.data_feat_select_path)


for col in ["lead_indicator", "lead_id", "customer_code"]:
    if col in data.columns:
        data[col].replace("", np.nan, inplace=True)


if "lead_indicator" in data.columns:
    data = data.dropna(axis=0, subset=["lead_indicator"])

if "lead_id" in data.columns:
    data = data.dropna(axis=0, subset=["lead_id"])


if "source" in data.columns:
    data = data[data["source"] == "signup"]

if "lead_indicator" in data.columns:
    result = data["lead_indicator"].value_counts(normalize=True)

    print("Target value counter")
    for val, n in zip(result.index, result):
        print(val, ": ", n)
else:
    print("lead_indicator not found — skipping target distribution")


#Create and Separate
vars = [
    "lead_id", "lead_indicator", "customer_group",
    "onboarding", "source", "customer_code"
]

for col in vars:
    if col in data.columns:
        data[col] = data[col].astype("object")
        print(f"Changed {col} to object type")

cont_vars = data.loc[:, ((data.dtypes=="float64")|(data.dtypes=="int64"))]
cat_vars = data.loc[:, (data.dtypes=="object")]

print("\nContinuous columns: \n")
pprint(list(cont_vars.columns), indent=4)
print("\n Categorical columns: \n")
pprint(list(cat_vars.columns), indent=4)

#Outliers
cont_vars = cont_vars.apply(lambda x: x.clip(lower = (x.mean()-2*x.std()),
                                             upper = (x.mean()+2*x.std())))
outlier_summary = cont_vars.apply(describe_numeric_col).T
outlier_summary.to_csv(cfg.outlier_summary_path)
outlier_summary
cat_missing_impute = cat_vars.mode(numeric_only=False, dropna=True)
cat_missing_impute.to_csv(cfg.cat_missing_impute_path)

#Impute Data
cont_vars = cont_vars.apply(impute_missing_values)
cont_vars.apply(describe_numeric_col).T
if "customer_code" in cat_vars.columns:
    cat_vars.loc[cat_vars["customer_code"].isna(), "customer_code"] = "None"

cat_vars = cat_vars.apply(impute_missing_values)
cat_vars.apply(lambda x: pd.Series([x.count(), x.isnull().sum()], index = ['Count', 'Missing'])).T


save_to_csv(data, cfg.data_clean_seperate_path)
save_to_csv(cont_vars, cfg.cont_vars_clean_seperate_path)
save_to_csv(cat_vars, cfg.cat_vars_clean_seperate_path)
