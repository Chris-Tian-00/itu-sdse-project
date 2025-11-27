import os
import shutil
import pandas as pd
import warnings
import datetime
import json
from pprint import pprint

# dbutils.widgets.text("Training data max date", "2024-01-31")
# dbutils.widgets.text("Training data min date", "2024-01-01")
# max_date = dbutils.widgets.get("Training data max date")
# min_date = dbutils.widgets.get("Training data min date")




# 
MAX_DATE = "2024-01-31"
MIN_DATE = "2024-01-01"


# 
os.makedirs("artifacts", exist_ok=True)


# --- Suppress Warnings ---
warnings.filterwarnings('ignore')
pd.set_option('display.float_format', lambda x: "%.3f" % x)


# 
def describe_numeric_col(x):
    """
    Generate descriptive statistics for a numeric pandas Series.
    
    Parameters:
        x (pd.Series): Column to describe.
        
    Returns:
        pd.Series: Series with count, missing, mean, min, max.
    """
    return pd.Series(
        [x.count(), x.isnull().sum(), x.mean(), x.min(), x.max()],
        index=["Count", "Missing", "Mean", "Min", "Max"]
    )


def impute_missing_values(x, method="mean"):
    """
    Impute missing values in a pandas Series.
    
    Parameters:
        x (pd.Series): Column to impute.
        method (str): Imputation method ("mean" or "median" for numeric, mode for categorical).
        
    Returns:
        pd.Series: Series with missing values imputed.
    """
    if x.dtype in ["float64", "int64"]:
        return x.fillna(x.mean()) if method == "mean" else x.fillna(x.median())
    else:
        return x.fillna(x.mode()[0])


# 
print("Loading training data")
data = pd.read_csv("./artifacts/raw_data.csv")
print("Total rows:", len(data))
pprint(data.head(5))


# 
def parse_date(date_str):
    """Convert a date string to a date object."""
    return pd.to_datetime(date_str).date()


max_date = parse_date(MAX_DATE)
min_date = parse_date(MIN_DATE)


# 
data["date_part"] = pd.to_datetime(data["date_part"]).dt.date
data = data[(data["date_part"] >= min_date) & (data["date_part"] <= max_date)]


# 
min_date = data["date_part"].min()
max_date = data["date_part"].max()
date_limits = {"min_date": str(min_date), "max_date": str(max_date)}
with open("./artifacts/date_limits.json", "w") as f:
    json.dump(date_limits, f)


