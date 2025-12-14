import os
import shutil
import pandas as pd
import warnings
import datetime
import json
from pprint import pprint

from config import config as cfg
from src.utils import save_to_csv

# dbutils.widgets.text("Training data max date", "2024-01-31")
# dbutils.widgets.text("Training data min date", "2024-01-01")
# max_date = dbutils.widgets.get("Training data max date")
# min_date = dbutils.widgets.get("Training data min date")




# 



# 
os.makedirs(cfg.artifacts_dir, exist_ok=True) #if folder already exists, this does nothing

# CI fallback: create dummy data if raw_data.csv is missing
if not os.path.exists(cfg.data_path):
    print("raw_data.csv not found — creating dummy dataset for CI")
    dummy = pd.DataFrame({
        "date_part": pd.date_range("2024-01-01", periods=3),
        "target": [0, 1, 0]
    })
    dummy.to_csv(cfg.data_path, index=False)



# --- Suppress Warnings ---
warnings.filterwarnings('ignore')
pd.set_option('display.float_format', lambda x: "%.3f" % x)


# 



# 
print("Loading training data")
data = pd.read_csv(cfg.data_path)
print("Total rows:", len(data))
pprint(data.head(5))


# 
def parse_date(date_str):
    """Convert a date string to a date object."""
    return pd.to_datetime(date_str).date()


max_date = parse_date(cfg.MAX_DATE)
min_date = parse_date(cfg.MIN_DATE)


# 
data["date_part"] = pd.to_datetime(data["date_part"]).dt.date
data = data[(data["date_part"] >= min_date) & (data["date_part"] <= max_date)]


# 
min_date = data["date_part"].min()
max_date = data["date_part"].max()
date_limits = {"min_date": str(min_date), "max_date": str(max_date)}
with open(cfg.date_limits_path, "w") as f:
    json.dump(date_limits, f)


save_to_csv(data, cfg.data_load_path)

