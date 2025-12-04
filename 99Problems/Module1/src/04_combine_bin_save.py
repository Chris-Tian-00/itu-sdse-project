from sklearn.preprocessing import MinMaxScaler
import joblib
import pandas as pd
import json

from Module1.config import config as cfg
from Module1.src.utils import load_csv_to_df, save_to_csv


data = load_csv_to_df(cfg.data_clean_seperate_path)
cont_vars = load_csv_to_df(cfg.cont_vars_clean_seperate_path)
cat_vars = load_csv_to_df(cfg.cat_vars_clean_seperate_path)

# standardization

scaler = MinMaxScaler()
scaler.fit(cont_vars)

joblib.dump(value=scaler, filename=cfg.scaler_path)
print("Saved scaler in artifacts")

cont_vars = pd.DataFrame(scaler.transform(cont_vars), columns=cont_vars.columns)
# cont_vars


# combine data
cont_vars = cont_vars.reset_index(drop=True)
cat_vars = cat_vars.reset_index(drop=True)
data = pd.concat([cat_vars, cont_vars], axis=1)
print(f"Data cleansed and combined.\nRows: {len(data)}")

# data drift
data_columns = list(data.columns)
with open(cfg.column_drift_path,'w+') as f:           
    json.dump(data_columns,f)
    
data.to_csv(cfg.training_data_path, index=False)

# binning object columns
data['bin_source'] = data['source']
data.loc[~data['source'].isin(cfg.values_list),'bin_source'] = 'Others'


data['bin_source'] = data['source'].map(cfg.mapping)

#
#spark.sql(f"drop table if exists train_gold")
# data_gold = spark.createDataFrame(data)
# data_gold.write.saveAsTable('train_gold')
# dbutils.notebook.exit(('training_golden_data',most_recent_date))

data.to_csv(cfg.data_gold_path, index=False)