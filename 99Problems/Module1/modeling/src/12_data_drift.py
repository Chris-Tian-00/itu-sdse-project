# data drift
data_columns = list(data.columns)
with open('./artifacts/columns_drift.json','w+') as f:           
    json.dump(data_columns,f)
    
data.to_csv('./artifacts/training_data.csv', index=False)