
# Create categorical data columns
vars = [
    "lead_id", "lead_indicator", "customer_group", "onboarding", "source", "customer_code"
]

for col in vars:
    data[col] = data[col].astype("object")
    print(f"Changed {col} to object type")

# Separate categorical and continuous columns
cont_vars = data.loc[:, ((data.dtypes=="float64")|(data.dtypes=="int64"))]
cat_vars = data.loc[:, (data.dtypes=="object")]

print("\nContinuous columns: \n")
pprint(list(cont_vars.columns), indent=4)
print("\n Categorical columns: \n")
pprint(list(cat_vars.columns), indent=4)