# General

def save_to_csv(df, path):
    import pandas as pd
    df.to_csv(path, index=False)

def load_csv_to_df(path):
    import pandas as pd
    return pd.read_csv(path)


# Load

def describe_numeric_col(x):
    import pandas as pd
    """
    Parameters:
        x (pd.Series): Pandas col to describe.
    Output:
        y (pd.Series): Pandas series with descriptive stats. 
    """
    return pd.Series(
        [x.count(), x.isnull().sum(), x.mean(), x.min(), x.max()],
        index=["Count", "Missing", "Mean", "Min", "Max"]
    )


def impute_missing_values(x, method="mean"):
    """
    Parameters:
        x (pd.Series): Pandas col to describe.
        method (str): Values: "mean", "median"
    """
    if x.dtype in ["float64", "int64"]:
        return x.fillna(x.mean()) if method == "mean" else x.fillna(x.median())
    else:
        return x.fillna(x.mode()[0])
    
# Feature Selection

# Clean Seperate

# Combine Bin Save


# Train

def create_dummy_cols(df, col):
    import pandas as pd
    df_dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
    new_df = pd.concat([df, df_dummies], axis=1)
    new_df = new_df.drop(col, axis=1)
    return new_df


# Model Selection

def wait_until_ready(model_name, model_version):
    import time
    from mlflow.tracking.client import MlflowClient
    from mlflow.entities.model_registry.model_version_status import ModelVersionStatus

    client = MlflowClient()
    for _ in range(10):
        model_version_details = client.get_model_version(
          name=model_name,
          version=model_version,
        )
        status = ModelVersionStatus.from_string(model_version_details.status)
        print(f"Model status: {ModelVersionStatus.to_string(status)}")
        if status == ModelVersionStatus.READY:
            break
        time.sleep(1)

# Deploy

def wait_for_deployment(model_name, model_version, stage='Staging'):
    from mlflow.tracking import MlflowClient
    import time
    
    client = MlflowClient()
    status = False
    while not status:
        model_version_details = dict(
            client.get_model_version(name=model_name,version=model_version)
            )
        if model_version_details['current_stage'] == stage:
            print(f'Transition completed to {stage}')
            status = True
            break
        else:
            time.sleep(2)
    return status