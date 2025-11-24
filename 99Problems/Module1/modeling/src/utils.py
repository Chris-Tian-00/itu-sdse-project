# Train

def make_ml_directories():
    import os
    import shutil
    import config.config as cfg
    os.makedirs(cfg.artifacts_dir, exist_ok=True)
    os.makedirs(cfg.mlruns_dir, exist_ok=True)
    os.makedirs(cfg.ml_runs_trash_dir, exist_ok=True)

def ml_experiment_set():
    import mlflow
    import config.config as cfg
    mlflow.set_experiment(cfg.experiment_name)

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