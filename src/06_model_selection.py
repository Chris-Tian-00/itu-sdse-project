import mlflow
import json
import pandas as pd 
from mlflow.tracking import MlflowClient

from src.config import config as cfg
from src.utils import wait_until_ready


#mlflow.set_tracking_uri(f"file:{cfg.mlruns_dir}") # track in /artifacts
#print("MLflow tracking URI:", mlflow.get_tracking_uri())


#
experiment_ids = [mlflow.get_experiment_by_name(cfg.experiment_name).experiment_id]
print(experiment_ids)

#
experiment_best = mlflow.search_runs(
    experiment_ids=experiment_ids,
    order_by=["metrics.f1_score DESC"],
    max_results=1
).iloc[0]
print(experiment_best)

#
with open(cfg.model_results_path, "r") as f:
    model_results = json.load(f)
results_df = pd.DataFrame({model: val["weighted avg"] for model, val in model_results.items()}).T

print(results_df)

#
best_model = results_df.sort_values("f1-score", ascending=False).iloc[0].name
print(f"Best model: {best_model}")

#
client = MlflowClient()
prod_model = [model for model in client.search_model_versions(f"name='{cfg.model_name}'") if dict(model)['current_stage']=='Production']
prod_model_exists = len(prod_model)>0

if prod_model_exists:
    prod_model_version = dict(prod_model[0])['version']
    prod_model_run_id = dict(prod_model[0])['run_id']
    
    print('Production model name: ', cfg.model_name)
    print('Production model version:', prod_model_version)
    print('Production model run id:', prod_model_run_id)
    
else:
    print('No model in production')

#
train_model_score = experiment_best["metrics.f1_score"]
model_details = {}
model_status = {}
run_id = None

if prod_model_exists:
    data, details = mlflow.get_run(prod_model_run_id)
    prod_model_score = data[1]["metrics.f1_score"]

    model_status["current"] = train_model_score
    model_status["prod"] = prod_model_score

    if train_model_score>prod_model_score:
        print("Registering new model")
        run_id = experiment_best["run_id"]
else:
    print("No model in production")
    run_id = experiment_best["run_id"]

print(f"Registered model: {run_id}")

#
if run_id is not None:
    print(f'Best model found: {run_id}')

    model_uri = "runs:/{run_id}/{artifact_path}".format( #????
        run_id=run_id,
        artifact_path=cfg.artifact_path
    )
    model_details = mlflow.register_model(model_uri=model_uri, name=cfg.model_name)
    wait_until_ready(model_details.name, model_details.version)
    model_details = dict(model_details)
    print(model_details)


