import mlflow
import json
import pandas as pd
from mlflow.tracking import MlflowClient

from Module1.config import config as cfg
from Module1.src.utils import wait_until_ready


# --------------------------------------------------
# Load model results (always exists from step 05)
# --------------------------------------------------
with open(cfg.model_results_path, "r") as f:
    model_results = json.load(f)

results_df = pd.DataFrame.from_dict(model_results, orient="index")

if results_df.empty:
    print("No model results found — skipping model selection")
    exit(0)

print("Model results:")
print(results_df)

# --------------------------------------------------
# Pick best model from model_results.json
# --------------------------------------------------
if "f1-score" in results_df.columns:
    best_model = results_df["f1-score"].astype(float).idxmax()
else:
    best_model = results_df.index[0]

print(f"Best model from artifacts: {best_model}")

# --------------------------------------------------
# Try MLflow (optional, CI-safe)
# --------------------------------------------------
exp = mlflow.get_experiment_by_name(cfg.experiment_name)

if exp is None:
    print("No MLflow experiment found — skipping MLflow model selection")
    exit(0)

runs_df = mlflow.search_runs(
    experiment_ids=[exp.experiment_id],
    order_by=["metrics.f1_score DESC"],
    max_results=1,
)

if runs_df.empty:
    print("No MLflow runs found — skipping model registration")
    exit(0)

experiment_best = runs_df.iloc[0]
print("Best MLflow run:")
print(experiment_best)

run_id = experiment_best["run_id"]

# --------------------------------------------------
# Production model check
# --------------------------------------------------
client = MlflowClient()
prod_models = [
    m for m in client.search_model_versions(f"name='{cfg.model_name}'")
    if dict(m)["current_stage"] == "Production"
]

if prod_models:
    print("Production model exists — skipping auto-promotion in CI")
    exit(0)

# --------------------------------------------------
# Register model
# --------------------------------------------------
print(f"Registering model from run {run_id}")

model_uri = f"runs:/{run_id}/{cfg.artifact_path}"
model_details = mlflow.register_model(
    model_uri=model_uri,
    name=cfg.model_name,
)

wait_until_ready(model_details.name, model_details.version)
print("Model registered:", dict(model_details))
