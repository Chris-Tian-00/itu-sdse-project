from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

from Module1.config import config as cfg
from Module1.src.utils import wait_for_deployment

client = MlflowClient()

# --------------------------------------------------
# Check if model exists at all
# --------------------------------------------------
try:
    versions = client.search_model_versions(f"name='{cfg.model_name}'")
except MlflowException:
    print("MLflow registry not available — skipping deployment")
    exit(0)

if not versions:
    print("No registered model versions found — skipping deployment")
    exit(0)

# --------------------------------------------------
# Pick latest version
# --------------------------------------------------
latest_version = max(int(v.version) for v in versions)
print(f"Latest model version: {latest_version}")

model_version_details = dict(
    client.get_model_version(
        name=cfg.model_name,
        version=str(latest_version),
    )
)

# --------------------------------------------------
# Transition to Staging if needed
# --------------------------------------------------
if model_version_details["current_stage"] != "Staging":
    print("Transitioning model to Staging")

    client.transition_model_version_stage(
        name=cfg.model_name,
        version=str(latest_version),
        stage="Staging",
        archive_existing_versions=True,
    )

    wait_for_deployment(cfg.model_name, str(latest_version), "Staging")

else:
    print("Model already in Staging")
