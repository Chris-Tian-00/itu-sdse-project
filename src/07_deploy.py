from mlflow.tracking import MlflowClient
import time

from src.config import config as cfg
from src.utils import wait_for_deployment

client = MlflowClient()

model_version_details = dict(client.get_model_version(name=cfg.model_name,version=cfg.model_version))
model_status = True
if model_version_details['current_stage'] != 'Staging':
    client.transition_model_version_stage(
        name=cfg.model_name,
        version=cfg.model_version,stage="Staging", 
        archive_existing_versions=True
    )
    model_status = wait_for_deployment(cfg.model_name, cfg.model_version, 'Staging')
else:
    print('Model already in staging')