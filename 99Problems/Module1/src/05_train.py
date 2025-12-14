import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBRFClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, cohen_kappa_score, f1_score
from scipy.stats import uniform, randint
import mlflow.pyfunc
from sklearn.linear_model import LogisticRegression
import os
from pprint import pprint
import matplotlib.pyplot as plt
import joblib
import json
import shutil

from Module1.config import config as cfg
from Module1.src.utils import create_dummy_cols



os.makedirs(cfg.artifacts_dir, exist_ok=True)
os.makedirs(cfg.mlruns_dir, exist_ok=True)
os.makedirs(cfg.ml_runs_trash_dir, exist_ok=True)


mlflow.set_experiment(cfg.experiment_name)


#
data = pd.read_csv(cfg.data_gold_path)
print(f"Training data length: {len(data)}")
data.head(5)

#
data = data.drop(
    ["lead_id", "customer_code", "date_part"],
    axis=1,
    errors="ignore"
)


cat_vars = data[cfg.cat_cols]
other_vars = data.drop(cfg.cat_cols, axis=1)

#
for col in cat_vars:
    cat_vars[col] = cat_vars[col].astype("category")
    cat_vars = create_dummy_cols(cat_vars, col)

data = pd.concat([other_vars, cat_vars], axis=1)

for col in data:
    data[col] = data[col].astype("float64")
    print(f"Changed column {col} to float")

#
y = data["lead_indicator"]
X = data.drop(["lead_indicator"], axis=1)

#
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.15, stratify=y
)

print(y_train.head(5))


#
model = XGBRFClassifier(random_state=42)


model_grid = RandomizedSearchCV(model, param_distributions=cfg.params_xgbrf, n_jobs=-1, verbose=3, n_iter=10, cv=10)

model_grid.fit(X_train, y_train)

#
best_model_xgboost_params = model_grid.best_params_
print("Best xgboost params")
pprint(best_model_xgboost_params)

y_pred_train = model_grid.predict(X_train)
y_pred_test = model_grid.predict(X_test)
print("Accuracy train", accuracy_score(y_pred_train, y_train ))
print("Accuracy test", accuracy_score(y_pred_test, y_test))

#
conf_matrix = confusion_matrix(y_test, y_pred_test)
print("Test actual/predicted\n")
print(pd.crosstab(y_test, y_pred_test, rownames=['Actual'], colnames=['Predicted'], margins=True),'\n')
print("Classification report\n")
print(classification_report(y_test, y_pred_test),'\n')

conf_matrix = confusion_matrix(y_train, y_pred_train)
print("Train actual/predicted\n")
print(pd.crosstab(y_train, y_pred_train, rownames=['Actual'], colnames=['Predicted'], margins=True),'\n')
print("Classification report\n")
print(classification_report(y_train, y_pred_train),'\n')

#
xgboost_model = model_grid.best_estimator_
xgboost_model_path = cfg.xgboost_model_path
xgboost_model.save_model(xgboost_model_path)

model_results = {
    xgboost_model_path: classification_report(y_train, y_pred_train, output_dict=True)
}

#
class lr_wrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model
    
    def predict(self, context, model_input):
        return self.model.predict_proba(model_input)[:, 1]


mlflow.sklearn.autolog(log_input_examples=True, log_models=False)
experiment_id = mlflow.get_experiment_by_name(cfg.experiment_name).experiment_id

with mlflow.start_run(experiment_id=experiment_id) as run:
    model = LogisticRegression()

    model_grid = RandomizedSearchCV(model, param_distributions= cfg.params_lr, verbose=3, n_iter=10, cv=3)
    model_grid.fit(X_train, y_train)

    best_model = model_grid.best_estimator_

    y_pred_train = model_grid.predict(X_train)
    y_pred_test = model_grid.predict(X_test)


    # log artifacts
    mlflow.log_metric('f1_score', f1_score(y_test, y_pred_test))
    mlflow.log_artifacts(cfg.artifacts_dir, artifact_path="model_artifacts")
    mlflow.log_param("data_version", cfg.data_version)
    
    # store model for model interpretability
    joblib.dump(value=model, filename=cfg.lr_model_path)
        
    # Custom python model for predicting probability 
    mlflow.pyfunc.log_model('model', python_model=lr_wrapper(model))


model_classification_report = classification_report(y_test, y_pred_test, output_dict=True)

best_model_lr_params = model_grid.best_params_

print("Best lr params")
pprint(best_model_lr_params)

print("Accuracy train:", accuracy_score(y_pred_train, y_train ))
print("Accuracy test:", accuracy_score(y_pred_test, y_test))

conf_matrix = confusion_matrix(y_test, y_pred_test)
print("Test actual/predicted\n")
print(pd.crosstab(y_test, y_pred_test, rownames=['Actual'], colnames=['Predicted'], margins=True),'\n')
print("Classification report\n")
print(classification_report(y_test, y_pred_test),'\n')

conf_matrix = confusion_matrix(y_train, y_pred_train)
print("Train actual/predicted\n")
print(pd.crosstab(y_train, y_pred_train, rownames=['Actual'], colnames=['Predicted'], margins=True),'\n')
print("Classification report\n")
print(classification_report(y_train, y_pred_train),'\n')

model_results[cfg.lr_model_path] = model_classification_report
print(model_classification_report["weighted avg"]["f1-score"])

#
with open(cfg.column_list_path, 'w+') as columns_file:
    columns = {'column_names': list(X_train.columns)}
    pprint(columns)
    json.dump(columns, columns_file)

print('Saved column list to ', cfg.column_list_path)

with open(cfg.model_results_path, 'w+') as results_file:
    json.dump(model_results, results_file)
