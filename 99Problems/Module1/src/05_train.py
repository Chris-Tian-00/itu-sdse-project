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

# Select only categorical columns that actually exist
existing_cat_cols = [c for c in cfg.cat_cols if c in data.columns]

if len(existing_cat_cols) == 0:
    print("No categorical columns found — skipping categorical processing")
    cat_vars = pd.DataFrame(index=data.index)
    other_vars = data.copy()
else:
    cat_vars = data[existing_cat_cols]
    other_vars = data.drop(existing_cat_cols, axis=1)


#
for col in cat_vars.columns:
    cat_vars[col] = cat_vars[col].astype("category")
    cat_vars = create_dummy_cols(cat_vars, col)


data = pd.concat([other_vars, cat_vars], axis=1)

for col in data:
    data[col] = data[col].astype("float64")
    print(f"Changed column {col} to float")

#
# Determine target column (CI vs real data)
if "lead_indicator" in data.columns:
    target_col = "lead_indicator"
elif "target" in data.columns:
    target_col = "target"
else:
    raise ValueError("No target column found in training data")

y = data[target_col]
X = data.drop([target_col], axis=1)


#
# Safe train-test split for tiny CI data
if len(y.unique()) > 1 and len(y) >= 10:
    stratify_arg = y
else:
    stratify_arg = None

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    random_state=42,
    test_size=0.2 if len(y) >= 5 else 0.5,
    stratify=stratify_arg,
)


print(y_train.head(5))


#
model = XGBRFClassifier(random_state=42)


# Adjust CV for CI
cv_folds = 3 if len(X_train) >= 10 else 2

model_grid = RandomizedSearchCV(
    model,
    param_distributions=cfg.params_xgbrf,
    n_jobs=-1,
    verbose=3,
    n_iter=1 if len(X_train) < 10 else 10,
    cv=cv_folds,
)


if len(X_train) < 10:
    print("Skipping XGBoost training — not enough data for CI")
    model_results = {}
else:
    model_grid.fit(X_train, y_train)
    ...


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
exp = mlflow.get_experiment_by_name(cfg.experiment_name)

if exp is None:
    print("MLflow experiment not found — skipping MLflow logging")
    experiment_id = None
else:
    experiment_id = exp.experiment_id


if experiment_id is not None:
    with mlflow.start_run(experiment_id=experiment_id) as run:
        model = LogisticRegression()

        lr_cv = 3 if len(X_train) >= 10 else 2
        lr_iter = 10 if len(X_train) >= 10 else 1

        model_grid = RandomizedSearchCV(
            model,
            param_distributions=cfg.params_lr,
            verbose=3,
            n_iter=lr_iter,
            cv=lr_cv,
        )

        model_grid.fit(X_train, y_train)

        best_model = model_grid.best_estimator_

        y_pred_train = model_grid.predict(X_train)
        y_pred_test = model_grid.predict(X_test)

        # log artifacts
        if len(y_test.unique()) > 1:
            mlflow.log_metric("f1_score", f1_score(y_test, y_pred_test))

        mlflow.log_artifacts(cfg.artifacts_dir, artifact_path="model_artifacts")
        mlflow.log_param("data_version", cfg.data_version)

        # store model
        joblib.dump(value=best_model, filename=cfg.lr_model_path)

        # Custom python model for predicting probability
        mlflow.pyfunc.log_model("model", python_model=lr_wrapper(best_model))

else:
    print("Skipping MLflow run in CI")

# Safe metrics & reports for CI

if len(y_test.unique()) > 1:
    model_classification_report = classification_report(
        y_test, y_pred_test, output_dict=True
    )
else:
    print("Skipping classification report — single class in y_test")
    model_classification_report = {"weighted avg": {"f1-score": 0.0}}

best_model_lr_params = model_grid.best_params_

print("Best lr params")
pprint(best_model_lr_params)

print("Accuracy train:", accuracy_score(y_pred_train, y_train))
print("Accuracy test:", accuracy_score(y_pred_test, y_test))

# Test confusion matrix
if len(y_test.unique()) > 1:
    conf_matrix = confusion_matrix(y_test, y_pred_test)
    print("Test actual/predicted\n")
    print(
        pd.crosstab(
            y_test,
            y_pred_test,
            rownames=["Actual"],
            colnames=["Predicted"],
            margins=True,
        ),
        "\n",
    )
    print("Classification report\n")
    print(classification_report(y_test, y_pred_test), "\n")
else:
    print("Skipping test confusion matrix — single class in y_test")

# Train confusion matrix
if len(y_train.unique()) > 1:
    conf_matrix = confusion_matrix(y_train, y_pred_train)
    print("Train actual/predicted\n")
    print(
        pd.crosstab(
            y_train,
            y_pred_train,
            rownames=["Actual"],
            colnames=["Predicted"],
            margins=True,
        ),
        "\n",
    )
    print("Classification report\n")
    print(classification_report(y_train, y_pred_train), "\n")
else:
    print("Skipping train confusion matrix — single class in y_train")

model_results[cfg.lr_model_path] = model_classification_report
print(model_classification_report["weighted avg"]["f1-score"])

# Always write outputs (CI-safe)
if not model_results:
    model_results = {"ci_dummy": {"f1-score": 0.0}}

with open(cfg.model_results_path, "w") as f:
    json.dump(model_results, f)

with open(cfg.column_list_path, "w") as f:
    json.dump({"column_names": list(X.columns)}, f)
