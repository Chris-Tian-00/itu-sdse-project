# ITU-sdse-project

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Exam_Project

## Contributors

| Name               | Email        |
|--------------------|--------------|
| **Christian Ainis** | chai@itu.dk  |
| **Lorand Ladnai**   | lorl@itu.dk  |
| **Wiktor Pedrycz**  | wipe@itu.dk  |
| **Matei Pop**       | mapop@itu.dk |


## Original Provided Materials (other/ Directory)

To separate the original course materials from our refactored MLOps pipeline, we placed everything from the initial forked repository into the other/ directory. This folder contains the dataset, the instructor’s main.ipynb notebook, and supporting scripts that formed the starting point of the project.

From these materials, we extracted the relevant logic, cleaned it, and restructured it into modular pipeline components inside 99Problems/Module1/src/. This makes our contribution clear: transforming the original notebook-based workflow into a maintainable, production-style MLOps pipeline using Cookiecutter, DVC, and Dagger.

## Project Organization

```
other/                                      <- Original materials from the course repository
│
├── docs/                                   <- Provided diagrams and architecture references
│   ├── diagrams.excalidraw
│   └── project-architecture.png
│
├── notebooks/                              <- Instructor's initial notebook-based pipeline
│   │
│   ├── .dvc/                               <- DVC metadata from the original project
│   │   ├── .gitignore
│   │   └── config
│   │
│   ├── artifacts/                          <- Example artifacts created by main.ipynb
│   │   ├── .gitignore
│   │   ├── X_test.csv
│   │   ├── lead_model_lr.pkl
│   │   ├── raw_data.csv.dvc
│   │   └── y_test.csv
│   │
│   ├── .dvcignore
│   ├── .gitignore
│   ├── main.ipynb                          <- Instructor’s original unstructured notebook
│   ├── model_inference.py                  <- Example inference script
│   └── requirements.txt                    <- Original environment requirements
│
├── workflows/                              <- Sample CI workflow from instructor
│   └── test_action.yml
│
├── README.md                               <- README from the original repository
└── action.yml                              <- Additional GitHub Action example
```
```
.
├── .dvc/                      # DVC internal metadata and cache
│ ├── cache/                   # Cached versions of data files
│ ├── config                   # DVC configuration
│ └── tmp/                     # Temporary DVC runtime files
├── .dvcignore                 # Files ignored by DVC
│
├── .github/
│ └── workflows/
│ └── pipeline.yml             # GitHub Actions CI pipeline (train + validate)
│
├── CC_produced/               # Cookiecutter Data Science scaffold (unused)
│                              # Kept for reference only
│
├── artifacts/                 # Intermediate artifacts produced by pipeline steps
│ ├── 01_data_load.csv
│ ├── 02_data_feat_select.csv
│ ├── 03_data_clean_seperate.csv
│ ├── train_data_gold.csv
│ ├── outlier_summary.csv
│ ├── columns_drift.json
│ └── date_limits.json
│
├── dagger/
│ ├── go.mod                   # Go module definition for Dagger
│ ├── go.sum                   # Go dependency lockfile
│ └── pipeline.go              # Dagger pipeline definition (end-to-end ML workflow)
│
├── data/
│ ├── raw_data.csv.dvc # DVC-tracked raw dataset
│ └── raw_data.csv             # Materialized data (git ignored)
│
├── model/                     # Trained models and model-related artifacts
│ ├── lead_model_lr.pkl        # Logistic regression model
│ ├── lead_model_xgboost.json  # XGBoost model
│ ├── model.pkl                # Final exported model (used by validator)
│ ├── scaler.pkl               # Feature scaler
│ ├── columns_list.json        # Feature list used during training
│ └── model_results.json       # Training and evaluation metrics
│
├── other/                     # Original provided materials (course content)
│                              # Includes original notebooks and scripts
│
├── src/                       # Source code for the ML pipeline
│ ├── 01_load.py               # Load and initial preprocessing of raw data
│ ├── 02_feature_selection.py  # Feature selection logic
│ ├── 03_clean_separate.py     # Data cleaning and separation
│ ├── 04_combine_bin_save.py   # Feature engineering and binning
│ ├── 05_train.py              # Model training
│ ├── 06_model_selection.py    # Model comparison and selection
│ ├── 07_deploy.py             # Model export and deployment preparation
│ ├── utils.py                 # Shared utility functions
│ └── config/
│ └── config.py                # Centralized configuration values
│
├── tests/
│ └── test_utils.py            # Unit tests for utility functions
│
├── requirements.txt           # Python dependencies
├── pyproject.toml             # Python project metadata
├── Makefile                   # Convenience commands (optional)
└── README.md                  # Project documentation

```

## Running the Pipeline via GitHub Actions (CI)

The full ML pipeline (training and validation) is executed using **GitHub Actions**. The workflow configuration is located at: `.github/workflows/pipeline.yml`.


### Manually Triggering the Workflow (`workflow_dispatch`)

To manually run the training and validation pipeline:

1. Find the **Actions** tab in the GitHub repository
2. In the left sidebar, select **Dagger Pipeline CI**
3. Click the **Run workflow** button (gray)
4. Choose the **main** branch from the dropdown menu
5. Click **Run workflow** (green)

This will start the full pipeline, including both training and validation steps.

---

### Accessing the Model Artifact

The trained model is uploaded as a GitHub Actions artifact named **`model`**.

After the workflow has completed:

1. Open the completed workflow run from the **Actions** tab
2. Scroll down to the **Artifacts** section
3. Locate the artifact named **model**
4. Click the download icon or the artifact name

The downloaded artifact contains `model.pkl`, which represents the best trained model produced by the pipeline.


--------
