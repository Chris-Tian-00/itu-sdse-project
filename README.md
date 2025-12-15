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
99Problems/
│
├── Module1/                           <- Core Python ML pipeline module
│   │
│   ├── .dvc/                          <- DVC internal metadata
│   │   ├── .gitignore
│   │   └── config
│   │
│   ├── .dvcignore                     <- Ignore rules for DVC
│   │
│   ├── artifacts/                     <- Raw data tracked by DVC
│   │   ├── .gitignore
│   │   ├── .gitkeep
│   │   └── raw_data.csv.dvc          <- Pointer to raw dataset
│   │
│   ├── config/                        <- Config settings for pipeline
│   │   ├── __init__.py
│   │   └── config.py                  <- Central configuration variables
│   │
│   ├── src/                           <- All ML pipeline steps
│   │   ├── 01_load.py                 <- Load raw data
│   │   ├── 02_feature_selection.py    <- Select relevant features
│   │   ├── 03_clean_separate.py       <- Clean & split variables
│   │   ├── 04_combine_bin_save.py     <- Combine & bin features
│   │   ├── 05_train.py                <- Train ML models
│   │   ├── 06_model_selection.py      <- Choose best model
│   │   ├── 07_deploy.py               <- Prepare deployment assets
│   │   ├── __init__.py
│   │   ├── test_utils.py              <- Tests for utilities
│   │   └── utils.py                   <- Shared helper functions
│   │
│   ├── plots.py                       <- Visualization utilities
│   └── __init__.py
│
│
├── dagger/                             <- Dagger containerized pipeline
│   │
│   ├── artifacts/                      <- All processed data artifacts
│   │   ├── .gitignore
│   │   ├── .gitkeep
│   │   ├── 01_cat_vars_clean_seperate.csv   <- Cleaned categorical vars
│   │   ├── 01_cont_vars_clean_seperate.csv  <- Cleaned continuous vars
│   │   ├── 01_data_load.csv                 <- Loaded dataset output
│   │   ├── 02_data_feat_select.csv          <- Feature-selected dataset
│   │   ├── 03_data_clean_seperate.csv       <- Clean + separated summary
│   │   ├── cat_missing_impute.csv           <- Missing categorical imputations
│   │   ├── columns_drift.json               <- Drift detection results
│   │   ├── columns_list.json                <- List of valid columns
│   │   ├── date_limits.json                 <- Allowed date boundaries
│   │   ├── model_results.json               <- Model evaluation summary
│   │   ├── outlier_summary.csv              <- Outlier analysis report
│   │   ├── raw_data.csv.dvc                 <- DVC pointer to dataset
│   │   ├── train_data_gold.csv              <- Final gold training data
│   │   └── training_data.csv                <- Prepared training dataset
│   │
│   ├── models/                         <- All trained ML models
│   │   ├── lead_model_lr.pkl           <- Logistic Regression model
│   │   ├── lead_model_xgboost.json     <- XGBoost trained model
│   │   └── scaler.pkl                  <- Feature scaler for inference
│   │
│   ├── go.mod                          <- Go module file
│   ├── go.sum                          <- Go dependencies checksums
│   └── pipeline.go                     <- Dagger pipeline definition
│
│
├── docs/                                <- Documentation system (MkDocs)
│   ├── mkdocs.yml                       <- MkDocs configuration
│   ├── README.md                        <- Docs index for repo
│   ├── .gitkeep
│   │
│   └── docs/                            <- Site documentation pages
│       ├── getting-started.md           <- Quickstart guide
│       └── index.md                     <- Documentation homepage
│    
│
├── notebooks/                           <- Jupyter notebook workspace
│   └── .gitkeep
│
├── references/                          <- Manuals, resources
│   └── .gitkeep
│
├── reports/                             <- Generated reports
│   ├── figures/                         <- Exported figures
│   │   └── .gitkeep
│   └── .gitkeep
│
├── tests/                               <- Test suite
│   ├── test_data.py                     
│
│
├── .gitignore
├── Makefile                             <- Automation commands
├── README.md                            <- Project overview
├── pyproject.toml                       <- Build & formatting config
└── requirements.txt                     <- Python dependencies

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
