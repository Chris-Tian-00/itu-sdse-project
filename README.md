# ITU-sdse-project

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Exam_Project

## 👥 Contributors

| Name               | Email        |
|--------------------|--------------|
| **Christian Ainis** | chai@itu.dk  |
| **Lorand Ladnai**   | lorl@itu.dk  |
| **Wiktor Pedrycz**  | wipe@itu.dk  |
| **Matei Pop**       | mapop@itu.dk |




## Project Organization

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

--------
