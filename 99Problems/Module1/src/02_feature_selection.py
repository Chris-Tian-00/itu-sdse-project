from Module1.config import config as cfg
from Module1.src.utils import load_csv_to_df, save_to_csv

data = load_csv_to_df(cfg.data_load_path)

# Not all columns are relevant for modelling
data = data.drop(
    [
        "is_active", "marketing_consent", "first_booking", "existing_customer", "last_seen"
    ],
    axis=1
)

# Removing columns that will be added back after the EDA
data = data.drop(
    ["domain", "country", "visited_learn_more_before_booking", "visited_faq"],
    axis=1
)

save_to_csv(data, cfg.data_feat_select_path)
