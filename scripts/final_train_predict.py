import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import log_loss, roc_auc_score
from lightgbm.callback import early_stopping, log_evaluation

# --- Load Data ---
print("📦 Loading train/test/main data...")
df = pd.read_parquet("D:/amexxx/data/train_data.parquet")
test_df = pd.read_parquet("D:/amexxx/data/test_data.parquet")

print("📦 Loading event + meta data...")
event = pd.read_parquet("D:/amexxx/data/add_event.parquet")
meta = pd.read_parquet("D:/amexxx/data/offer_metadata.parquet")

# --- Align merge keys ---
for col in ["id3"]:
    df[col] = df[col].astype(str)
    test_df[col] = test_df[col].astype(str)
    meta[col] = meta[col].astype(str)

for col in ["id2", "id3"]:
    event[col] = event[col].astype(str)

# --- Merge metadata ---
print("🔗 Merging offer metadata...")
df = df.merge(meta, on="id3", how="left")
test_df = test_df.merge(meta, on="id3", how="left")

# --- Merge event features (with leakage prevention) ---
print("🔗 Merging event features with leak prevention...")
df["offer_time"] = pd.to_datetime(df["id4"], errors="coerce")  # offer time
event["event_time"] = pd.to_datetime(event["id4"], errors="coerce")

event_with_time = event.merge(
    df[["id2", "id3", "offer_time"]],
    on=["id2", "id3"],
    how="left"
)

event_filtered = event_with_time[
    event_with_time["event_time"] < event_with_time["offer_time"]
].copy()

event_features = event_filtered.groupby(["id2", "id3"]).agg(
    event_count=("event_time", "count"),
    event_min_time=("event_time", "min"),
    event_max_time=("event_time", "max"),
    event_nunique_id6=("id6", "nunique"),
    event_nunique_id7=("id7", "nunique")
).reset_index()

df = df.merge(event_features, on=["id2", "id3"], how="left")
test_df = test_df.merge(event_features, on=["id2", "id3"], how="left")

# --- Load final selected features ---
with open("D:/amexxx/clean_feature.txt", "r") as f:
    final_features = [line.strip() for line in f if line.strip()]

final_features = [f for f in final_features if f in df.columns]

# --- Rebuild validation split (MAP@7 meaningful) ---
offer_counts = df["id1"].value_counts()
valid_users = offer_counts[offer_counts >= 2].index
df_valid = df[df["id1"].isin(valid_users)].copy()

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(splitter.split(df_valid, groups=df_valid["id1"]))

train_df = df_valid.iloc[train_idx]
val_df = df_valid.iloc[val_idx]

val_info = val_df[["id1", "id3"]].copy()
val_info["y"] = val_df["y"].values

X_train = train_df[final_features].apply(pd.to_numeric, errors='coerce')
y_train = train_df["y"].astype(float)

X_val = val_df[final_features].apply(pd.to_numeric, errors='coerce')
y_val = val_df["y"].astype(float)

# --- LightGBM dataset ---
dtrain = lgb.Dataset(X_train, label=y_train)
dval = lgb.Dataset(X_val, label=y_val)

# --- LightGBM parameters ---
params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "verbosity": -1,
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 64,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 3,
    "seed": 42,
}

# --- Train model with callbacks ---
print("🚀 Training LightGBM with callbacks...")
model = lgb.train(
    params,
    dtrain,
    num_boost_round=1000,
    valid_sets=[dval],
    callbacks=[
        early_stopping(stopping_rounds=50, verbose=True),
        log_evaluation(period=100)
    ]
)

# --- Validation metrics ---
y_pred_val = model.predict(X_val)
print("✅ Logloss:", log_loss(y_val, y_pred_val))
print("✅ AUC:", roc_auc_score(y_val, y_pred_val))

val_info["pred"] = y_pred_val
val_info.to_parquet("D:/amexxx/val_data.parquet")
print("📦 Saved val_data.parquet for MAP@7 evaluation")

# --- Predict on test ---
print("🧪 Predicting on test set...")
X_test = test_df[final_features].apply(pd.to_numeric, errors='coerce')
test_df["pred"] = model.predict(X_test, num_iteration=model.best_iteration)

submission = test_df[["id1", "id2", "id3", "id5", "pred"]]
submission.to_csv("D:/amexxx/final_submission.csv", index=False)
print("✅ Submission saved: final_submission.csv")
