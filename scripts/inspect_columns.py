import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import KFold
import os

# Paths
DATA_DIR = "D:/amexxx/data"
TRAIN_PATH = os.path.join(DATA_DIR, "train_data.parquet")
TEST_PATH = os.path.join(DATA_DIR, "test_data.parquet")
META_PATH = os.path.join(DATA_DIR, "offer_metadata.parquet")
EVENT_PATH = os.path.join(DATA_DIR, "add_event.parquet")


# Load and prepare data
def load_data():
    print("\U0001F4E5 Loading train:", TRAIN_PATH)
    train_df = pd.read_parquet(TRAIN_PATH)
    print("\U0001F4E5 Loading test:", TEST_PATH)
    test_df = pd.read_parquet(TEST_PATH)
    print("\U0001F4E5 Loading metadata:", META_PATH)
    offer_meta = pd.read_parquet(META_PATH)

    # Ensure id3 is string for merge compatibility
    train_df["id3"] = train_df["id3"].astype(str)
    test_df["id3"] = test_df["id3"].astype(str)
    offer_meta["id3"] = offer_meta["id3"].astype(str)

    # Merge metadata
    train_df = train_df.merge(offer_meta, on="id3", how="left")
    test_df = test_df.merge(offer_meta, on="id3", how="left")

    # Attempt to load and process event data
    try:
        print("\U0001F4C5 Loading event data:", EVENT_PATH)
        event_df = pd.read_parquet(EVENT_PATH)
        if "id2" in event_df.columns:
            event_df["id2"] = event_df["id2"].astype(train_df["id2"].dtype)
            event_feats = event_df.groupby("id2").agg({"id6": "nunique"}).reset_index()
            event_feats.rename(columns={"id6": "n_unique_tiles"}, inplace=True)
            train_df = train_df.merge(event_feats, on="id2", how="left")
            test_df = test_df.merge(event_feats, on="id2", how="left")
        else:
            print("⚠️ 'event_type' or 'id2' column missing in add_event.parquet – skipping event features.")
    except Exception as e:
        print("⚠️ Failed to load event data:", str(e))

    # Convert feature columns to float
    feature_cols = [col for col in train_df.columns if col.startswith("f")]
    for col in feature_cols:
        train_df[col] = pd.to_numeric(train_df[col], errors="coerce")
        test_df[col] = pd.to_numeric(test_df[col], errors="coerce")

    # Fill NaNs
    train_df = train_df.fillna(-1)
    test_df = test_df.fillna(-1)

    features = feature_cols + [col for col in offer_meta.columns if col != "id3"]
    if "n_unique_tiles" in train_df.columns:
        features.append("n_unique_tiles")

    print(f"✅ Using {len(features)} features")
    return train_df, test_df, features


# Training function
def train_model():
    train_df, test_df, features = load_data()
    target = "y"

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    test_preds = np.zeros(len(test_df))

    for fold, (tr_idx, val_idx) in enumerate(kf.split(train_df)):
        print(f"\n🔁 Fold {fold+1}")
        tr_data = train_df.iloc[tr_idx]
        val_data = train_df.iloc[val_idx]

        train_set = lgb.Dataset(tr_data[features], label=tr_data[target])
        val_set = lgb.Dataset(val_data[features], label=val_data[target])

        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
            "num_leaves": 64,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "seed": 42,
        }

        model = lgb.train(
            params,
            train_set,
            num_boost_round=1000,
            valid_sets=[train_set, val_set],
            valid_names=["train", "valid"],
            callbacks=[
                lgb.early_stopping(50),
                lgb.log_evaluation(100)
            ]
        )

        test_preds += model.predict(test_df[features], num_iteration=model.best_iteration) / kf.n_splits

    # Save submission
    submission = test_df[["id1", "id2", "id3", "id5"]].copy()
    submission["pred"] = test_preds
    submission.to_csv("submission.csv", index=False)
    print("\n✅ Saved submission to submission.csv")


if __name__ == "__main__":
    train_model()
