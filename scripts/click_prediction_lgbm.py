import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os

# File paths
BASE_PATH = "D:/amexxx/data"
TRAIN_PATH = os.path.join(BASE_PATH, "train_data.parquet")
TEST_PATH = os.path.join(BASE_PATH, "test_data.parquet")
META_PATH = os.path.join(BASE_PATH, "offer_metadata.parquet")
EVENT_PATH = os.path.join(BASE_PATH, "add_event.parquet")
OUTPUT_PATH = os.path.join(BASE_PATH, "submission.csv")

def load_data():
    print(f"📅 Loading train: {TRAIN_PATH}")
    train_df = pd.read_parquet(TRAIN_PATH)
    print(f"📅 Loading test: {TEST_PATH}")
    test_df = pd.read_parquet(TEST_PATH)

    print(f"📅 Loading metadata: {META_PATH}")
    offer_meta = pd.read_parquet(META_PATH)

    # Ensure same type for merging
    train_df['id3'] = train_df['id3'].astype(offer_meta['id3'].dtype)
    test_df['id3'] = test_df['id3'].astype(offer_meta['id3'].dtype)

    train_df = train_df.merge(offer_meta, on="id3", how="left")
    test_df = test_df.merge(offer_meta, on="id3", how="left")

    # Convert id2 to string for merging
    train_df['id2'] = train_df['id2'].astype(str)
    test_df['id2'] = test_df['id2'].astype(str)

    try:
        print(f"📅 Loading event data: {EVENT_PATH}")
        event_df = pd.read_parquet(EVENT_PATH)
        event_df['id2'] = event_df['id2'].astype(str)
        event_agg = event_df.groupby("id2").size().reset_index(name="event_count")
        train_df = train_df.merge(event_agg, on="id2", how="left")
        test_df = test_df.merge(event_agg, on="id2", how="left")
        print("✅ Merged event features.")
    except Exception as e:
        print(f"⚠️ Failed to load event data: {e}")

    # Define features
    features = [col for col in train_df.columns if col.startswith("f") or col == "event_count"]

    # Convert object features to float or label encode
    for col in features:
        if train_df[col].dtype == 'object':
            try:
                train_df[col] = train_df[col].astype(float)
                test_df[col] = test_df[col].astype(float)
            except:
                le = LabelEncoder()
                train_df[col] = le.fit_transform(train_df[col].astype(str))
                test_df[col] = le.transform(test_df[col].astype(str))
                print(f"🧠 Encoded object column: {col}")

    print(f"✅ Using {len(features)} features")
    return train_df, test_df, features

def train_model():
    train_df, test_df, features = load_data()

    X = train_df[features]
    y = train_df["y"]
    groups = train_df["id2"]
    test_X = test_df[features]

    preds = np.zeros(len(test_df))
    gkf = GroupKFold(n_splits=5)

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        print(f"\n🔁 Fold {fold+1}")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = lgb.LGBMClassifier(
            objective="binary",
            learning_rate=0.05,
            n_estimators=1000,
            num_leaves=64,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            verbosity=-1
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="binary_logloss",
            callbacks=[
                lgb.early_stopping(50),
                lgb.log_evaluation(100)
            ]
        )

        preds += model.predict_proba(test_X, num_iteration=model.best_iteration_)[:, 1] / gkf.n_splits

    test_df["pred"] = preds
    test_df[["id1", "id2", "id3", "id5", "pred"]].to_csv(OUTPUT_PATH, index=False)
    print(f"🚀 Submission saved to: {OUTPUT_PATH}")

if __name__ == '__main__':
    train_model()
