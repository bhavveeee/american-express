import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
import numpy as np
import os

# File paths
BASE_PATH = "D:/amexxx/data"
TRAIN_PATH = os.path.join(BASE_PATH, "train_data.parquet")
META_PATH = os.path.join(BASE_PATH, "offer_metadata.parquet")
EVENT_PATH = os.path.join(BASE_PATH, "add_event.parquet")
OUTPUT_PATH = os.path.join(BASE_PATH, "feature_logloss_scores.csv")


def load_data():
    print("📥 Loading train data...")
    df = pd.read_parquet(TRAIN_PATH)
    meta = pd.read_parquet(META_PATH)
    df = df.merge(meta, on="id3", how="left")

    df['id2'] = df['id2'].astype(str)

    try:
        print("📥 Loading event data...")
        event_df = pd.read_parquet(EVENT_PATH)
        event_df['id2'] = event_df['id2'].astype(str)
        event_agg = event_df.groupby("id2").size().reset_index(name="event_count")
        df = df.merge(event_agg, on="id2", how="left")
    except Exception as e:
        print(f"⚠️ Failed to load event data: {e}")

    return df


def evaluate_feature(df, feature):
    X = df[[feature]].copy()
    y = df['y']
    groups = df['id2']

    if X[feature].dtype == 'object':
        try:
            X[feature] = X[feature].astype(float)
        except:
            le = LabelEncoder()
            X[feature] = le.fit_transform(X[feature].astype(str))

    kf = GroupKFold(n_splits=5)
    losses = []

    for train_idx, val_idx in kf.split(X, y, groups):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = lgb.LGBMClassifier(
            objective='binary',
            metric='binary_logloss',
            verbosity=-1,
            n_estimators=200,
            learning_rate=0.05
        )

        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(20, verbose=False)])
        y_pred = model.predict_proba(X_val)[:, 1]
        loss = log_loss(y_val, y_pred)
        losses.append(loss)

    return np.mean(losses)


def main():
    df = load_data()
    all_features = [col for col in df.columns if col.startswith('f') or col == 'event_count']

    # Resume support
    if os.path.exists(OUTPUT_PATH):
        existing = pd.read_csv(OUTPUT_PATH)
        done_features = set(existing['feature'])
        results = existing.to_dict('records')
        print(f"🔁 Resuming from existing file with {len(done_features)} features done.")
    else:
        done_features = set()
        results = []

    for f in all_features:
        if f in done_features:
            continue

        try:
            print(f"🔍 Evaluating: {f}")
            score = evaluate_feature(df, f)
            print(f"✅ {f}: avg logloss = {score:.6f}")
            results.append({'feature': f, 'avg_logloss': score})
            pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)
        except Exception as e:
            print(f"❌ Failed on {f}: {e}")

    print("✅ Feature evaluation complete.")


if __name__ == '__main__':
    main()
