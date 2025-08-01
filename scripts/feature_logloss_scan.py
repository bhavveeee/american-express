import pandas as pd
import lightgbm as lgb
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import warnings

warnings.filterwarnings("ignore")

FEATURE_RESULT_PATH = "feature_logloss_results.csv"

def load_data():
    print("📥 Loading train data...")
    df = pd.read_parquet("../data/train_data.parquet")
    meta = pd.read_parquet("../data/offer_metadata.parquet")
    event = pd.read_parquet("../data/add_event.parquet")

    print("🔗 Merging offer metadata...")
    df["id3"] = df["id3"].astype(meta["id3"].dtype)
    df = df.merge(meta, on="id3", how="left")

    print("📅 Merging event data...")
    try:
        # 🚨 Ensure matching dtypes for id2
        df["id2"] = df["id2"].astype(str)
        event["id2"] = event["id2"].astype(str)

        df = df.merge(
            event.groupby("id2").agg(event_count=("id4", "count")).reset_index(),
            on="id2",
            how="left",
        )
    except Exception as e:
        print(f"⚠️ Failed to merge event data: {e}")

    feature_cols = [col for col in df.columns if col.startswith("f")]
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    df[feature_cols] = df[feature_cols].fillna(0)

    df = df.fillna(0)

    # 💥 Ensure y is numeric
    df["y"] = pd.to_numeric(df["y"], errors="coerce").fillna(0).astype("int8")

    return df

def evaluate_feature(df, feature):
    try:
        X = df[[feature]]
        y = df["y"]

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "learning_rate": 0.1,
            "num_leaves": 15,
            "seed": 42,
        }

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=100,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(stopping_rounds=10)],
        )

        y_pred = model.predict(X_val, num_iteration=model.best_iteration)
        loss = log_loss(y_val, y_pred)

        print(f"✅ Feature {feature}: Log Loss = {loss:.5f}")
        return loss

    except Exception as e:
        print(f"❌ Failed on {feature}: {e}")
        return None

def main():
    df = load_data()
    all_features = [col for col in df.columns if col.startswith("f")]

    if os.path.exists(FEATURE_RESULT_PATH):
        results_df = pd.read_csv(FEATURE_RESULT_PATH)
        done_features = set(results_df["feature"])
    else:
        results_df = pd.DataFrame(columns=["feature", "logloss"])
        done_features = set()

    for feature in all_features:
        if feature in done_features:
            continue

        print(f"🔍 Evaluating feature: {feature}")
        loss = evaluate_feature(df, feature)

        if loss is not None:
            results_df = pd.concat(
                [results_df, pd.DataFrame([[feature, loss]], columns=["feature", "logloss"])]
            )
            results_df.to_csv(FEATURE_RESULT_PATH, index=False)

if __name__ == "__main__":
    main()
