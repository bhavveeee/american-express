import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import log_loss
from collections import defaultdict
from tqdm import tqdm

DATA_DIR = Path("D:/amexxx/data")
NFOLDS = 5
RANDOM_STATE = 42

# Load data
print("📦 Loading data...")
df = pd.read_parquet(DATA_DIR / "train_data2.parquet")
with open(DATA_DIR / "clean_feature.txt") as f:
    base_features = f.read().splitlines()

target = "y"
trans_cols = [col for col in df.columns if col.startswith("trans_")]
trans_cols = [col for col in trans_cols if col not in base_features]

print(f"📌 Starting with {len(base_features)} base features.")
print(f"🧪 Testing {len(trans_cols)} transaction features.")

# ✅ Convert features to numeric to avoid LightGBM dtype errors
all_feats = base_features + trans_cols
df[all_feats] = df[all_feats].apply(pd.to_numeric, errors="coerce")

results = []
used_features = base_features.copy()

def map_at_7(y_true, y_pred, ids):
    df_pred = pd.DataFrame({"id2": ids, "y_true": y_true, "y_pred": y_pred})
    df_pred["rank"] = df_pred.groupby("id2")["y_pred"].rank("dense", ascending=False)

    df_pred = df_pred[df_pred["rank"] <= 7]
    df_pred["relevant"] = df_pred["y_true"]

    map_scores = df_pred.groupby("id2")["relevant"].apply(lambda x: (x.astype(int) / (np.arange(len(x)) + 1)).sum())

    return map_scores.mean()

# Cross-validation
cv = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=RANDOM_STATE)

for feat in tqdm(trans_cols):
    all_logloss = []
    all_map = []

    for train_idx, val_idx in cv.split(df, df[target]):
        X_train = df.iloc[train_idx][used_features + [feat]]
        y_train = df.iloc[train_idx][target]
        X_val = df.iloc[val_idx][used_features + [feat]]
        y_val = df.iloc[val_idx][target]
        id2_val = df.iloc[val_idx]["id2"]
        id3_val = df.iloc[val_idx]["id3"]

        model = LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.01,
            random_state=RANDOM_STATE,
            verbosity=-1,
            n_jobs=-1
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="binary_logloss",
            callbacks=[
                early_stopping(stopping_rounds=50),
                log_evaluation(0)
            ]
        )

        y_pred = model.predict_proba(X_val)[:, 1]
        ll = log_loss(y_val, y_pred)
        map7 = map_at_7(y_val.values, y_pred, id2_val.values)

        all_logloss.append(ll)
        all_map.append(map7)

    avg_ll = np.mean(all_logloss)
    avg_map = np.mean(all_map)

    results.append({
        "feature": feat,
        "logloss": avg_ll,
        "map@7": avg_map
    })

    print(f"✅ Tried {feat:<30} | LogLoss: {avg_ll:.6f} | MAP@7: {avg_map:.6f}")

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv(DATA_DIR / "greedy_trans_results.csv", index=False)
print("📁 Saved results to greedy_trans_results.csv")
