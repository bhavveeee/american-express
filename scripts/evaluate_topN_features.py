# save as evaluate_topN_features.py
import pandas as pd
import lightgbm as lgb
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score
from collections import defaultdict

# === MAP@7 function ===
def mapk(actual, predicted, k=7):
    score = 0.0
    for a, p in zip(actual, predicted):
        p = p[:k]
        try:
            score += 1.0 / (p.index(a) + 1)
        except ValueError:
            continue
    return score / len(actual)

# === Config ===
DATA_DIR = "D:/amexxx/data"
TRAIN_PATH = os.path.join(DATA_DIR, "train_with_new_feats_v2.parquet")
FEATURE_FILE = os.path.join(DATA_DIR, "sorted_features_v2.txt")

# === Load data ===
print("🧾 Loading data...")
train = pd.read_parquet(TRAIN_PATH)
all_features = [f.strip() for f in open(FEATURE_FILE).readlines() if f.strip()]

# === Label encode object-type features ===
cat_cols = [f for f in all_features if train[f].dtype == 'object']
print(f"🔠 Label encoding {len(cat_cols)} categorical columns...")
for col in cat_cols:
    train[col], _ = train[col].factorize()

# === Evaluate top-N features ===
results = []
N_values = list(range(20, len(all_features), 20)) + [len(all_features)]  # e.g., ..., 160, 173
print(f"📊 Evaluating top-N feature subsets: {N_values}")

for N in N_values:
    features = all_features[:N]
    X = train[features]
    y = train["y"]

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    logloss_scores, auc_scores, map7_scores = [], [], []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval = lgb.Dataset(X_val, label=y_val)

        model = lgb.train(
            {
                "objective": "binary",
                "learning_rate": 0.03,
                "metric": "binary_logloss",
                "verbosity": -1,
                "seed": 42,
            },
            dtrain,
            num_boost_round=1000,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )

        val_pred = model.predict(X_val)
        logloss_scores.append(log_loss(y_val, val_pred))
        auc_scores.append(roc_auc_score(y_val, val_pred))

        # Prepare for MAP@7
        val_df = X_val.copy()
        val_df["id2"] = train.iloc[val_idx]["id2"].values
        val_df["id3"] = train.iloc[val_idx]["id3"].values
        val_df["y"] = y_val.values
        val_df["pred"] = val_pred

        grouped = val_df.groupby("id2")
        actual = []
        predicted = []
        for uid, group in grouped:
            truth = group[group["y"] == 1]["id3"].tolist()
            preds = group.sort_values("pred", ascending=False)["id3"].tolist()
            if truth:
                actual.append(truth[0])
                predicted.append(preds)
        map7_scores.append(mapk(actual, predicted, k=7))

    result = {
        "top_N": N,
        "logloss_mean": np.mean(logloss_scores),
        "logloss_std": np.std(logloss_scores),
        "auc_mean": np.mean(auc_scores),
        "auc_std": np.std(auc_scores),
        "map7_mean": np.mean(map7_scores),
        "map7_std": np.std(map7_scores)
    }
    print(f"✅ Top {N} | LogLoss={result['logloss_mean']:.5f} | AUC={result['auc_mean']:.5f} | MAP@7={result['map7_mean']:.5f}")
    results.append(result)

# === Save results ===
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(DATA_DIR, "topN_feature_results.csv"), index=False)
print("📁 Saved top-N evaluation results to topN_feature_results.csv")
