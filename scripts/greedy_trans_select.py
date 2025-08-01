import pandas as pd
import lightgbm as lgb
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from utils import map7  # Ensure utils.py is in the same directory or Python path
import time
import warnings
warnings.filterwarnings("ignore")

start_time = time.time()

# --- Load data ---
df = pd.read_parquet("D:/amexxx/data/train_data2.parquet")
print("✅ Loaded", df.shape)

# Fix object dtype features
for col in df.columns:
    if df[col].dtype == "object":
        try:
            df[col] = df[col].astype(float)
        except:
            df[col] = df[col].astype("category")

# --- Feature setup ---
base_all = pd.read_csv("D:/amexxx/data/clean_feature.txt", header=None)[0].tolist()
base_features = [f for f in base_all if f in df.columns]

# Load transaction features
trans_df = pd.read_parquet("D:/amexxx/data/trans_features.parquet")
trans_features_all = [col for col in trans_df.columns if col != "id2"]
available_trans_features = [f for f in trans_features_all if f in df.columns]
print("✅ Transaction features found in dataset:", available_trans_features)

# --- Prepare features and labels ---
X = df[base_features + ['id1', 'id2', 'id3', 'id5'] + available_trans_features]
y = df['y']

# --- CV Setup ---
cv = GroupKFold(n_splits=5)

# --- Greedy selection ---
selected = []
best_map = 0

print("\n🚀 Starting greedy transaction feature selection...\n")
for feat in available_trans_features:
    current_feats = base_features + selected + [feat]

    oof_preds = []
    oof_true = []
    oof_id1 = []
    oof_id2 = []
    oof_id3 = []
    oof_id5 = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups=X['id3'])):
        X_train = X.iloc[train_idx][current_feats]
        X_val = X.iloc[val_idx][current_feats]
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_val = lgb.Dataset(X_val, y_val)

        model = lgb.train(
            params={
                "objective": "binary",
                "metric": "binary_logloss",
                "verbosity": -1,
                "learning_rate": 0.01,
                "num_leaves": 64,
            },
            train_set=lgb_train,
            valid_sets=[lgb_val],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )

        pred = model.predict(X_val, num_iteration=model.best_iteration)
        oof_preds.extend(pred)
        oof_true.extend(y_val)
        oof_id1.extend(X.iloc[val_idx]['id1'])
        oof_id2.extend(X.iloc[val_idx]['id2'])
        oof_id3.extend(X.iloc[val_idx]['id3'])
        oof_id5.extend(X.iloc[val_idx]['id5'])

    df_val = pd.DataFrame({
        "id1": oof_id1,
        "id2": oof_id2,
        "id3": oof_id3,
        "id5": oof_id5,
        "pred": oof_preds
    })

    score = map7(df_val)
    logloss = log_loss(oof_true, oof_preds)
    auc = roc_auc_score(oof_true, oof_preds)

    print(f"🧪 Tried: {feat:<20} | MAP@7: {score:.6f} | Logloss: {logloss:.6f} | AUC: {auc:.6f}")

    if score > best_map:
        selected.append(feat)
        best_map = score
        print(f"✅ Added: {feat} → New Best MAP@7: {best_map:.6f}")

print("\n🎯 Final selected transaction features:")
print(selected)
print(f"⏱️ Total time: {time.time() - start_time:.1f} seconds")
