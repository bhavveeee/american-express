import pandas as pd

def apk(actual, predicted, k=7):
    if not actual:
        return 0.0
    actual_set = set(actual)
    score = 0.0
    for i, p in enumerate(predicted[:k]):
        if p in actual_set:
            score += 1.0 / (i + 1)
    return score / min(len(actual), k)

def mapk(df, k=7):
    total_score = 0.0
    user_count = 0

    for user_id, group in df.groupby("id1"):
        actual_clicked = group.loc[group["y"] == 1, "id3"].tolist()
        predicted = group.sort_values("pred", ascending=False)["id3"].tolist()

        if len(actual_clicked) == 0:
            continue  # skip users with no clicks (important!)
        
        total_score += apk(actual_clicked, predicted, k)
        user_count += 1

    return total_score / user_count if user_count > 0 else 0.0

# --- Load validation data ---
df = pd.read_parquet("D:/amexxx/val_data.parquet")

# --- Compute MAP@7 ---
score = mapk(df, k=7)

print(f"✅ MAP@7 (realistic): {score:.6f}")
print(f"🧠 Total clicked offers in val set: {df[df['y'] == 1].shape[0]}")
print(f"👥 Unique users with clicks: {df[df['y'] == 1]['id1'].nunique()}")
