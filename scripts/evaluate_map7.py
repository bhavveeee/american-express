import pandas as pd

def apk(actual, predicted, k=7):
    """
    Computes the average precision at k.
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / min(len(actual), k)

def mapk(df, k=7):
    """
    Computes the mean average precision at k.
    Skips users with no actual clicks.
    """
    map7_total = 0
    valid_users = 0
    group = df.groupby("id1")
    for _, rows in group:
        rows_sorted = rows.sort_values("pred", ascending=False)
        predicted = rows_sorted["id3"].tolist()
        actual = rows_sorted[rows_sorted["y"] == 1]["id3"].tolist()
        if not actual:
            continue  # Skip if no clicked offers
        map7_total += apk(actual, predicted, k)
        valid_users += 1
    return map7_total / valid_users if valid_users > 0 else 0


# --- Load validation data ---
val = pd.read_parquet("D:/amexxx/val_data.parquet")

# --- Compute MAP@7 ---
score = mapk(val, k=7)
print(f"✅ MAP@7 score: {score:.6f}")


print("🧠 Total clicked offers in val set:", val[val["y"] == 1].shape[0])
print("👥 Unique users with clicks:", val[val["y"] == 1]["id1"].nunique())

