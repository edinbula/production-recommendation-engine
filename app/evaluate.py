import pandas as pd

from app.recommend import recommend_items


DATA_PATH = "data/interactions.csv"


def precision_at_k(recommended_items, relevant_items, k):
    recommended_k = recommended_items[:k]

    if not recommended_k:
        return 0.0

    hits = len(
        set(recommended_k) & set(relevant_items)
    )

    return hits / k


def recall_at_k(recommended_items, relevant_items, k):
    recommended_k = recommended_items[:k]

    if not relevant_items:
        return 0.0

    hits = len(
        set(recommended_k) & set(relevant_items)
    )

    return hits / len(relevant_items)


def hit_rate_at_k(recommended_items, relevant_items, k):
    recommended_k = recommended_items[:k]

    hits = len(
        set(recommended_k) & set(relevant_items)
    )

    return 1.0 if hits > 0 else 0.0


def evaluate(k=5):
    df = pd.read_csv(DATA_PATH)

    users = sorted(df["user_id"].unique())

    precision_scores = []
    recall_scores = []
    hit_rate_scores = []

    for user_id in users:
        user_items = df[df["user_id"] == user_id]["item_id"].tolist()

        if len(user_items) < 2:
            continue

        relevant_items = [user_items[-1]]

        recommendations = recommend_items(
            user_id=user_id,
            top_k=k
        )

        recommended_items = [
            item["item_id"]
            for item in recommendations
        ]

        precision_scores.append(
            precision_at_k(
                recommended_items,
                relevant_items,
                k
            )
        )

        recall_scores.append(
            recall_at_k(
                recommended_items,
                relevant_items,
                k
            )
        )

        hit_rate_scores.append(
            hit_rate_at_k(
                recommended_items,
                relevant_items,
                k
            )
        )

    metrics = {
        "precision_at_k": round(sum(precision_scores) / len(precision_scores), 4),
        "recall_at_k": round(sum(recall_scores) / len(recall_scores), 4),
        "hit_rate_at_k": round(sum(hit_rate_scores) / len(hit_rate_scores), 4),
    }

    return metrics


if __name__ == "__main__":
    results = evaluate(k=5)

    print("Evaluation Metrics:")
    print(results)