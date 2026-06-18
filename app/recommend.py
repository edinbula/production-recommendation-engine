import pandas as pd
import torch

from app.model import MatrixFactorizationModel


MODEL_PATH = "models/recommender.pt"
DATA_PATH = "data/interactions.csv"


def get_seen_items(user_id):
    df = pd.read_csv(DATA_PATH)

    seen_items = df[df["user_id"] == user_id]["item_id"].tolist()

    return set(seen_items)


def recommend_items(user_id, top_k=5):
    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False
    )

    user_map = checkpoint["user_map"]
    item_map = checkpoint["item_map"]
    unique_items = checkpoint["unique_items"]

    if user_id not in user_map:
        return []

    seen_items = get_seen_items(user_id)

    reverse_item_map = {
        index: item_id
        for item_id, index in item_map.items()
    }

    model = MatrixFactorizationModel(
        num_users=len(user_map),
        num_items=len(item_map)
    )

    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    internal_user_id = user_map[user_id]

    recommendations = []

    with torch.no_grad():
        for internal_item_id in range(len(unique_items)):
            original_item_id = int(reverse_item_map[internal_item_id])

            if original_item_id in seen_items:
                continue

            user_tensor = torch.tensor(
                [internal_user_id],
                dtype=torch.long
            )

            item_tensor = torch.tensor(
                [internal_item_id],
                dtype=torch.long
            )

            score = model(user_tensor, item_tensor)
            probability = torch.sigmoid(score).item()

            recommendations.append(
                {
                    "item_id": original_item_id,
                    "score": round(probability, 4)
                }
            )

    recommendations = sorted(
        recommendations,
        key=lambda x: x["score"],
        reverse=True
    )

    return recommendations[:top_k]


if __name__ == "__main__":
    user_id = int(input("Enter user_id: "))

    recommendations = recommend_items(
        user_id,
        top_k=5
    )

    print("Recommendations:")
    print(recommendations)