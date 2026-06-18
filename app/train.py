import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

from app.model import MatrixFactorizationModel


DATA_PATH = "data/interactions.csv"
MODEL_PATH = "models/recommender.pt"


class InteractionDataset(Dataset):
    def __init__(self, dataframe, user_map, item_map):
        self.users = dataframe["user_id"].map(user_map).values
        self.items = dataframe["item_id"].map(item_map).values
        self.ratings = dataframe["rating"].values

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, index):
        return (
            torch.tensor(self.users[index], dtype=torch.long),
            torch.tensor(self.items[index], dtype=torch.long),
            torch.tensor(self.ratings[index], dtype=torch.float)
        )


def train_model():
    df = pd.read_csv(DATA_PATH)

    unique_users = sorted(df["user_id"].unique())
    unique_items = sorted(df["item_id"].unique())

    user_map = {
        user_id: index
        for index, user_id in enumerate(unique_users)
    }

    item_map = {
        item_id: index
        for index, item_id in enumerate(unique_items)
    }

    dataset = InteractionDataset(
        df,
        user_map,
        item_map
    )

    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True
    )

    model = MatrixFactorizationModel(
        num_users=len(unique_users),
        num_items=len(unique_items)
    )

    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01
    )

    for epoch in range(20):
        total_loss = 0

        for user_ids, item_ids, ratings in dataloader:
            optimizer.zero_grad()

            predictions = model(user_ids, item_ids)

            loss = loss_function(
                predictions,
                ratings
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

    os.makedirs("models", exist_ok=True)

    torch.save(
        {
            "model_state": model.state_dict(),
            "user_map": user_map,
            "item_map": item_map,
            "unique_users": unique_users,
            "unique_items": unique_items
        },
        MODEL_PATH
    )

    print("Model saved to models/recommender.pt")


if __name__ == "__main__":
    train_model()