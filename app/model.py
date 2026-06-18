import torch
import torch.nn as nn


class MatrixFactorizationModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32):
        super().__init__()

        self.user_embedding = nn.Embedding(
            num_users,
            embedding_dim
        )

        self.item_embedding = nn.Embedding(
            num_items,
            embedding_dim
        )

    def forward(self, user_ids, item_ids):
        user_vectors = self.user_embedding(user_ids)
        item_vectors = self.item_embedding(item_ids)

        scores = (user_vectors * item_vectors).sum(dim=1)

        return scores