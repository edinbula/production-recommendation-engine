from app.recommend import recommend_items, get_seen_items


def test_recommendations_return_items():
    recommendations = recommend_items(user_id=1, top_k=5)

    assert len(recommendations) > 0
    assert "item_id" in recommendations[0]
    assert "score" in recommendations[0]


def test_seen_items_are_excluded():
    user_id = 1
    seen_items = get_seen_items(user_id)

    recommendations = recommend_items(user_id=user_id, top_k=5)

    recommended_item_ids = {
        item["item_id"]
        for item in recommendations
    }

    assert recommended_item_ids.isdisjoint(seen_items)