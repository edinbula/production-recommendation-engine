from fastapi import FastAPI

from app.recommend import recommend_items


app = FastAPI(
    title="Production Recommendation Engine",
    description="A PyTorch-based recommendation API",
    version="1.0.0"
)


@app.get("/")
def home():
    return {
        "message": "Recommendation API is running"
    }


@app.get("/recommend/{user_id}")
def get_recommendations(user_id: int, top_k: int = 5):
    recommendations = recommend_items(
        user_id=user_id,
        top_k=top_k
    )

    if not recommendations:
        return {
            "user_id": user_id,
            "message": "User not found or no recommendations available",
            "recommendations": []
        }

    return {
        "user_id": user_id,
        "top_k": top_k,
        "recommendations": recommendations
    }