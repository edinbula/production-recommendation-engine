# Production Recommendation Engine

A PyTorch-based recommendation engine with a FastAPI backend.

This project demonstrates how a recommendation system can be trained, saved, loaded, and served through a REST API. It uses a small sample user-item interaction dataset and a matrix factorization model built with PyTorch.

## Features

- User-item interaction dataset
- PyTorch matrix factorization model
- Training pipeline
- Saved model checkpoint
- Top-N recommendations
- Seen-item filtering
- FastAPI REST API
- Swagger documentation
- Unit tests

## Tech Stack

- Python
- PyTorch
- Pandas
- NumPy
- FastAPI
- Uvicorn
- Pytest

## Run Training

```bash
python -m app.train