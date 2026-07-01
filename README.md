# Production Recommendation Engine

A production-style recommendation engine built with **Python, PyTorch, FastAPI, Pandas, and NumPy**.

This project demonstrates how a recommendation system can be trained, saved, loaded, and served through a REST API. It includes a complete workflow from user-item interaction data to model training, inference, top-N recommendation generation, and API delivery.

---

## Overview

The goal of this project is to demonstrate a clean, backend-ready recommendation system architecture.

The system uses a matrix factorization approach built with PyTorch to learn user and item embeddings from interaction data. After training, the model can generate personalized recommendations for users while filtering out items they have already interacted with.

This project is designed as a portfolio-ready example of how machine learning models can be integrated into a production API environment.

---

## Key Features

* User-item interaction dataset
* PyTorch-based matrix factorization model
* User and item embeddings
* Training pipeline
* Model checkpoint saving and loading
* Top-N personalized recommendations
* Seen-item filtering
* FastAPI REST API
* Swagger/OpenAPI documentation
* Modular project structure
* Unit tests with Pytest
* Clean separation between training, model logic, and serving layer

---

## Tech Stack

* Python
* PyTorch
* Pandas
* NumPy
* FastAPI
* Uvicorn
* Pytest

---

## Project Architecture

```text
production-recommendation-engine/
│
├── app/
│   ├── api.py              # FastAPI application
│   ├── data.py             # Dataset loading and preprocessing
│   ├── model.py            # PyTorch recommendation model
│   ├── recommend.py        # Recommendation generation logic
│   └── train.py            # Training pipeline
│
├── data/
│   └── interactions.csv    # Sample user-item interaction data
│
├── models/
│   └── recommender.pt      # Saved model checkpoint
│
├── tests/
│   └── test_recommend.py   # Unit tests
│
├── requirements.txt
└── README.md
```

---

## How It Works

### 1. Interaction Data

The system starts with user-item interaction data.

Each interaction represents a signal that a user engaged with an item.

Example:

```text
user_id,item_id,rating
1,101,1
1,105,1
2,103,1
```

---

### 2. Model Training

The recommendation model learns user and item embeddings using PyTorch.

During training, the model learns which users are likely to prefer which items based on historical interaction patterns.

---

### 3. Model Saving

After training, the model is saved as a checkpoint so it can be loaded later for inference.

---

### 4. Recommendation Generation

For a given user, the system scores candidate items and returns the top-N highest-ranked recommendations.

Already-seen items are filtered out so users receive new recommendations.

---

### 5. API Serving

FastAPI exposes the recommendation system through a REST API.

The API can be used by frontend applications, backend services, or other systems that need personalized recommendations.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/production-recommendation-engine.git
cd production-recommendation-engine
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment.

### Windows

```bash
.\.venv\Scripts\activate
```

### macOS / Linux

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Train the Model

Run the training pipeline:

```bash
python -m app.train
```

This will train the PyTorch recommendation model and save the checkpoint.

---

## Start the API

Run the FastAPI server:

```bash
uvicorn app.api:app --reload
```

The API will be available at:

```text
http://localhost:8000
```

Swagger documentation:

```text
http://localhost:8000/docs
```

---

## Example API Usage

### Get Recommendations

```http
GET /recommend/{user_id}
```

Example:

```http
GET /recommend/1
```

Example response:

```json
{
  "user_id": 1,
  "recommendations": [104, 109, 112, 118, 121]
}
```

---

## Testing

Run tests with:

```bash
pytest
```

---

## Example Use Cases

This project can be adapted for:

* Social media feed recommendations
* E-commerce product recommendations
* Movie or music recommendations
* Learning platform course recommendations
* Content discovery systems
* Personalized ranking systems

---

## What This Project Demonstrates

This project demonstrates practical skills in:

* Recommendation systems
* Machine learning model training
* PyTorch development
* Backend API development
* Model serving
* REST API design
* Python project structure
* Testing and modular software engineering

---

## Future Improvements

Planned improvements include:

* Negative sampling
* Hybrid recommendation logic
* Content-based features
* Evaluation metrics such as Precision@K and Recall@K
* Batch recommendation generation
* Database integration
* Docker support
* Authentication for API access
* Monitoring and logging
* Deployment to cloud infrastructure

---

## Author

Edin Bula
AI/ML & Backend Engineer
Prishtina, Kosovo

---

## License

This project is intended for educational and portfolio purposes.
