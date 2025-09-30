import pytest
from fastapi.testclient import TestClient
from api.main import app
from ml.loader import get_models

@pytest.fixture(scope="session", autouse=True)
def setup_models():
    vectorizer, classifier = get_models()
    app.state.vectorizer = vectorizer
    app.state.classifier = classifier
    yield


client = TestClient(app)

def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200

    data = response.json()

    # Required keys should exist
    assert "status" in data
    assert "uptime_seconds" in data
    assert "model_loaded" in data
    assert "vectorizer_loaded" in data

    # Status must be either ok or degraded
    assert data["status"] in ["ok", "degraded"]

    # Uptime should be non-negative
    assert isinstance(data["uptime_seconds"], (int, float))
    assert data["uptime_seconds"] >= 0

    # Model/vectorizer flags must be boolean
    assert isinstance(data["model_loaded"], bool)
    assert isinstance(data["vectorizer_loaded"], bool)


def test_single_classify():
    response = client.post("/api/classify", json={"text": "Send an email"})
    assert response.status_code == 200
    assert "intent" in response.json()
    assert "confidence" in response.json()

def test_batch_classify():
    response = client.post("/api/classify/batch", json={"texts": ["book meeting", "search google"]})
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert "intent" in data[0]
    assert "confidence" in data[0]

# Test unauthorized access
def test_model_info_unauthorized():
    response = client.get("/api/model/info")  # no creds
    assert response.status_code == 401
    # assert response.json()["detail"] == "Invalid credentials"
    assert response.json()["detail"] == "Not authenticated"
    
# Test wrong credentials
def test_model_info_wrong_credentials():
    response = client.get(
        "/api/model/info",
        auth=("wrong_user", "wrong_pass")
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid credentials"

# Test correct credentials
def test_model_info_authorized():
    response = client.get(
        "/api/model/info",
        auth=("admin", "admin@786")
    )
    assert response.status_code == 200
    assert "model_type" in response.json()
    assert "vectorizer_type" in response.json()
    assert response.json()["status"] == "loaded"