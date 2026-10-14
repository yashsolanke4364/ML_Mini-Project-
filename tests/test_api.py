"""Endpoint tests for the MNIST Flask API."""

import base64
import io
import os
import sys

import numpy as np
import pytest
from PIL import Image, ImageDraw

BACKEND_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend"
)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

import predict
import app as backend_app
from app import create_app


class FixedModel:
    def predict(self, _tensor, verbose=0):
        assert verbose == 0
        return np.array([[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.55]])


def image_as_base64(background, stroke):
    image = Image.new("L", (100, 100), color=background)
    draw = ImageDraw.Draw(image)
    draw.line([(50, 15), (50, 85)], fill=stroke, width=12)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return base64.b64encode(output.getvalue()).decode("ascii")


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(predict, "MODEL", FixedModel())
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


def test_health_endpoint_when_model_is_loaded(client):
    response = client.get("/api/health")

    assert response.status_code == 200
    result = response.get_json()
    assert result["status"] == "healthy"
    assert result["model_loaded"] is True
    assert result["model_file"] == "mnist_cnn.keras"
    expected_size = (
        backend_app.predict.MODEL_PATH.stat().st_size
        if backend_app.predict.MODEL_PATH.is_file()
        else None
    )
    assert result["model_size_bytes"] == expected_size


def test_health_endpoint_when_model_is_missing(client, monkeypatch):
    monkeypatch.setattr(predict, "MODEL", None)
    monkeypatch.setattr(predict, "MODEL_LOAD_ERROR", "Model file not found.")

    response = client.get("/api/health")

    assert response.status_code == 503
    assert response.get_json()["model_loaded"] is False
    assert "Model file not found" in response.get_json()["error"]


def test_evaluation_endpoint_reports_missing_notebook_results(client, tmp_path, monkeypatch):
    monkeypatch.setattr(backend_app, "EVALUATION_RESULTS_DIR", tmp_path)

    response = client.get("/api/evaluation")

    assert response.status_code == 404
    assert response.get_json()["available"] is False
    assert "train_model.ipynb" in response.get_json()["message"]


def test_evaluation_endpoint_serves_notebook_results_and_named_artifacts(
    client, tmp_path, monkeypatch
):
    monkeypatch.setattr(backend_app, "EVALUATION_RESULTS_DIR", tmp_path)
    (tmp_path / "evaluation.json").write_text(
        '{"selected_model":"CNN Config 1","comparison":[]}', encoding="utf-8"
    )
    (tmp_path / "training_curves.png").write_bytes(b"test plot")

    result_response = client.get("/api/evaluation")
    artifact_response = client.get("/api/evaluation/artifacts/training_curves.png")
    unknown_response = client.get("/api/evaluation/artifacts/other.txt")

    assert result_response.status_code == 200
    assert result_response.get_json()["available"] is True
    assert result_response.get_json()["selected_model"] == "CNN Config 1"
    assert artifact_response.status_code == 200
    assert artifact_response.data == b"test plot"
    assert unknown_response.status_code == 404


def test_predict_endpoint_returns_digit_confidence_and_all_probabilities(client):
    image = image_as_base64(background=0, stroke=255)
    response = client.post(
        "/api/predict",
        json={"image": f"data:image/png;base64,{image}"},
    )

    assert response.status_code == 200
    result = response.get_json()
    assert result["success"] is True
    assert result["prediction"] == 9
    assert result["confidence"] == pytest.approx(0.55)
    assert result["confidence_percentage"] == 55.0
    assert list(result["probabilities"]) == [str(digit) for digit in range(10)]
    assert len(result["probabilities_list"]) == 10
    assert result["probabilities"]["9"] == pytest.approx(0.55)


def test_predict_returns_service_unavailable_without_trained_model(client, monkeypatch):
    monkeypatch.setattr(predict, "MODEL", None)
    monkeypatch.setattr(predict, "MODEL_LOAD_ERROR", "Add the trained Keras model.")

    response = client.post(
        "/api/predict",
        json={"image": image_as_base64(background=0, stroke=255)},
    )

    assert response.status_code == 503
    assert "trained Keras model" in response.get_json()["message"]


@pytest.mark.parametrize(
    ("background", "stroke"),
    [(0, 255), (255, 0)],
)
def test_preprocessing_normalizes_both_foreground_conventions(background, stroke):
    tensor = predict.preprocess_image(image_as_base64(background, stroke))

    assert tensor.shape == (1, 28, 28, 1)
    assert tensor.dtype == np.float32
    assert 0.0 <= float(tensor.min()) <= float(tensor.max()) <= 1.0
    assert float(tensor.max()) > 0.9
    assert float(tensor[0, 0, 0, 0]) == 0.0


@pytest.mark.parametrize(
    "image",
    [
        "",
        "   ",
        "not valid base64!",
        base64.b64encode(b"not an image").decode("ascii"),
    ],
)
def test_predict_rejects_empty_or_invalid_image_data(client, image):
    response = client.post("/api/predict", json={"image": image})

    assert response.status_code == 400
    assert response.get_json()["error"] == "Bad Request"


def test_predict_rejects_blank_image(client):
    image = Image.new("L", (100, 100), color=0)
    output = io.BytesIO()
    image.save(output, format="PNG")
    encoded = base64.b64encode(output.getvalue()).decode("ascii")

    response = client.post("/api/predict", json={"image": encoded})

    assert response.status_code == 400
    assert "empty" in response.get_json()["message"].lower()


def test_predict_rejects_non_json_request(client):
    response = client.post("/api/predict", data="plain text")

    assert response.status_code == 400


@pytest.mark.parametrize(
    "origin",
    [
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:5501",
        "http://localhost:5501",
    ],
)
def test_local_frontend_origin_is_allowed(client, origin):
    response = client.options(
        "/api/predict",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.status_code == 200
    assert response.headers["Access-Control-Allow-Origin"] == origin


def test_non_local_frontend_origin_is_not_allowed(client):
    response = client.options(
        "/api/predict",
        headers={
            "Origin": "http://example.com:5500",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.status_code == 200
    assert "Access-Control-Allow-Origin" not in response.headers


def test_configured_production_frontend_origins_are_allowed(monkeypatch):
    monkeypatch.setenv(
        "ALLOWED_ORIGINS",
        "https://digits.example,https://preview.example",
    )
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        response = test_client.options(
            "/api/health",
            headers={
                "Origin": "https://digits.example",
                "Access-Control-Request-Method": "GET",
            },
        )

    assert response.headers["Access-Control-Allow-Origin"] == "https://digits.example"
