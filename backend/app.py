"""Flask API for handwritten MNIST digit predictions."""

import json
import logging
import os
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

if __package__:
    from . import predict
else:
    import predict

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
EVALUATION_RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
EVALUATION_ARTIFACTS = {
    "training_curves.png",
    "confusion_matrix.png",
    "predictions_analysis.png",
}


def _allowed_origins():
    configured_origins = os.environ.get("ALLOWED_ORIGINS")
    if configured_origins is None:
        return [r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$"]
    return [origin.strip() for origin in configured_origins.split(",") if origin.strip()]


def create_app():
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
    origins = _allowed_origins()
    CORS(app, resources={r"/api/*": {"origins": origins}})
    logger.info("CORS enabled for origins: %s", origins)

    @app.get("/")
    def index():
        return jsonify(
            {
                "service": "MNIST Handwritten Digit Recognition API",
                "endpoints": {
                    "health": "/api/health",
                    "evaluation": "/api/evaluation",
                    "predict": "/api/predict",
                },
            }
        )

    @app.get("/api/health")
    def health():
        loaded = predict.MODEL is not None
        result = {
            "status": "healthy" if loaded else "unavailable",
            "model_loaded": loaded,
            "model_path": str(predict.MODEL_PATH),
        }
        if not loaded:
            result["error"] = predict.MODEL_LOAD_ERROR or "The model is not loaded."
            return jsonify(result), 503
        return jsonify(result)

    @app.get("/api/evaluation")
    def evaluation():
        evaluation_path = EVALUATION_RESULTS_DIR / "evaluation.json"
        if not evaluation_path.is_file():
            return jsonify(
                {
                    "available": False,
                    "message": (
                        "Evaluation results are not available yet. Run all cells in "
                        "ml/train_model.ipynb to generate them."
                    ),
                }
            ), 404

        try:
            with evaluation_path.open(encoding="utf-8") as result_file:
                result = json.load(result_file)
        except (OSError, json.JSONDecodeError):
            logger.exception("Could not read evaluation results from %s.", evaluation_path)
            return jsonify(
                {
                    "available": False,
                    "message": "Evaluation results exist but could not be read.",
                }
            ), 500

        if not isinstance(result, dict):
            logger.error("Evaluation results must contain a JSON object.")
            return jsonify(
                {
                    "available": False,
                    "message": "Evaluation results have an invalid format.",
                }
            ), 500
        return jsonify({"available": True, **result})

    @app.get("/api/evaluation/artifacts/<path:filename>")
    def evaluation_artifact(filename):
        if filename not in EVALUATION_ARTIFACTS:
            return jsonify(
                {"error": "Not Found", "message": "The evaluation artifact does not exist."}
            ), 404

        return send_from_directory(EVALUATION_RESULTS_DIR, filename)

    @app.post("/api/predict")
    def predict_endpoint():
        if not request.is_json:
            return jsonify(
                {
                    "error": "Bad Request",
                    "message": "Request must have Content-Type: application/json.",
                }
            ), 400

        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify(
                {"error": "Bad Request", "message": "Request body must be a JSON object."}
            ), 400

        image_data = data.get("image")
        if not isinstance(image_data, str) or not image_data.strip():
            return jsonify(
                {
                    "error": "Bad Request",
                    "message": "The 'image' field must contain a non-empty base64 image.",
                }
            ), 400

        try:
            result = predict.predict_digit(image_data)
            return jsonify({"success": True, **result})
        except ValueError as exc:
            logger.info("Rejected invalid prediction image: %s", exc)
            return jsonify({"error": "Bad Request", "message": str(exc)}), 400
        except RuntimeError as exc:
            logger.error("Prediction service unavailable: %s", exc)
            return jsonify({"error": "Service Unavailable", "message": str(exc)}), 503
        except Exception:
            logger.exception("Unexpected error while predicting a digit.")
            return jsonify(
                {
                    "error": "Internal Server Error",
                    "message": "An unexpected error occurred while processing the image.",
                }
            ), 500

    @app.errorhandler(404)
    def not_found(_error):
        return jsonify(
            {"error": "Not Found", "message": "The requested endpoint does not exist."}
        ), 404

    @app.errorhandler(405)
    def method_not_allowed(_error):
        return jsonify(
            {"error": "Method Not Allowed", "message": "HTTP method not supported."}
        ), 405

    @app.errorhandler(413)
    def request_too_large(_error):
        return jsonify(
            {"error": "Payload Too Large", "message": "Request body must be at most 8 MB."}
        ), 413

    return app


app = create_app()

if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() in {"true", "1", "yes"}
    app.run(host=host, port=port, debug=debug)
