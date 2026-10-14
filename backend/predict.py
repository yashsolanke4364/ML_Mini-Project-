"""MNIST image preprocessing and inference using the saved Keras model."""

import base64
import binascii
import io
import logging
import os
import re
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(
    os.environ.get("MODEL_PATH", str(PROJECT_ROOT / "models" / "mnist_cnn.keras"))
).expanduser().resolve()
MODEL = None
MODEL_LOAD_ERROR = None


def load_model():
    """Load the trained model once when this module starts."""
    global MODEL, MODEL_LOAD_ERROR

    if MODEL is not None:
        return True

    if not MODEL_PATH.is_file():
        MODEL_LOAD_ERROR = (
            f"Model file not found: {MODEL_PATH}. Add the trained "
            "models/mnist_cnn.keras artifact produced by ml/train_model.ipynb; "
            "the API does not generate predictions without it."
        )
        logger.error(MODEL_LOAD_ERROR)
        return False

    if not zipfile.is_zipfile(MODEL_PATH):
        MODEL_LOAD_ERROR = (
            f"'{MODEL_PATH}' is not a valid .keras model archive. Replace it with "
            "the trained model saved from ml/train_model.ipynb."
        )
        logger.error(MODEL_LOAD_ERROR)
        return False

    try:
        from tensorflow import keras

        logger.info("Loading trained Keras model from %s", MODEL_PATH)
        MODEL = keras.models.load_model(MODEL_PATH)
        MODEL_LOAD_ERROR = None
        logger.info("Trained Keras model loaded successfully.")
        return True
    except Exception as exc:
        MODEL = None
        MODEL_LOAD_ERROR = f"Could not load Keras model from '{MODEL_PATH}': {exc}"
        logger.exception("Failed to load the trained Keras model.")
        return False


def _decode_image(base64_data: str) -> Image.Image:
    if not isinstance(base64_data, str) or not base64_data.strip():
        raise ValueError("Image data must be a non-empty base64 string.")

    encoded = base64_data.strip()
    if encoded.startswith("data:"):
        header, separator, encoded = encoded.partition(",")
        if not separator or ";base64" not in header.lower():
            raise ValueError("Image data URI must contain base64-encoded image data.")

    encoded = re.sub(r"\s+", "", encoded)
    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Image data is not valid base64.") from exc
    if not image_bytes:
        raise ValueError("Decoded image data is empty.")

    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            if image.width <= 0 or image.height <= 0 or image.width * image.height > 25_000_000:
                raise ValueError("Image dimensions are invalid or too large.")
            image.verify()

        with Image.open(io.BytesIO(image_bytes)) as image:
            return ImageOps.exif_transpose(image).copy()
    except ValueError:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Image data is not a supported or valid image.") from exc


def preprocess_image(base64_data: str) -> np.ndarray:
    """Convert an image to an MNIST-style (1, 28, 28, 1) float32 input."""
    image = _decode_image(base64_data)
    if "A" in image.getbands():
        rgba = image.convert("RGBA")
        rgb = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
        rgb.alpha_composite(rgba)
        grayscale = rgb.convert("L")
    else:
        grayscale = image.convert("L")

    pixels = np.asarray(grayscale, dtype=np.uint8)
    border = np.concatenate(
        (pixels[0, :], pixels[-1, :], pixels[:, 0], pixels[:, -1])
    )
    background_is_light = float(np.median(border)) >= 127.5
    foreground = 255 - pixels if background_is_light else pixels

    if int(foreground.max()) < 20 or int(foreground.max()) - int(foreground.min()) < 20:
        raise ValueError("Image appears empty. Please provide a clearly drawn digit.")

    threshold = max(20, int(foreground.max() * 0.1))
    active = Image.fromarray(np.where(foreground >= threshold, 255, 0).astype(np.uint8))
    bounds = active.getbbox()
    if bounds is None:
        raise ValueError("Image appears empty. Please provide a clearly drawn digit.")

    digit = Image.fromarray(foreground).crop(bounds)
    crop_width, crop_height = digit.size
    scale = 20.0 / max(crop_width, crop_height)
    resized_width = max(1, int(round(crop_width * scale)))
    resized_height = max(1, int(round(crop_height * scale)))
    digit = digit.resize((resized_width, resized_height), Image.Resampling.LANCZOS)

    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(digit, ((28 - resized_width) // 2, (28 - resized_height) // 2))
    tensor = np.asarray(canvas, dtype=np.float32) / 255.0
    return np.expand_dims(tensor, axis=(0, -1))


def predict_digit(base64_data: str) -> dict:
    """Predict a digit with the trained model and return JSON-safe values."""
    if MODEL is None:
        raise RuntimeError(
            MODEL_LOAD_ERROR or "The trained Keras model is not loaded."
        )

    input_tensor = preprocess_image(base64_data)
    raw_probabilities = np.asarray(MODEL.predict(input_tensor, verbose=0))
    if raw_probabilities.shape != (1, 10):
        raise RuntimeError(
            f"Model output must have shape (1, 10); got {raw_probabilities.shape}."
        )
    probabilities = raw_probabilities[0]
    if not np.isfinite(probabilities).all():
        raise RuntimeError("Model returned non-finite probabilities.")

    prediction = int(np.argmax(probabilities))
    confidence = float(probabilities[prediction])
    probability_list = [float(value) for value in probabilities]
    return {
        "prediction": prediction,
        "confidence": confidence,
        "confidence_percentage": round(confidence * 100, 2),
        "probabilities": {
            str(digit): probability_list[digit] for digit in range(10)
        },
        "probabilities_list": probability_list,
    }


# Loading here ensures one model load per server process, rather than per request.
load_model()
