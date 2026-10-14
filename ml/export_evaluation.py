"""Export reproducible evaluation metrics and plots from a completed notebook run."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_DETAILS = {
    "Baseline ANN": {
        "role": "Flattened dense baseline",
        "architecture": "Flatten -> Dense(256, ReLU) -> Dropout(0.2) -> Dense(128, ReLU) -> Dropout(0.2) -> Softmax(10)",
        "reason": "Provides a dense-network baseline for measuring the value of spatial feature extraction.",
    },
    "CNN Config 1 (Standard)": {
        "role": "Two-block CNN",
        "architecture": "Conv2D(32, 3x3, same, ReLU) -> MaxPool(2x2) -> Conv2D(64, 3x3, same, ReLU) -> MaxPool(2x2) -> Dense(128, ReLU) -> Dropout(0.5) -> Softmax(10)",
        "reason": "Tests a compact CNN that learns local image features before classifying digits.",
    },
    "CNN Config 2 (Deep + BN)": {
        "role": "Deeper CNN with batch normalization",
        "architecture": "Two Conv2D(32, 3x3, same, ReLU) + BatchNorm layers -> MaxPool(2x2) -> Dropout(0.25) -> Two Conv2D(64, 3x3, same, ReLU) + BatchNorm layers -> MaxPool(2x2) -> Dropout(0.25) -> Dense(128, ReLU) + BatchNorm -> Dropout(0.5) -> Softmax(10)",
        "reason": "Tests whether additional convolutional depth and batch normalization improve held-out performance.",
    },
}


def _history_values(history):
    return {
        key: [float(value) for value in history.history.get(key, [])]
        for key in ("accuracy", "val_accuracy", "loss", "val_loss")
    }


def _rounded(value):
    return round(float(value), 6)


def _plot_training_curves(models_dict, output_path):
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    colors = ("#2865d8", "#27a776", "#d64b5b")

    for (name, (_, history)), color in zip(models_dict.items(), colors):
        values = _history_values(history)
        epochs = range(1, len(values["accuracy"]) + 1)
        axes[0].plot(epochs, values["accuracy"], color=color, label=f"{name} train")
        axes[0].plot(
            epochs,
            values["val_accuracy"],
            color=color,
            linestyle="--",
            label=f"{name} validation",
        )
        axes[1].plot(epochs, values["loss"], color=color, label=f"{name} train")
        axes[1].plot(
            epochs,
            values["val_loss"],
            color=color,
            linestyle="--",
            label=f"{name} validation",
        )

    axes[0].set(title="Training and validation accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[1].set(title="Training and validation loss", xlabel="Epoch", ylabel="Loss")
    for axis in axes:
        axis.grid(True, linestyle="--", alpha=0.35)
        axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plot_confusion_matrix(matrix, output_path, model_name):
    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis, label="Test samples")
    axis.set(
        title=f"Test confusion matrix: {model_name}",
        xlabel="Predicted digit",
        ylabel="True digit",
        xticks=range(10),
        yticks=range(10),
    )
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                str(int(matrix[row, column])),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "#172238",
                fontsize=8,
            )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plot_prediction_examples(images, labels, predictions, probabilities, output_path):
    correct_indices = np.flatnonzero(predictions == labels)[:5]
    incorrect_indices = np.flatnonzero(predictions != labels)[:5]
    figure, axes = plt.subplots(2, 5, figsize=(12, 5))

    for row, indices, heading, color in (
        (0, correct_indices, "Correct", "#16845d"),
        (1, incorrect_indices, "Misclassified", "#c23b50"),
    ):
        for column, axis in enumerate(axes[row]):
            axis.axis("off")
            if column >= len(indices):
                continue
            index = indices[column]
            confidence = float(probabilities[index, predictions[index]]) * 100
            axis.imshow(images[index].squeeze(), cmap="gray")
            axis.set_title(
                f"True: {labels[index]} | Pred: {predictions[index]}\nConfidence: {confidence:.1f}%",
                color=color,
                fontsize=8,
            )
        axes[row, 0].set_ylabel(heading, color=color, fontweight="bold")

    figure.suptitle("Examples from the held-out MNIST test set")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def export_evaluation(
    models_dict,
    x_test,
    y_test,
    selected_model_name,
    seed=42,
    batch_size=128,
    max_epochs=20,
    early_stopping_patience=5,
):
    """Evaluate notebook-trained models and write the dashboard's actual run artifacts."""
    if selected_model_name not in models_dict:
        raise ValueError(f"Selected model {selected_model_name!r} is missing from models_dict.")
    if not all(name in models_dict for name in MODEL_DETAILS):
        raise ValueError("The notebook must provide the baseline ANN and both CNN configurations.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(y_test)
    comparison = []
    histories = {}
    selected_probabilities = None
    selected_predictions = None
    selected_confusion = None
    selected_metrics = None

    for name, (model, history) in models_dict.items():
        probabilities = np.asarray(model.predict(x_test, batch_size=256, verbose=0))
        predictions = np.argmax(probabilities, axis=1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average="macro", zero_division=0
        )
        loss = model.evaluate(x_test, labels, batch_size=256, verbose=0, return_dict=True)["loss"]
        history_data = _history_values(history)
        best_epoch = int(np.argmin(history_data["val_loss"]) + 1)
        model_metrics = {
            "test_accuracy": _rounded(accuracy_score(labels, predictions)),
            "test_loss": _rounded(loss),
            "macro_precision": _rounded(precision),
            "macro_recall": _rounded(recall),
            "macro_f1_score": _rounded(f1),
        }
        comparison.append(
            {
                "name": name,
                **MODEL_DETAILS[name],
                **model_metrics,
                "best_epoch": best_epoch,
                "epochs_trained": len(history_data["accuracy"]),
                "validation_accuracy_at_best_epoch": _rounded(
                    history_data["val_accuracy"][best_epoch - 1]
                ),
                "validation_loss_at_best_epoch": _rounded(
                    history_data["val_loss"][best_epoch - 1]
                ),
                "parameter_count": int(model.count_params()),
            }
        )
        histories[name] = history_data
        if name == selected_model_name:
            selected_probabilities = probabilities
            selected_predictions = predictions
            selected_confusion = confusion_matrix(labels, predictions, labels=range(10))
            selected_metrics = model_metrics

    _plot_training_curves(models_dict, RESULTS_DIR / "training_curves.png")
    _plot_confusion_matrix(
        selected_confusion,
        RESULTS_DIR / "confusion_matrix.png",
        selected_model_name,
    )
    _plot_prediction_examples(
        x_test,
        labels,
        selected_predictions,
        selected_probabilities,
        RESULTS_DIR / "predictions_analysis.png",
    )

    payload = {
        "dataset": {
            "name": "MNIST handwritten digits (Keras dataset)",
            "image_shape": [int(size) for size in x_test.shape[1:]],
            "classes": list(range(10)),
            "training_samples": 50000,
            "validation_samples": 10000,
            "test_samples": int(len(labels)),
            "source_training_samples": 60000,
            "preprocessing": [
                "Convert grayscale pixel values to float32 and divide by 255, scaling pixels to [0, 1].",
                "Add a single channel dimension, producing 28 x 28 x 1 input tensors.",
                "Create a stratified 10,000-image validation split from the 60,000-image training partition (seed 42).",
                "Keep the original 10,000-image test partition separate for final model evaluation.",
            ],
        },
        "hyperparameters": {
            "seed": int(seed),
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "loss": "Sparse categorical cross-entropy (integer digit labels)",
            "batch_size": int(batch_size),
            "maximum_epochs": int(max_epochs),
            "early_stopping": {
                "monitor": "val_loss",
                "patience": int(early_stopping_patience),
                "restore_best_weights": True,
            },
            "reasons": [
                "Adam with a 0.001 starting learning rate provides an adaptive optimizer baseline.",
                "Batch size 128 balances gradient updates with the amount of data processed per step.",
                "Sparse categorical cross-entropy matches integer labels and the ten-class softmax output.",
                "Early stopping monitors validation loss and restores the best weights to limit overfitting.",
                "A fixed seed makes data splitting and training comparisons more repeatable.",
            ],
        },
        "selected_model": selected_model_name,
        "selected_metrics": selected_metrics,
        "comparison": comparison,
        "training_histories": histories,
        "confusion_matrix": selected_confusion.tolist(),
    }
    result_path = RESULTS_DIR / "evaluation.json"
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Evaluation data and plots saved to: {RESULTS_DIR}")
    print(f"Dashboard data saved to: {result_path}")
    return payload
