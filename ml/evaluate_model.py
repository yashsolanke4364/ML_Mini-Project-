"""
Evaluation script for MNIST Handwritten Digit Classification.
Evaluates the trained CNN model on 10,000 test images, generates the confusion matrix heatmap,
and saves quantitative evaluation metrics to results/metrics.json.
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras

def evaluate_and_generate_reports():
    print("=" * 60)
    print("Evaluating Trained MNIST CNN Model")
    print("=" * 60)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "models", "mnist_cnn.keras")
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model not found at '{model_path}'. "
            "Please run 'python ml/train_model.py' first."
        )

    # 1. Load Model
    print(f"\n[1/4] Loading trained model from: {model_path}")
    model = keras.models.load_model(model_path)
    print("Model loaded successfully.")

    # 2. Load MNIST Test Data
    print("\n[2/4] Loading MNIST test dataset...")
    (_, _), (x_test_raw, y_test) = keras.datasets.mnist.load_data()
    x_test = (x_test_raw.astype("float32") / 255.0)[..., np.newaxis]
    print(f"Loaded {len(x_test)} test images.")

    # 3. Model Inference
    print("\n[3/4] Performing inference on all 10,000 test images...")
    predictions = model.predict(x_test, batch_size=256, verbose=1)
    y_pred = np.argmax(predictions, axis=1)

    # 4. Compute Metrics
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    cm = confusion_matrix(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_text = classification_report(y_test, y_pred, digits=4)

    print("\n" + "-" * 50)
    print(f"Test Accuracy: {test_acc * 100:.2f}%")
    print(f"Test Loss:     {test_loss:.4f}")
    print("-" * 50)
    print("\nClassification Report:\n")
    print(report_text)

    # 5. Plot and Save Confusion Matrix Heatmap
    print("\n[4/4] Plotting and saving Confusion Matrix heatmap...")
    plt.figure(figsize=(9, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        xticklabels=[str(i) for i in range(10)],
        yticklabels=[str(i) for i in range(10)],
        linewidths=0.5,
        linecolor="#e2e8f0",
    )
    plt.title("MNIST Handwritten Digit Confusion Matrix", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Predicted Digit Label", fontsize=12, labelpad=10)
    plt.ylabel("True Digit Label", fontsize=12, labelpad=10)
    plt.tight_layout()

    cm_save_path = os.path.join(results_dir, "confusion_matrix.png")
    plt.savefig(cm_save_path, dpi=200)
    plt.close()
    print(f"Confusion matrix plot saved to: {cm_save_path}")

    # 6. Save JSON Metrics Summary
    per_class_metrics = {}
    for digit in range(10):
        digit_str = str(digit)
        if digit_str in report_dict:
            per_class_metrics[digit_str] = {
                "precision": round(report_dict[digit_str]["precision"], 4),
                "recall": round(report_dict[digit_str]["recall"], 4),
                "f1_score": round(report_dict[digit_str]["f1-score"], 4),
                "support": int(report_dict[digit_str]["support"]),
            }

    metrics_payload = {
        "accuracy": round(float(test_acc), 4),
        "loss": round(float(test_loss), 4),
        "total_test_samples": int(len(y_test)),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class_metrics,
        "macro_avg": {
            "precision": round(report_dict["macro avg"]["precision"], 4),
            "recall": round(report_dict["macro avg"]["recall"], 4),
            "f1_score": round(report_dict["macro avg"]["f1-score"], 4),
        },
    }

    metrics_json_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"Quantitative metrics saved to: {metrics_json_path}")

    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)

if __name__ == "__main__":
    evaluate_and_generate_reports()
