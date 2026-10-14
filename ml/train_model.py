"""
Training script for Handwritten Digit Classification using MNIST dataset.
Trains a Convolutional Neural Network (CNN) and saves the model to models/mnist_cnn.keras.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Configure reproducible seeds
np.random.seed(42)
tf.random.set_seed(42)

def train_and_save_model():
    print("=" * 60)
    print("Starting MNIST CNN Model Training Pipeline")
    print("=" * 60)

    # 1. Ensure output directories exist
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models")
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # 2. Load MNIST Dataset
    print("\n[Step 1/5] Loading MNIST dataset from Keras...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    print(f"  Training samples:   {x_train.shape[0]} images of shape {x_train.shape[1:]}")
    print(f"  Test samples:       {x_test.shape[0]} images of shape {x_test.shape[1:]}")

    # 3. Preprocessing: Reshape and normalize to [0, 1]
    print("\n[Step 2/5] Preprocessing and normalizing images...")
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Expand dimensions to (28, 28, 1) for grayscale channel
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    print(f"  Processed training tensor shape: {x_train.shape}")
    print(f"  Processed test tensor shape:     {x_test.shape}")

    # 4. Define CNN Architecture
    print("\n[Step 3/5] Constructing CNN architecture...")
    model = keras.Sequential(
        [
            layers.Input(shape=(28, 28, 1)),
            # First Convolutional Block
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu", name="conv1"),
            layers.MaxPooling2D(pool_size=(2, 2), name="pool1"),
            # Second Convolutional Block
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu", name="conv2"),
            layers.MaxPooling2D(pool_size=(2, 2), name="pool2"),
            # Flatten & Fully Connected Classifier Block
            layers.Flatten(name="flatten"),
            layers.Dropout(0.5, name="dropout"),
            layers.Dense(128, activation="relu", name="dense1"),
            layers.Dense(10, activation="softmax", name="predictions"),
        ],
        name="mnist_cnn",
    )

    model.summary()

    # 5. Compile the Model
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # 6. Train the Model
    epochs = 6
    batch_size = 128
    print(f"\n[Step 4/5] Training model for {epochs} epochs (batch size: {batch_size})...")
    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1,
        verbose=1,
    )

    # 7. Evaluate on Test Set
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n  Final Test Loss:     {test_loss:.4f}")
    print(f"  Final Test Accuracy: {test_accuracy * 100:.2f}%")

    # 8. Save the Model
    model_save_path = os.path.join(models_dir, "mnist_cnn.keras")
    print(f"\n[Step 5/5] Saving model weights to: {model_save_path}")
    model.save(model_save_path)
    print("  Model saved successfully!")

    # 9. Plot and Save Training Curves
    print("\nGenerating training history plot...")
    plt.figure(figsize=(12, 4))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Training Accuracy", color="#2563eb", linewidth=2)
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy", color="#10b981", linewidth=2)
    plt.title("Model Accuracy over Epochs", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Training Loss", color="#ef4444", linewidth=2)
    plt.plot(history.history["val_loss"], label="Validation Loss", color="#f59e0b", linewidth=2)
    plt.title("Model Loss over Epochs", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    curves_save_path = os.path.join(results_dir, "training_curves.png")
    plt.savefig(curves_save_path, dpi=200)
    plt.close()
    print(f"  Training curves saved to: {curves_save_path}")
    print("\n" + "=" * 60)
    print("Training Pipeline Completed Successfully!")
    print("=" * 60)

if __name__ == "__main__":
    train_and_save_model()
