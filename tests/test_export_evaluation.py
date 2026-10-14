import json
from types import SimpleNamespace

import numpy as np

from ml import export_evaluation


class FixedModel:
    def __init__(self, predictions):
        self.predictions = predictions

    def predict(self, images, batch_size, verbose):
        probabilities = np.full((len(images), 10), 0.1 / 9)
        probabilities[np.arange(len(images)), self.predictions] = 0.9
        return probabilities

    def evaluate(self, images, labels, batch_size, verbose, return_dict):
        assert return_dict is True
        return {"loss": 0.1}

    def count_params(self):
        return 100


def test_export_evaluation_writes_test_metrics_histories_and_plots(tmp_path, monkeypatch):
    monkeypatch.setattr(export_evaluation, "RESULTS_DIR", tmp_path)
    labels = np.arange(10)
    images = np.zeros((10, 28, 28, 1), dtype=np.float32)
    histories = {
        "accuracy": [0.8, 0.9],
        "val_accuracy": [0.75, 0.85],
        "loss": [0.7, 0.3],
        "val_loss": [0.8, 0.4],
    }
    models = {
        "Baseline ANN": (FixedModel((labels + 1) % 10), SimpleNamespace(history=histories)),
        "CNN Config 1 (Standard)": (FixedModel(labels), SimpleNamespace(history=histories)),
        "CNN Config 2 (Deep + BN)": (
            FixedModel(np.where(labels == 0, 1, labels)),
            SimpleNamespace(history=histories),
        ),
    }

    result = export_evaluation.export_evaluation(
        models,
        images,
        labels,
        "CNN Config 1 (Standard)",
    )
    saved_result = json.loads((tmp_path / "evaluation.json").read_text(encoding="utf-8"))

    assert result == saved_result
    assert len(result["comparison"]) == 3
    assert result["comparison"][0]["test_accuracy"] == 0
    assert result["selected_metrics"]["test_accuracy"] == 1
    assert result["selected_metrics"]["macro_precision"] == 1
    assert result["training_histories"]["CNN Config 1 (Standard)"]["val_loss"] == [0.8, 0.4]
    assert np.asarray(result["confusion_matrix"]).trace() == 10
    for filename in ("training_curves.png", "confusion_matrix.png", "predictions_analysis.png"):
        assert (tmp_path / filename).read_bytes().startswith(b"\x89PNG")
