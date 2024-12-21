"""
Test the trained model and save classification metrics
This script evaluates the model performance using test data and saves metrics
for use in CML (Continuous Machine Learning) reporting
"""

import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from src.logging.console_log import setup_logging
from settings import settings

# setup logging
logger = setup_logging()


def load_model():
    """Load the trained model"""
    with open(settings.MODEL_PATH, "rb") as file:
        model = pickle.load(file)
    return model


def evaluate_model(y_true, y_pred):
    """Calculate classification metrics"""
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted"
    )

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
    }
    return metrics


def save_metrics(metrics):
    """Save metrics to JSON file"""
    settings.METRICS_PATH.mkdir(parents=True, exist_ok=True)

    metrics_file = settings.METRICS_PATH / "classification_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Metrics saved to {metrics_file}")


def main():
    """Run model testing and evaluation"""
    # Load test data
    logger.info("Loading test data")
    X_test = pd.read_csv(settings.TRAIN_TEST_FOLDER / "X_test.csv")
    y_test = pd.read_csv(settings.TRAIN_TEST_FOLDER / "y_test.csv")

    # Load and preprocess test data using the same preprocessing steps
    logger.info("Loading scaler")
    with open(settings.SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    # Apply preprocessing
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test[settings.NUMERICAL_FEATURE_COLUMNS]),
        columns=settings.NUMERICAL_FEATURE_COLUMNS,
    )

    for col in settings.CATEGORICAL_COLUMNS:
        X_test_scaled[col] = X_test[col]

    # Load model and make predictions
    logger.info("Loading model and making predictions")
    model = load_model()
    y_pred = model.predict(X_test_scaled)

    # Calculate and save metrics
    logger.info("Calculating metrics")
    metrics = evaluate_model(y_test[settings.TARGET_COLUMN_NAME], y_pred)
    logger.info(f"Model performance metrics: {metrics}")

    # Save metrics for CML
    save_metrics(metrics)


if __name__ == "__main__":
    main()
