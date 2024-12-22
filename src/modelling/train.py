"""
Train the model using Random Forest Classifier
"""

from sklearn.ensemble import RandomForestClassifier
from src.logging.console_log import setup_logging
import pandas as pd
import pickle
from settings import settings

# setup logging
logger = setup_logging()


def train_model(X_train: pd.DataFrame, y_train: pd.DataFrame) -> RandomForestClassifier:
    """Train the model using Random Forest

    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.DataFrame): Training target

    Returns:
        RandomForestClassifier: Trained model
    """    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=settings.RANDOM_STATE
    )
    logger.info(f"Training the model using {model}")
    model.fit(X_train, y_train[settings.TARGET_COLUMN_NAME])
    return model


def main():
    """Run Model Training"""

    # Load preprocessed data
    logger.info("Loading data")
    X_train = pd.read_csv(settings.PROCESSED_DATA_FOLDER / "X_train_processed.csv")
    y_train = pd.read_csv(settings.PROCESSED_DATA_FOLDER / "y_train_processed.csv")

    # Train the model
    model = train_model(X_train, y_train)

    # Create model directory if it doesn't exist
    settings.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Save the model
    with open(settings.MODEL_PATH, "wb") as file:
        pickle.dump(model, file)

    logger.info(f"Model saved successfully at {settings.MODEL_PATH}")


if __name__ == "__main__":
    main()
