"""
Preprocess the data for training the model
"""

from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
from imblearn.over_sampling import ADASYN
from src.logging.console_log import setup_logging
from typing import Tuple, Union
from settings import settings


# setup logging
logger = setup_logging()


def oversample_data(
    X: pd.DataFrame, y: pd.DataFrame, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Oversample the minority classes using ADASYN.

    Args:
        X (pd.DataFrame): Feature DataFrame
        y (pd.DataFrame): Target DataFrame
        random_state (int): Random state for reproducibility

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Oversampled features and targets
    """
    oversampler = ADASYN(random_state=random_state)
    X_resampled, y_resampled = oversampler.fit_resample(X, y)
    logger.info(
        f"After oversampling: {y_resampled[settings.TARGET_COLUMN_NAME].value_counts()}"
    )
    return X_resampled, y_resampled


def preprocess_fit(
    X: pd.DataFrame, feature_columns: list, scaler_path: Union[str, Path]
) -> StandardScaler:
    """
    Fit the StandardScaler and save it.

    Args:
        X (pd.DataFrame): Feature DataFrame
        feature_columns (list): List of numerical feature columns
        scaler_path (Union[str, Path]): Path to save the scaler

    Returns:
        StandardScaler: Fitted scaler object
    """
    scaler = StandardScaler()
    scaler.fit(X[feature_columns])

    # Save the scaler object
    scaler_path = Path(scaler_path)
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    logger.info(f"Scaler saved successfully to {scaler_path}")

    return scaler


def preprocess_transform(
    X: pd.DataFrame,
    scaler: StandardScaler,
    feature_columns: list,
    categorical_columns: list = ["Channel"],
) -> pd.DataFrame:
    """
    Transform the data using fitted scaler.

    Args:
        X (pd.DataFrame): Feature DataFrame
        scaler (StandardScaler): Fitted scaler object
        feature_columns (list): List of numerical feature columns
        categorical_columns (list): List of categorical columns to preserve

    Returns:
        pd.DataFrame: Transformed DataFrame
    """
    X_scaled = pd.DataFrame(
        scaler.transform(X[feature_columns]), columns=feature_columns
    )

    # Add categorical columns
    for col in categorical_columns:
        X_scaled[col] = X[col]

    return X_scaled


def main():
    """Run preprocessing steps"""
    # Load the data
    logger.info("Loading data")
    X_train = pd.read_csv(settings.TRAIN_TEST_FOLDER / "X_train.csv")
    y_train = pd.read_csv(settings.TRAIN_TEST_FOLDER / "y_train.csv")

    # Print data shape of label to check for class imbalance
    label_counts = y_train[settings.TARGET_COLUMN_NAME].value_counts()
    logger.info(f"Label counts: {label_counts}")

    # Oversample the data
    X_train_resampled, y_train_resampled = oversample_data(
        X_train, y_train, random_state=settings.RANDOM_STATE
    )

    # Fit and save the scaler
    scaler = preprocess_fit(
        X_train_resampled,
        settings.NUMERICAL_FEATURE_COLUMNS,
        settings.SCALER_PATH,
    )

    # Transform the data
    X_train_scaled = preprocess_transform(
        X_train_resampled,
        scaler,
        settings.NUMERICAL_FEATURE_COLUMNS,
        settings.CATEGORICAL_COLUMNS,
    )

    # Create processed folder if it doesn't exist
    settings.PROCESSED_DATA_FOLDER.mkdir(parents=True, exist_ok=True)

    # Save the processed data
    X_train_scaled.to_csv(
        settings.PROCESSED_DATA_FOLDER / "X_train_processed.csv", index=False
    )
    y_train_resampled.to_csv(
        settings.PROCESSED_DATA_FOLDER / "y_train_processed.csv", index=False
    )
    logger.info("Processed data saved successfully")


if __name__ == "__main__":
    main()
