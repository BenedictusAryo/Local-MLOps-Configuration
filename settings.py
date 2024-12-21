"""
Repository Configuration Settings
"""

from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings
from src.logging.console_log import setup_logging

logger = setup_logging()


class Settings(BaseSettings):
    # Paths
    PROJECT_ROOT: Path = Path(__file__).resolve().parent
    DATA_FOLDER: Path = PROJECT_ROOT / "data"
    RAW_DATA_FOLDER: Path = DATA_FOLDER / "raw"
    PROCESSED_DATA_FOLDER: Path = DATA_FOLDER / "data/processed"
    TRAIN_TEST_FOLDER: Path = DATA_FOLDER / "train_test"
    REFERENCE_FOLDER: Path = DATA_FOLDER / "reference"
    SAVED_MODEL_FOLDER: Path = PROJECT_ROOT / "saved_model"
    MODEL_PATH: Path = SAVED_MODEL_FOLDER / "model/model.pkl"
    SCALER_PATH: Path = SAVED_MODEL_FOLDER / "preprocessor/scaler.pkl"
    METRICS_PATH: Path = SAVED_MODEL_FOLDER / "metrics"

    # Data Configuration
    TARGET_COLUMN_NAME: str = "Region"
    NUMERICAL_FEATURE_COLUMNS: List[str] = [
        "Fresh",
        "Milk",
        "Grocery",
        "Frozen",
        "Detergents_Paper",
        "Delicassen",
    ]
    CATEGORICAL_COLUMNS: List[str] = ["Channel"]

    # Model Configuration
    RANDOM_STATE: int = 42
    TEST_SIZE: float = 0.2

    # API Configuration
    API_PORT: int = 8000

    class Config:
        env_file = ".config_params"
        env_file_encoding = "utf-8"


# Create settings instance
settings = Settings()
