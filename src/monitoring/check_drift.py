"""
Check for data drift by comparing current data distribution
with the reference distribution saved during training
"""
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
from settings import settings
from src.logging.console_log import setup_logging
from typing import Tuple

# setup logging
logger = setup_logging()


def check_distribution_drift(reference_data: pd.Series, current_data: pd.Series, threshold: float = 0.05) -> bool:
    """
    Perform Kolmogorov-Smirnov test for distribution drift

    Args:
        reference_data (pd.Series): Historical reference data
        current_data (pd.Series): Current data to check for drift
        threshold (float): Significance level for the KS test

    Returns:
        bool: True if drift is detected, False otherwise
    """
    _, p_value = stats.ks_2samp(reference_data, current_data)
    return p_value < threshold


def main():
    """Run drift detection analysis"""
    # Load reference statistics
    logger.info("Loading reference statistics")
    reference_stats = pd.read_csv(
        settings.REFERENCE_FOLDER / "wholesale_customers_statistical_summary.csv"
    )
    
    # Load current data
    logger.info("Loading current data for drift analysis")
    current_data = pd.read_csv(settings.RAW_DATA_FOLDER / "wholesale_customers_data.csv")
    
    # Check drift for each feature
    drift_detected = False
    logger.info("Checking for distribution drift in features")
    
    for column in settings.NUMERICAL_FEATURE_COLUMNS:
        if check_distribution_drift(reference_stats[column], current_data[column]):
            logger.warning(f"Drift detected in feature: {column}")
            drift_detected = True
    
    # Create drift status file
    drift_status_path = Path(settings.METRICS_PATH) / "drift_status.txt"
    drift_status_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(drift_status_path, "w") as f:
        if drift_detected:
            f.write("true")
            logger.warning("Data drift detected - retraining required")
        else:
            f.write("false")
            logger.info("No significant data drift detected")


if __name__ == "__main__":
    main()
