"""
Collect Raw Data from original source,
get the metadata,
save the data in the raw folder
and generate the statistical summary of the data for data drift detection.
"""

from pathlib import Path
from ucimlrepo import fetch_ucirepo
from src.logging.console_log import setup_logging
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_FOLDER = PROJECT_ROOT / "data"

# setup logging
logger = setup_logging()


def main():
    """Pipeline for collecting data from the original source"""

    # fetch dataset
    logger.info("Fetching dataset")
    wholesale_customers = fetch_ucirepo(id=292)

    # data (as pandas dataframes)
    X = wholesale_customers.data.features
    y = wholesale_customers.data.targets

    # metadata
    metadata = wholesale_customers.variables

    logger.info("Data fetched successfully")

    # Generate the statistical summary of the data for data drift detection
    statistical_summary = X.describe()
    logger.info("Statistical summary generated successfully")

    # save the data in the raw folder
    X.to_csv(DATA_FOLDER / "raw/wholesale_customers_data.csv", index=False)
    y.to_csv(DATA_FOLDER / "raw/wholesale_customers_targets.csv", index=False)
    logger.info("Data saved successfully")

    # Create train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logger.info("Train-test split created successfully")

    # Create train_test folder if it doesn't exist
    (DATA_FOLDER / "train_test").mkdir(exist_ok=True)

    # Save train-test splits
    X_train.to_csv(DATA_FOLDER / "train_test/X_train.csv", index=False)
    X_test.to_csv(DATA_FOLDER / "train_test/X_test.csv", index=False)
    y_train.to_csv(DATA_FOLDER / "train_test/y_train.csv", index=False)
    y_test.to_csv(DATA_FOLDER / "train_test/y_test.csv", index=False)
    logger.info("Train-test split saved successfully")

    # save the metadata in the reference folder
    metadata.to_csv(DATA_FOLDER / "reference/wholesale_customers_metadata.csv")

    # save the statistical summary in the reference folder
    statistical_summary.to_csv(
        DATA_FOLDER / "reference/wholesale_customers_statistical_summary.csv"
    )
    logger.info("Metadata and statistical summary saved successfully")


if __name__ == "__main__":
    main()
