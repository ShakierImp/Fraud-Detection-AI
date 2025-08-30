import os
import subprocess
import logging
import pandas as pd

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)


def load_local(path: str) -> pd.DataFrame:
    """Load a CSV file from the local filesystem into a pandas DataFrame.

    Args:
        path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: The loaded DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist at the given path.
        pd.errors.EmptyDataError: If the file is empty.
        pd.errors.ParserError: If the file format is invalid.
        Exception: For any other unexpected errors during reading.
    """
    if not os.path.exists(path) or not os.path.isfile(path):
        logger.error("File not found at path: %s", path)
        raise FileNotFoundError(f"File not found: {path}")

    try:
        df = pd.read_csv(path)
        logger.info("Successfully loaded CSV file: %s", path)
        return df
    except pd.errors.EmptyDataError as e:
        logger.error("Empty file error while reading CSV: %s", e)
        raise
    except pd.errors.ParserError as e:
        logger.error("Parsing error while reading CSV: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error while reading CSV: %s", e)
        raise


def get_sample_df() -> pd.DataFrame:
    """Load the default sample dataset from 'data/sample_input.csv'.

    Returns:
        pd.DataFrame: The sample dataset as a DataFrame.

    Raises:
        FileNotFoundError: If the sample file is missing.
        Exception: For any other errors encountered during loading.
    """
    sample_path = "data/sample_input.csv"
    return load_local(sample_path)


def download_kaggle(dataset_slug: str, dest_dir: str = "data/raw") -> str:
    """Download a dataset from Kaggle using the Kaggle command-line API.

    Args:
        dataset_slug (str): The Kaggle dataset slug (e.g., 'zynicide/wine-reviews').
        dest_dir (str, optional): Directory to save the downloaded dataset. Defaults to 'data/raw'.

    Returns:
        str: Absolute path to the destination directory where dataset is stored.

    Raises:
        RuntimeError: If Kaggle API credentials are not configured or if the download fails.
    """
    if not os.getenv("KAGGLE_USERNAME"):
        msg = (
            "Kaggle API not configured. "
            "Please set KAGGLE_USERNAME and KAGGLE_KEY environment variables."
        )
        logger.error(msg)
        raise RuntimeError(msg)

    os.makedirs(dest_dir, exist_ok=True)

    command = [
        "kaggle",
        "datasets",
        "download",
        "-p",
        dest_dir,
        "--unzip",
        dataset_slug,
    ]
    logger.info("Running command: %s", " ".join(command))

    result = subprocess.run(
        command, capture_output=True, text=True
    )

    if result.returncode != 0:
        logger.error(
            "Failed to download dataset from Kaggle. Error: %s", result.stderr
        )
        raise RuntimeError(
            f"Failed to download dataset {dataset_slug}: {result.stderr}"
        )

    logger.info("Successfully downloaded and extracted dataset: %s", dataset_slug)
    return os.path.abspath(dest_dir)


if __name__ == "__main__":
    # Demonstration of functionality
    print("=== Demonstrating data_collection module ===")

    # Try loading sample data
    try:
        df_sample = get_sample_df()
        print("Sample data loaded successfully:")
        print(df_sample.head())
    except Exception as e:
        print(f"Could not load sample data: {e}")

    # Try downloading a Kaggle dataset
    try:
        dataset_path = download_kaggle("zynicide/wine-reviews")
        print(f"Dataset downloaded to: {dataset_path}")
    except RuntimeError as e:
        print(f"Skipping Kaggle download demo: {e}")