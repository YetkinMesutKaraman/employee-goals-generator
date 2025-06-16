import logging

import pandas as pd

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_employee_data(file_path: str) -> pd.DataFrame:
    """
    Load employee data from a CSV file.

    Args:
        file_path (str): Path to the CSV file containing employee data.

    Returns:
        pd.DataFrame: DataFrame containing employee data.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        logger.error(f"Error loading employee data: {e}")
        raise
