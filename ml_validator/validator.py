# ml_validator/validator.py

import pandas as pd

def validate_schema(df: pd.DataFrame, required_columns: dict) -> None:
    """
    Validate that required columns exist and have the correct dtype.

    Parameters:
    - df: pd.DataFrame - The dataframe to validate.
    - required_columns: dict - Column names and expected dtypes, e.g. {"age": "int64"}

    Raises:
    - ValueError if a column is missing or has incorrect type.
    """
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    for col, dtype in required_columns.items():
        if df[col].dtype != dtype:
            raise ValueError(f"Incorrect dtype for column '{col}': expected {dtype}, got {df[col].dtype}")

