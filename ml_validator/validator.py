import pandas as pd 

def validate_schema(df: pd.DataFrame, required_columns: dict) -> None:
    """
    Validate that a DataFrame matches the expected column schema.
    
    Args:
        df: Input DataFrame to validate
        required_columns: Dictionary of {column_name: expected_dtype}
                         Example: {"age": "int64", "name": "object"}
    
    Raises:
        ValueError: If any columns are missing or have incorrect data types
                   Error message includes detailed mismatch information
    
    Example:
        >>> df = pd.DataFrame({"age": [25], "name": ["Alice"]})
        >>> validate_schema(df, {"age": "int64", "name": "object"})  # Passes
        >>> validate_schema(df, {"age": "float64"})  # Raises ValueError
    """
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"Schema violation\n"
            f"Missing columns: {missing}\n"
            f"Expected columns: {list(required_columns.keys())}\n"
            f"Actual columns: {list(df.columns)}"
        )
    
    for col, dtype in required_columns.items():
        if df[col].dtype != dtype:
            raise ValueError(
                f"Schema violation\n"
                f"Column: '{col}'\n"
                f"Expected dtype: {dtype}\n"
                f"Actual dtype: {df[col].dtype}\n"
                f"All dtypes:\n{df.dtypes.to_dict()}"
            )
        

def validate_no_nulls(df: pd.DataFrame, allow_null: list = None) -> None:
    """
    Validate no unexpected null values exist.
    
    Args:
        df: DataFrame to check
        allow_null: List of columns where nulls are permitted
    """
    if allow_null is None:
        allow_null = []
    
    null_cols = df.isnull().any()
    problematic = [col for col, has_nulls in null_cols.items() 
             if has_nulls and col not in allow_null]
    
    if problematic:
        raise ValueError(
            f"Null values detected in columns: {problematic}\n"
            f"Columns allowing nulls: {allow_null}\n"
            f"Null counts:\n{df.isnull().sum().to_dict()}"
        )
        

def validate_ranges(df: pd.DataFrame, ranges: dict) -> None:
    """
    Validate numerical columns fall within expected ranges.
    
    Args:
        ranges: Dict like {'age': {'min': 0, 'max': 120}}
    """
    for col, specs in ranges.items():
        if col in df:
            if 'min' in specs and (df[col] < specs['min']).any():
                raise ValueError(
                    f"Value below minimum in '{col}'\n"
                    f"Min allowed: {specs['min']}\n"
                    f"Invalid values: {df[df[col] < specs['min']][col].tolist()}"
                )
            if 'max' in specs and (df[col] > specs['max']).any():
                raise ValueError(
                    f"Value above maximum in '{col}'\n"
                    f"Max allowed: {specs['max']}\n"
                    f"Invalid values: {df[df[col] > specs['max']][col].tolist()}"
                )
                


