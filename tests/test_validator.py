# tests/test_validator.py

import pandas as pd
import pytest
from ml_validator.validator import validate_schema

def test_valid_schema():
    df = pd.DataFrame({
        "age": [25, 30],
        "name": ["Alice", "Bob"]
    })
    schema = {"age": "int64", "name": "object"}
    validate_schema(df, schema)  # Should not raise

def test_missing_column():
    df = pd.DataFrame({
        "age": [25, 30]
    })
    schema = {"age": "int64", "name": "object"}
    with pytest.raises(ValueError):
        validate_schema(df, schema)

def test_wrong_dtype():
    df = pd.DataFrame({
        "age": ["25", "30"],  # wrong type
        "name": ["Alice", "Bob"]
    })
    schema = {"age": "int64", "name": "object"}
    with pytest.raises(ValueError):
        validate_schema(df, schema)

