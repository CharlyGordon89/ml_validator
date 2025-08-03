import pandas as pd
import numpy as np
import pytest
from ml_validator.core.validator import SchemaValidator

def test_valid_schema_passes():
    df = pd.DataFrame({"age": [30], "name": ["Alice"]})
    schema = {"age": "int64", "name": "object"}
    validator = SchemaValidator(schema=schema)
    validator.validate(df)  # Should not raise

def test_missing_column_raises():
    df = pd.DataFrame({"age": [30]})
    schema = {"age": "int64", "name": "object"}
    validator = SchemaValidator(schema=schema)
    with pytest.raises(ValueError, match="Missing columns"):
        validator.validate(df)

def test_wrong_dtype_raises():
    df = pd.DataFrame({"age": ["30"], "name": ["Alice"]})
    schema = {"age": "int64", "name": "object"}
    validator = SchemaValidator(schema=schema)
    with pytest.raises(ValueError, match="Expected dtype"):
        validator.validate(df)

def test_nulls_in_disallowed_column():
    df = pd.DataFrame({"age": [np.nan], "name": ["Bob"]})
    schema = {"age": "float64", "name": "object"}
    validator = SchemaValidator(schema=schema, allow_null=["name"])
    with pytest.raises(ValueError, match="Null values detected"):
        validator.validate(df)

def test_nulls_allowed_column_passes():
    df = pd.DataFrame({"age": [np.nan], "name": ["Bob"]})
    schema = {"age": "float64", "name": "object"}
    validator = SchemaValidator(schema=schema, allow_null=["age", "name"])
    validator.validate(df)  # Should not raise

def test_value_below_min_raises():
    df = pd.DataFrame({"age": [-5]})
    ranges = {"age": {"min": 0}}
    validator = SchemaValidator(schema={"age": "int64"}, ranges=ranges)
    with pytest.raises(ValueError, match="below minimum"):
        validator.validate(df)

def test_value_above_max_raises():
    df = pd.DataFrame({"age": [130]})
    ranges = {"age": {"max": 120}}
    validator = SchemaValidator(schema={"age": "int64"}, ranges=ranges)
    with pytest.raises(ValueError, match="above maximum"):
        validator.validate(df)

def test_range_within_bounds_passes():
    df = pd.DataFrame({"age": [25]})
    ranges = {"age": {"min": 0, "max": 120}}
    validator = SchemaValidator(schema={"age": "int64"}, ranges=ranges)
    validator.validate(df)  # Should not raise
