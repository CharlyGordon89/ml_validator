import pandas as pd

class SchemaValidator:
    def __init__(
        self,
        schema: dict = None,
        allow_null: list = None,
        ranges: dict = None,
    ):
        """
        Initialize SchemaValidator.

        Args:
            schema: Dict of expected column dtypes. Example: {'age': 'int64'}
            allow_null: List of columns that can have nulls.
            ranges: Dict of {'col': {'min': x, 'max': y}} to validate numeric ranges.
        """
        self.schema = schema or {}
        self.allow_null = allow_null or []
        self.ranges = ranges or {}

    def validate(self, df: pd.DataFrame) -> None:
        """
        Run all validation checks.
        """
        if self.schema:
            self.validate_schema(df, self.schema)
        if self.allow_null is not None:
            self.validate_no_nulls(df, self.allow_null)
        if self.ranges:
            self.validate_ranges(df, self.ranges)

    @staticmethod
    def validate_schema(df: pd.DataFrame, required_columns: dict) -> None:
        """
        Validate DataFrame columns and dtypes.
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

    @staticmethod
    def validate_no_nulls(df: pd.DataFrame, allow_null: list = None) -> None:
        """
        Validate that no unexpected null values exist.
        """
        allow_null = allow_null or []
        null_cols = df.isnull().any()
        problematic = [
            col for col, has_nulls in null_cols.items()
            if has_nulls and col not in allow_null
        ]
        if problematic:
            raise ValueError(
                f"Null values detected in columns: {problematic}\n"
                f"Columns allowing nulls: {allow_null}\n"
                f"Null counts:\n{df.isnull().sum().to_dict()}"
            )

    @staticmethod
    def validate_ranges(df: pd.DataFrame, ranges: dict) -> None:
        """
        Validate numerical columns fall within expected ranges.
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
