import json

class DatasetValidationError(Exception):
    pass


class DatasetValidator:
    def __init__(self, contract_path: str):
        with open(contract_path, "r") as f:
            self.contract = json.load(f)

    def validate(self, df):
        self._check_missing(df)
        self._check_labels(df)
        self._check_columns(df)

    def _check_missing(self, df):
        missing_pct = df.isna().mean().mean()
        if missing_pct > self.contract["max_missing_pct"]:
            raise DatasetValidationError(
                f"Missing value ratio too high: {missing_pct:.2f}"
            )

    def _check_labels(self, df):
        allowed = set(self.contract["label_values"])
        found = set(df["label"].unique())
        if not found.issubset(allowed):
            raise DatasetValidationError(
                f"Invalid label values found: {found}"
            )

    def _check_columns(self, df):
        required = (
            self.contract["categorical_features"]
            + self.contract["numeric_features"]
            + ["label"]
        )
        missing = set(required) - set(df.columns)
        if missing:
            raise DatasetValidationError(
                f"Missing required columns: {missing}"
            )
