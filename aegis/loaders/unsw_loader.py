import json
import pandas as pd

from aegis.loaders.base_loader import BaseLoader
from aegis.loaders.column_mapper import ColumnMapper
from aegis.validation.dataset_validator import DatasetValidator


class UNSWNB15Loader(BaseLoader):
    def __init__(self, schema_path: str, contract_path: str):
        super().__init__(schema_path)

        with open(contract_path, "r") as f:
            contract = json.load(f)

        self.contract = contract
        self.validator = DatasetValidator(contract_path)
        self.mapper = ColumnMapper(contract)

    def load(self, parquet_path: str) -> pd.DataFrame:
        # 1. Load parquet
        df = pd.read_parquet(parquet_path)

        # 2. Auto-map columns using contract
        df = self.mapper.map_columns(df)

        # 3. Enforce universal schema (fills missing features safely)
        df = self.enforce_schema(df)

        # 4. Validate dataset
        self.validator.validate(df)

        return df
