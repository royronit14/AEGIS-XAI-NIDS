import json
import pyarrow.parquet as pq
import pandas as pd

from aegis.loaders.base_loader import BaseLoader
from aegis.loaders.column_mapper import ColumnMapper
from aegis.validation.dataset_validator import DatasetValidator


class BoTIoTLoader(BaseLoader):
    def __init__(self, schema_path: str, contract_path: str, batch_size=200_000):
        super().__init__(schema_path)

        with open(contract_path, "r") as f:
            contract = json.load(f)

        self.contract = contract
        self.validator = DatasetValidator(contract_path)
        self.mapper = ColumnMapper(contract)
        self.batch_size = batch_size

    def load_iter(self, parquet_path: str):
        """
        Generator that yields validated, schema-enforced chunks.
        RAM-safe for 8GB systems.
        """
        pf = pq.ParquetFile(parquet_path)

        for batch in pf.iter_batches(batch_size=self.batch_size):
            df = batch.to_pandas()

            # Map + normalize
            df = self.mapper.map_columns(df)

            # Enforce schema (fills missing safely)
            df = self.enforce_schema(df)

            # Validate (cheap checks)
            self.validator.validate(df)

            yield df
