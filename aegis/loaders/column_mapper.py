class ColumnMappingError(Exception):
    pass


class ColumnMapper:
    def __init__(self, contract: dict):
        self.raw_label_column = contract["label_column"]
        self.aliases = contract.get("column_aliases", {})
        self.label_mapping = contract.get("label_mapping", None)

    def map_columns(self, df):
        df_columns = set(df.columns)

        # 1. Check label column exists
        if self.raw_label_column not in df_columns:
            raise ColumnMappingError(
                f"Label column '{self.raw_label_column}' not found in dataset"
            )

        # 2. Normalize label → canonical "label"
        df = df.rename(columns={self.raw_label_column: "label"})

        # 3. Apply label mapping if provided
        if self.label_mapping:
            benign_key = None
            for k, v in self.label_mapping.items():
                if v == 0:
                    benign_key = k

            if benign_key is None:
                raise ColumnMappingError("No benign label defined in contract")

            df["label"] = df["label"].apply(
                lambda x: 0 if x == benign_key else 1
            )

        # 4. Rename feature aliases
        rename_map = {}
        for raw_col, unified_col in self.aliases.items():
            if raw_col in df_columns:
                rename_map[raw_col] = unified_col

        df = df.rename(columns=rename_map)

        return df
