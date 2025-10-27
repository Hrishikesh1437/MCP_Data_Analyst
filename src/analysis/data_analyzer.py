import pandas as pd

class DataAnalyzer:
    def __init__(self):
        pass

    def summarize_dataframe(self, df: pd.DataFrame, n_rows: int = 5) -> str:
        desc = df.describe(include='all').to_string()
        head = df.head(n_rows).to_csv(index=False)
        return f"DATAFRAME HEAD:\n{head}\n\nDESCRIBE:\n{desc}"
