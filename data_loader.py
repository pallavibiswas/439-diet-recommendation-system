# data_loader.py
"""
data_loader.py
Handles loading the recipes dataset from CSV or SQL database into a DataFrame.
"""

import pandas as pd
from sqlalchemy import create_engine
from typing import Optional

def load_csv(path: str) -> pd.DataFrame:
    """
    Load recipes data from a local CSV file.
    """
    return pd.read_csv(path)

# Optional: load from database if needed
def load_database(connection_string: str, table_name: str) -> pd.DataFrame:
    """
    Load recipes data from a SQL database via SQLAlchemy.
    """
    engine = create_engine(connection_string)
    return pd.read_sql_table(table_name, con=engine)