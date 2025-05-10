# eda.py
"""
eda.py
Exploratory Data Analysis utilities for the recipes dataset.
"""

import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Union

def summary_statistics(df: pd.DataFrame, cols: Union[List[str], None] = None) -> pd.DataFrame:
    return df[cols].describe() if cols else df.describe()

def plot_distribution(df: pd.DataFrame, column: str, bins: int = 50) -> None:
    plt.figure(figsize=(8,4))
    plt.hist(df[column].dropna(), bins=bins, edgecolor='k')
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

def plot_scatter(df: pd.DataFrame, x: str, y: str) -> None:
    plt.figure(figsize=(6,6))
    plt.scatter(df[x], df[y], alpha=0.3)
    plt.title(f'{y} vs. {x}')
    plt.xlabel(x)
    plt.ylabel(y)
    plt.tight_layout()
    plt.show()
