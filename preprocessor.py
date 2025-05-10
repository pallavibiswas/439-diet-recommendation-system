# preprocessor.py
"""
preprocessor.py
Cleans and transforms raw recipe data.
"""

import pandas as pd
import re
from typing import List

def drop_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    return df.drop(columns=cols, errors='ignore')

def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    # Drop rows without instructions
    df = df.dropna(subset=['RecipeInstructions'])
    # Fill strings
    str_cols = df.select_dtypes(include='object').columns
    df[str_cols] = df[str_cols].fillna('')
    # Coerce numeric nutrition
    num_cols = [
        'AggregatedRating','ReviewCount','Calories','FatContent',
        'SaturatedFatContent','CholesterolContent','SodiumContent',
        'CarbohydrateContent','FiberContent','SugarContent',
        'ProteinContent','RecipeServings'
    ]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    return df

def parse_duration(pt: str) -> int:
    if not pt or not pt.startswith('PT'):
        return 0
    hrs = re.search(r'PT(\d+)H', pt)
    mins = re.search(r'(\d+)M', pt)
    total = 0
    if hrs:
        total += int(hrs.group(1)) * 60
    if mins:
        total += int(mins.group(1))
    return total

def parse_r_list(s: str) -> List[str]:
    if s.startswith('c("') and s.endswith('")'):
        inner = s[3:-2]
        return [item.strip().strip('"') for item in inner.split('", "')]
    return [s] if s else []

def transform_times_and_lists(df: pd.DataFrame) -> pd.DataFrame:
    for col in ['CookTime','PrepTime','TotalTime']:
        df[col + '_Minutes'] = df[col].apply(parse_duration)
    for col in ['RecipeIngredientQuantities','RecipeIngredientParts','RecipeInstructions']:
        df[col + '_List'] = df[col].apply(parse_r_list)
    return df

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = [
        'RecipeId','AuthorId','AuthorName','DatePublished','Images','Keywords'
    ]
    df = drop_columns(df, cols_to_drop)
    df = fill_missing(df)
    df = transform_times_and_lists(df)
    return df

