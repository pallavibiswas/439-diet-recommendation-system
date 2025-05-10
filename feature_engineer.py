# feature_engineer.py
"""
feature_engineer.py
Generates derived features for modeling and analysis.
"""

import pandas as pd
import numpy as np

def add_nutrient_per_serving(df: pd.DataFrame) -> pd.DataFrame:
    df['Calories_per_Serving'] = df['Calories'] / df['RecipeServings'].replace(0, np.nan)
    df['Protein_per_Serving'] = df['ProteinContent'] / df['RecipeServings'].replace(0, np.nan)
    df['Fat_per_Serving']     = df['FatContent']    / df['RecipeServings'].replace(0, np.nan)
    df['Carbs_per_Serving']   = df['CarbohydrateContent'] / df['RecipeServings'].replace(0, np.nan)
    return df

def add_macro_ratios(df: pd.DataFrame) -> pd.DataFrame:
    df['Protein_to_Calorie'] = df['ProteinContent'] / df['Calories'].replace(0, np.nan)
    df['Fat_to_Calorie']     = df['FatContent']    / df['Calories'].replace(0, np.nan)
    df['Carb_to_Calorie']    = df['CarbohydrateContent'] / df['Calories'].replace(0, np.nan)
    return df

def add_complexity_and_popularity(df: pd.DataFrame) -> pd.DataFrame:
    df['Ingredient_Count'] = df['RecipeIngredientParts_List'].apply(len)
    df['Num_Steps']        = df['RecipeInstructions_List'].apply(len)
    df['Complexity_Score'] = (df['TotalTime_Minutes'] / 10).fillna(0) + df['Ingredient_Count'] + df['Num_Steps']
    df['Popularity_Score'] = df['AggregatedRating'] * np.log1p(df['ReviewCount'])
    return df

def engineer_all(df: pd.DataFrame) -> pd.DataFrame:
    df = add_nutrient_per_serving(df)
    df = add_macro_ratios(df)
    df = add_complexity_and_popularity(df)
    return df
