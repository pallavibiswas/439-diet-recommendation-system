# model_builder.py
"""
model_builder.py
Train a regression model to predict recipe ratings, evaluate, and optionally save the trained model.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from evaluator import regression_metrics, top_k_accuracy
import pickle
from typing import Tuple, List, Optional

def load_features_and_target(path: str, feature_cols: List[str], target_col: str = 'AggregatedRating') -> Tuple[pd.DataFrame,pd.Series]:
    df = pd.read_csv(path, index_col=0)
    X = df[feature_cols]
    y = df[target_col]
    return X, y

def train_and_evaluate(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42, top_k: int = 10, save_model: Optional[str] = None) -> RandomForestRegressor:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    model = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)
    print("Training RandomForestRegressor...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse, rmse = regression_metrics(y_test.values, y_pred)
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    y_test_series = pd.Series(y_test.values, index=y_test.index)
    y_pred_series = pd.Series(y_pred, index=y_test.index)
    acc = top_k_accuracy(y_test_series, y_pred_series, k=top_k)
    print(f"Top-{top_k} ranking accuracy: {acc:.2%}")
    if save_model:
        with open(save_model, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved to {save_model}")
    return model

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Train & evaluate recipe rating predictor")
    parser.add_argument('--input', type=str, default='recipes_processed.csv')
    parser.add_argument('--features', nargs='+', required=True)
    parser.add_argument('--target', type=str, default='AggregatedRating')
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--save_model', type=str, default=None)
    args = parser.parse_args()
    X, y = load_features_and_target(args.input, args.features, args.target)
    print(f"Loaded {len(X)} recipes, {len(args.features)} features.")
    train_and_evaluate(X, y, top_k=args.top_k, save_model=args.save_model)
