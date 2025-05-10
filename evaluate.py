import argparse
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from evaluator import regression_metrics, top_k_accuracy
from model_builder import load_features_and_target

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate regression model with MSE, RMSE, and Top-K accuracy."
    )
    parser.add_argument('--input', type=str, default='recipes_processed.csv',
                        help='Path to processed dataset CSV')
    parser.add_argument('--model', type=str, default='rating_model.pkl',
                        help='Path to pickled trained model')
    parser.add_argument('--features', nargs='+', required=True,
                        help='List of feature column names')
    parser.add_argument('--target', type=str, default='AggregatedRating',
                        help='Target column name')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Fraction of data to use for testing')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--top_k', type=int, default=20,
                        help='k for Top-K ranking accuracy')
    return parser.parse_args()

# -------- Main Evaluation --------
def main():
    args = parse_args()
    print(f"Loading data from {args.input}...")
    X, y = load_features_and_target(args.input, args.features, args.target)

    print(f"Loading model from {args.model}...")
    with open(args.model, 'rb') as f:
        model = pickle.load(f)

    # Split into train/test for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.random_state
    )

    # Predict on test set
    print("Predicting on test set...")
    y_pred = model.predict(X_test)

    # Compute regression metrics
    mse, rmse = regression_metrics(y_test.values, y_pred)
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")

    # Compute Top-K ranking accuracy
    y_true_series = pd.Series(y_test.values, index=y_test.index)
    y_pred_series = pd.Series(y_pred, index=y_test.index)
    acc = top_k_accuracy(y_true_series, y_pred_series, k=args.top_k)
    print(f"Top-{args.top_k} ranking accuracy: {acc:.2%}")

if __name__ == '__main__':
    main()