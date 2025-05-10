# pipeline.py
"""
pipeline.py
Orchestrates the full data workflow.
"""

import argparse
from data_loader import load_csv, load_database
from preprocessor import preprocess
from feature_engineer import engineer_all

def main(args):
    if args.db:
        df = load_database(args.conn, args.table)
    else:
        df = load_csv(args.input)
    print(f"Loaded {len(df)} rows.")
    df_clean = preprocess(df)
    print("Preprocessing complete.")
    df_feat = engineer_all(df_clean)
    print("Feature engineering complete.")
    df_feat.to_csv(args.output, index=False)
    print(f"Processed data saved to {args.output}.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='recipes.csv')
    parser.add_argument('--output', type=str, default='recipes_processed.csv')
    parser.add_argument('--db', action='store_true')
    parser.add_argument('--conn', type=str)
    parser.add_argument('--table', type=str, default='recipes_raw')
    args = parser.parse_args()
    main(args)
