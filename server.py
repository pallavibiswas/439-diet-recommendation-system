"""
server.py
Flask API for rating predictions and top-K recommendations based solely on recipe features.
"""

from flask import Flask, request, jsonify
import pandas as pd
import pickle
import os

# -------- Configuration --------
DATA_PATH = 'recipes_processed.csv'
MODEL_PATH = 'rating_model.pkl'
FEATURE_COLS = [
    'Calories_per_Serving', 'Protein_per_Serving', 'Fat_per_Serving',
    'Carbs_per_Serving', 'Complexity_Score', 'Popularity_Score'
]

# -------- App Setup --------
app = Flask(__name__)

# Load processed recipes and reset to integer index
print(f"Loading data from {DATA_PATH}...")
df = pd.read_csv(DATA_PATH)
df.reset_index(drop=True, inplace=True)

# Load or train model
if os.path.exists(MODEL_PATH):
    print(f"Loading model from {MODEL_PATH}...")
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
else:
    from model_builder import load_features_and_target, train_and_evaluate
    X, y = load_features_and_target(DATA_PATH, FEATURE_COLS)
    model = train_and_evaluate(X, y, save_model=MODEL_PATH)

@app.route('/')
def index():
    """Health check endpoint"""
    return jsonify({
        'message': 'Diet Recommendation API is running',
        'endpoints': ['/predict', '/recommend']
    })

@app.route('/predict')
def predict():
    """Predict rating for a single recipe by its zero-based row index."""
    rid = request.args.get('recipe_id', type=int)
    if rid is None:
        return jsonify(error="Missing 'recipe_id' parameter"), 400
    if rid < 0 or rid >= len(df):
        return jsonify(error=f"Recipe {rid} not found"), 404

    x = df.loc[[rid], FEATURE_COLS]
    pred = float(model.predict(x)[0])
    return jsonify(recipe_id=rid, predicted_rating=round(pred, 4))

@app.route('/recommend')
def recommend():
    """Return top-K recipe indices by predicted rating."""
    k = request.args.get('k', default=10, type=int)
    X = df[FEATURE_COLS]
    preds = model.predict(X)
    top_ids = preds.argsort()[::-1][:k].tolist()
    return jsonify(top_k=top_ids)

if __name__ == '__main__':
    # Bind to localhost and disable the reloader so you see startup prints
    print("Starting Flask server on http://127.0.0.1:5001 …")
    app.run(host='127.0.0.1', port=5001, debug=True, use_reloader=False)
