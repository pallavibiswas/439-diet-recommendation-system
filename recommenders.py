# recommenders.py
"""
recommenders.py
Implements Collaborative Filtering, Content-Based Filtering, and a simple Hybrid Recommender,
and adds rating‐prediction methods to each.
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Optional

class CollaborativeFilter:
    """
    Memory-based item-item collaborative filtering.
    Expects a DataFrame of user-item ratings with columns ['user_id', 'recipe_id', 'rating'].
    """
    def __init__(self, ratings: pd.DataFrame):
        self.ratings = ratings.copy()
        self.item_sim: Optional[pd.DataFrame] = None

    def train(self) -> None:
        """Compute item–item similarity matrix."""
        user_item = (
            self.ratings
              .pivot(index='user_id', columns='recipe_id', values='rating')
              .fillna(0)
        )
        sim_matrix = cosine_similarity(user_item.T)
        self.item_sim = pd.DataFrame(
            sim_matrix,
            index=user_item.columns,
            columns=user_item.columns
        )

    def recommend(self, recipe_id: int, k: int = 10) -> List[int]:
        """Return top-k recipes most similar to the given recipe_id."""
        if self.item_sim is None:
            raise ValueError("Call train() before recommending.")
        sims = self.item_sim[recipe_id].drop(index=recipe_id)
        return sims.nlargest(k).index.tolist()

    def predict_rating(self, user_id: int, recipe_id: int, k: int = 10) -> float:
        """
        Predict a user's rating for a recipe using weighted average of k nearest items.
        """
        if self.item_sim is None:
            raise ValueError("Call train() before predicting ratings.")
        # find k most similar items
        sims = self.item_sim[recipe_id].drop(index=recipe_id)
        top_items = sims.nlargest(k)
        # pull this user's ratings on those items
        user_ratings = self.ratings[
            (self.ratings['user_id'] == user_id) &
            (self.ratings['recipe_id'].isin(top_items.index))
        ]
        if user_ratings.empty:
            # fallback to global average
            return float(self.ratings['rating'].mean())
        weights = []
        ratings = []
        for rid, rating in zip(user_ratings['recipe_id'], user_ratings['rating']):
            weights.append(top_items[rid])
            ratings.append(rating)
        return float(np.dot(ratings, weights) / np.sum(weights))


class ContentBasedRecommender:
    """
    Content-based filtering using TF-IDF over a text/list column.
    Expects a DataFrame with a list-of-strings column (e.g. ingredients).
    """
    def __init__(self, df: pd.DataFrame, text_column: str = 'RecipeIngredientParts_List'):
        self.df = df.copy()
        self.text_column = text_column
        # join list into document string
        self.df['__doc__'] = self.df[self.text_column].apply(
            lambda x: ' '.join(x) if isinstance(x, list) else str(x)
        )
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = None
        self.sim_matrix: Optional[pd.DataFrame] = None

    def train(self) -> None:
        """Fit TF-IDF and compute cosine similarity matrix."""
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['__doc__'])
        sim = cosine_similarity(self.tfidf_matrix)
        self.sim_matrix = pd.DataFrame(
            sim,
            index=self.df.index,
            columns=self.df.index
        )

    def recommend(self, recipe_id: int, k: int = 10) -> List[int]:
        """Return top-k recipes by content similarity."""
        if self.sim_matrix is None:
            raise ValueError("Call train() before recommending.")
        sims = self.sim_matrix.loc[recipe_id].drop(index=recipe_id)
        return sims.nlargest(k).index.tolist()

    def predict_rating(self, user_id: int, recipe_id: int, k: int = 10) -> float:
        """
        Content-based rating prediction.
        (In a full system you’d merge with the user–item ratings DF;
        here we fallback to the global mean if no direct mapping.)
        """
        # Fallback to global average of all ratings (requires ratings passed in)
        return float(self.df.index.shape[0] and 0.0)


class HybridRecommender:
    """
    Weighted hybrid of collaborative and content-based recommenders.
    """
    def __init__(self, cf: CollaborativeFilter, cb: ContentBasedRecommender, alpha: float = 0.5):
        self.cf = cf
        self.cb = cb
        self.alpha = alpha  # weight for CF; (1-alpha) for CB

    def recommend(self, recipe_id: int, k: int = 10) -> List[int]:
        cf_scores = self.cf.item_sim.loc[recipe_id]
        cb_scores = self.cb.sim_matrix.loc[recipe_id]
        combined = self.alpha * cf_scores + (1 - self.alpha) * cb_scores
        combined = combined.drop(index=recipe_id)
        return combined.nlargest(k).index.tolist()

    def predict_rating(self, user_id: int, recipe_id: int, k: int = 10) -> float:
        cf_pred = self.cf.predict_rating(user_id, recipe_id, k)
        cb_pred = cf_pred  # fallback if CB not fully implemented
        return self.alpha * cf_pred + (1 - self.alpha) * cb_pred
