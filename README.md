# 439-diet-recommendation-system

# 🍽️ Personalized Recipe Recommendation System

This project is a machine learning–based recipe recommendation system designed to help users find meals that match their health goals and preferences.

## 💡 What It Does

- Recommends recipes based on goals like weight loss, muscle gain, or balanced nutrition.  
- Lets users filter recipes by calories, protein, total cook time, and ingredients to avoid.  
- Computes ingredient-based similarity between recipes using TF–IDF vectors and cosine similarity.
- Groups recipes into macro-nutrient categories (e.g., desserts, mains, snacks) to offer alternative suggestions within each group.  

## 🛠️ How It Works

- The cleaned dataset is stored in AWS RDS (PostgreSQL) with separate `recipes_raw` and `recipes_processed` tables to ensure reproducibility.  
- Feature engineering calculates metrics such as `Calories_per_Serving`, `Protein_per_Serving`, `Complexity_Score`, `Popularity_Score`, and macro ratios like `Protein_to_Calorie`.  
- Computes cosine similarity on TF–IDF ingredient embeddings to identify similar recipe.  
- Utilizes a hybrid recommendation engine combining item–item collaborative filtering, content-based filtering, and a `RandomForestRegressor` trained on engineered features for rating prediction.  
- The `RandomForestRegressor` is trained on 80% of the 522,517 recipes and evaluated on a 20% holdout, using inputs like nutritional and complexity features to fine-tune recommendation rankings.  
