# ============================
# Predict + Collaborative Filtering Recommender
# ============================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier


# 1) Load dataset
df = pd.read_csv("dropout_dataset.csv")

# 2) Feature definitions
# Categorical cols for CatBoost (strings)
cat_features = ["Gender", "Education_Level", "Course_Name", "Engagement_Level", "Learning_Style"]

# Numeric cols (used by CatBoost as numeric AND for similarity)
numeric_features = [
    "Age",
    "Time_Spent_on_Videos",
    "Quiz_Attempts",
    "Quiz_Scores",
    "Forum_Participation",
    "Assignment_Completion_Rate",
    "Feedback_Score"
    # Note: We exclude Final_Exam_Score from X (as in your code)
]

target = "Dropout_Likelihood"

# For collaborative filtering similarity, we want an ordinal engagement
engagement_map = {"Low": 1, "Medium": 2, "High": 3}
ordinal_col = "Engagement_Level_ordinal"
numeric_features_sim = numeric_features + [ordinal_col]


# 3) Prepare data for CatBoost
X = df[numeric_features + cat_features].copy()
y = df[target].copy()

# Ensure categoricals are strings
for c in cat_features:
    X[c] = X[c].astype(str)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)


# 4) Train CatBoost predictor
clf = CatBoostClassifier(
    iterations=1500,
    depth=8,
    learning_rate=0.05,
    eval_metric="F1",
    loss_function="Logloss",
    cat_features=cat_features,
    random_seed=42,
    verbose=200,
    early_stopping_rounds=100
)

clf.fit(X_train, y_train, eval_set=(X_test, y_test))


# 5) Build Collaborative Filtering on successful peers
successful = df[df[target] == 0].copy()
if successful.empty:
    # Fallback: if your dataset has no successful examples (shouldn’t happen in balanced set)
    successful = df.copy()

# Build ordinal engagement for similarity
successful[ordinal_col] = successful["Engagement_Level"].map(engagement_map).fillna(2).astype(float)

# Standardize similarity features to avoid scale issues
scaler = StandardScaler()
successful_sim_matrix = scaler.fit_transform(successful[numeric_features_sim])

# Fit KNN (euclidean in standardized space)
k_neighbors = min(10, len(successful))  # cap neighbors to dataset size
knn = NearestNeighbors(n_neighbors=k_neighbors, metric="euclidean")
knn.fit(successful_sim_matrix)


# 6) Action dictionary & helper
action_text = {
    "Time_Spent_on_Videos": "Increase time spent on course videos",
    "Quiz_Attempts": "Attempt more quizzes for spaced practice",
    "Quiz_Scores": "Improve quiz accuracy via revision & feedback",
    "Forum_Participation": "Engage more in forum discussions",
    "Assignment_Completion_Rate": "Complete assignments consistently and on time",
    "Feedback_Score": "Share/seek constructive feedback more regularly",
    "Age": "Maintain steady study habits suitable for your level"  # usually not actionable; included for completeness
}

def _to_model_df(user_dict: dict, X_columns):
    """Align a new user dict to model's expected columns/order."""
    df_in = pd.DataFrame([user_dict]).copy()
    # Ensure categorical as string
    for c in cat_features:
        df_in[c] = df_in[c].astype(str)
    # Reindex & fill any missing numerics with column means (safe default)
    aligned = df_in.reindex(columns=X_columns)
    # If any numeric missing, fill with training means (from original df)
    for col in numeric_features:
        if col in aligned.columns and aligned[col].isna().any():
            aligned[col] = aligned[col].fillna(df[col].mean())
    return aligned

def _peer_based_recommendations(user_df_row: pd.Series, neighbors_df: pd.DataFrame, margin: float = 0.05):
    """
    Compare user vs. neighbors' average and produce targeted actions.
    margin = tolerance band (5%): only recommend if clearly below peers.
    """
    recs = []

    peer_avg = neighbors_df[numeric_features].mean()

    for col in numeric_features:
        user_val = float(user_df_row[col])
        peer_val = float(peer_avg[col])
        # Recommend if user is below peers by > margin
        if peer_val > 0 and user_val < peer_val * (1 - margin):
            if col in action_text:
                recs.append(f"{action_text[col]} (peers avg: {peer_val:.2f}, yours: {user_val:.2f})")

    if not recs:
        recs.append("Keep up the good work! Your study profile matches successful peers.")
    return recs


# 7) Main function: predict + recommend
def predict_and_recommend(new_student: dict, k: int = 5, margin: float = 0.05):
    """
    new_student: dict with keys covering numeric_features + cat_features
                 Example keys:
                 Gender, Education_Level, Course_Name, Engagement_Level, Learning_Style,
                 Age, Time_Spent_on_Videos, Quiz_Attempts, Quiz_Scores, Forum_Participation,
                 Assignment_Completion_Rate, Feedback_Score
    k: number of nearest successful peers to compare against
    margin: tolerance when deciding if a metric is below peers
    """
    # 1) Predict dropout
    new_df = _to_model_df(new_student, X.columns)
    pred = int(clf.predict(new_df)[0])
    prob = float(clf.predict_proba(new_df)[0][1])  # probability of dropout (class=1)

    # 2) Prepare vector for similarity (standardized)
    #    Build ordinal engagement for the input
    eng_val = new_df["Engagement_Level"].iloc[0]
    eng_ord = float(engagement_map.get(str(eng_val), 2))
    sim_vec_raw = new_df[numeric_features].iloc[0].copy()
    sim_vec = pd.concat([sim_vec_raw, pd.Series({ordinal_col: eng_ord})], axis=0)

    sim_vec_scaled = scaler.transform([sim_vec.values])
    n_use = min(k, len(successful))  # safe cap
    distances, indices = knn.kneighbors(sim_vec_scaled, n_neighbors=n_use)
    neighbors = successful.iloc[indices[0]]

    # 3) Generate recommendations versus nearest successful peers
    recs = _peer_based_recommendations(new_df.iloc[0], neighbors, margin=margin)

    return {
        "dropout_prediction": pred,
        "dropout_probability": prob,
        "neighbors_count": n_use,
        "recommendations": recs
    }


# 8) Example usage
example_student = {
    # Categorical
    "Gender": "Male",
    "Education_Level": "Undergraduate",
    "Course_Name": "Data Science",
    "Engagement_Level": "Low",
    "Learning_Style": "Visual",
    # Numeric
    "Age": 20,
    "Time_Spent_on_Videos": 18,
    "Quiz_Attempts": 2,
    "Quiz_Scores": 58,
    "Forum_Participation": 1,
    "Assignment_Completion_Rate": 55,
    "Feedback_Score": 3
}

result = predict_and_recommend(example_student, k=5, margin=0.05)

print(f"Predicted Dropout Likelihood: {result['dropout_prediction']} "
      f"(probability of dropout: {result['dropout_probability']:.3f})")
print(f"Compared against {result['neighbors_count']} nearest successful peers.\n")
print("Personalized Recommendations:")
for r in result["recommendations"]:
    print("-", r)
