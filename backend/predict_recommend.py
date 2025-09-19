from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

# Load trained bundle
bundle = joblib.load("dropout_model.pkl")

clf = bundle["model"]
scaler = bundle["scaler"]
knn = bundle["knn"]
successful = bundle["successful"]
X_columns = bundle["X_columns"]
cat_features = bundle["cat_features"]
engagement_map = bundle["engagement_map"]
ordinal_col = bundle["ordinal_col"]
numeric_features = bundle["numeric_features"]

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Step 1: Convert numeric fields to float safely
    for field in numeric_features:
        if field in data:
            try:
                data[field] = float(data[field])
            except (ValueError, TypeError):
                data[field] = 0.0
        else:
            data[field] = 0.0

    # Step 2: Create DataFrame and reindex to expected columns
    new_df = pd.DataFrame([data])

    # Ensure categorical columns are strings
    for c in cat_features:
        if c in new_df:
            new_df[c] = new_df[c].astype(str)

    # Ensure all expected columns exist
    new_df = new_df.reindex(columns=X_columns, fill_value=0)

    # Step 3: Map Engagement_Level to numeric
    new_df[ordinal_col] = new_df["Engagement_Level"].map(engagement_map).fillna(2).astype(float)

    # Step 4: Model prediction
    pred = int(clf.predict(new_df)[0])
    prob = float(clf.predict_proba(new_df)[0][1])

    # Step 5: KNN nearest peers
    X_scaled = scaler.transform(new_df[numeric_features + [ordinal_col]])
    distances, indices = knn.kneighbors(X_scaled, n_neighbors=5)
    peer_group = successful.iloc[indices[0]]

    # Step 6: Generate actionable suggestions
    peer_means = peer_group[numeric_features].mean()
    suggestions = []

    # Customize friendly messages for specific fields
    field_messages = {
        "Time_Spent_on_Videos": "Increase time spent on course videos",
        "Forum_Participation": "Engage more in forum discussions",
        "Assignment_Completion_Rate": "Complete assignments consistently and on time",
        "Feedback_Score": "Share/seek constructive feedback more regularly",
        "Quiz_Attempts": "Attempt quizzes more frequently",
        "Quiz_Scores": "Improve quiz scores",
        "Age": "Consider age-related learning strategies"
    }

    for feature, message in field_messages.items():
        student_value = float(new_df[feature].iloc[0])
        peer_value = peer_means[feature]
        # Suggest only if student's value is significantly below peers (10% threshold)
        if student_value < peer_value * 0.9:
            suggestions.append(f"- {message} (peers avg: {peer_value:.2f}, yours: {student_value:.2f})")

    return jsonify({
        "prediction": pred,
        "probability": prob,
        "suggestions": suggestions
    })

if __name__ == "__main__":
    app.run(port=5000, debug=True)
