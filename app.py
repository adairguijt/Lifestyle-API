from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("lifestyle_model.pkl")

actionable_features = {
    "SUPPORTING_OTHERS": {"direction": "increase", "step": 1, "max": 10},
    "TODO_COMPLETED": {"direction": "increase", "step": 1, "max": 10},
    "TIME_FOR_PASSION": {"direction": "increase", "step": 1, "max": 10},
    "SUFFICIENT_INCOME": {"direction": "target", "step": 1, "target": 2},
    "CORE_CIRCLE": {"direction": "increase", "step": 1, "max": 10},
    "WEEKLY_MEDITATION": {"direction": "increase", "step": 1, "max": 10},
    "DAILY_STEPS": {"direction": "increase", "step": 1000, "max": 10000},
    "LIFE_VISION": {"direction": "increase", "step": 1, "max": 10},
    "BMI_RANGE": {"direction": "target", "step": 1, "target": 1},
    "FRUITS_VEGGIES": {"direction": "increase", "step": 1, "max": 5},
    "DAILY_STRESS": {"direction": "decrease", "step": 1, "min": 1},
    "DAILY_SHOUTING": {"direction": "decrease", "step": 1, "min": 0}
}

def generate_optimized_advice(user_answers, model, actionable_features, top_n=3):
    current_input = pd.DataFrame([user_answers])
    current_score = model.predict(current_input)[0]

    recommendations = []

    for feature, rule in actionable_features.items():
        if feature not in user_answers:
            continue

        new_user = user_answers.copy()
        current_value = user_answers[feature]

        if rule["direction"] == "increase":
            new_value = min(current_value + rule["step"], rule["max"])
        elif rule["direction"] == "decrease":
            new_value = max(current_value - rule["step"], rule["min"])
        elif rule["direction"] == "target":
            target = rule["target"]
            if current_value < target:
                new_value = current_value + rule["step"]
            elif current_value > target:
                new_value = current_value - rule["step"]
            else:
                continue

        if new_value == current_value:
            continue

        new_user[feature] = new_value
        new_score = model.predict(pd.DataFrame([new_user]))[0]
        improvement = new_score - current_score

        if improvement > 0:
            recommendations.append({
                "feature": feature,
                "current_value": current_value,
                "suggested_value": new_value,
                "estimated_score_gain": round(float(improvement), 2),
                "new_estimated_score": round(float(new_score), 2)
            })

    recommendations = sorted(
        recommendations,
        key=lambda x: x["estimated_score_gain"],
        reverse=True
    )

    return {
        "current_lifestyle_score": round(float(current_score), 2),
        "best_recommendations": recommendations[:top_n]
    }

@app.route("/")
def home():
    return {"message": "Lifestyle Score API is running"}

@app.route("/predict", methods=["POST"])
def predict():
    user_answers = request.get_json()

    result = generate_optimized_advice(
        user_answers=user_answers,
        model=model,
        actionable_features=actionable_features,
        top_n=3
    )

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)