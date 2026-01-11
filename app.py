from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd

# -------------------- Load Model & Columns --------------------

with open("model/credit_risk_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

# -------------------- Initialize App --------------------

app = Flask(__name__)
CORS(app)   # VERY IMPORTANT for frontend-backend connection

# -------------------- Prediction Logic --------------------

def predict_credit_risk(input_data):
    df = pd.DataFrame([input_data])

    numeric_cols = [
        "person_age",
        "person_income",
        "person_emp_length",
        "loan_amnt",
        "loan_int_rate",
        "loan_percent_income",
        "cb_person_cred_hist_length"
    ]

    for col in numeric_cols:
        df[col] = df[col].fillna(0)

    # One-hot encoding
    df_encoded = pd.get_dummies(df, drop_first=True)
    df_encoded = df_encoded.reindex(columns=feature_columns, fill_value=0)

    # Predict probability
    risk_prob = model.predict_proba(df_encoded)[0][1]

    # Decision thresholds
    if risk_prob < 0.3:
        decision = "Approve"
    elif risk_prob <= 0.6:
        decision = "Manual Review"
    else:
        decision = "Reject"

    return {
        "risk_probability": round(float(risk_prob), 2),
        "decision": decision
    }

# -------------------- Routes --------------------

@app.route("/", methods=["GET"])
def home():
    return "Credit Risk Prediction API is running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    result = predict_credit_risk(data)
    return jsonify(result)

# -------------------- Run Server --------------------

if __name__ == "__main__":
    app.run(debug=True)

