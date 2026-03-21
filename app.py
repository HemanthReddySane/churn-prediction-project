from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load trained model
model = joblib.load("model.pkl")

# Load column names (VERY IMPORTANT)
columns = joblib.load("columns.pkl")

app = Flask(__name__)

# Home route
@app.route('/')
def home():
    return "Churn Prediction API is running"

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json  # input from user

        # Create input list in correct order
        input_data = [data.get(col, 0) for col in columns]

        # Convert to numpy array
        features = np.array(input_data).reshape(1, -1)

        # Predict
        prediction = model.predict(features)[0]

        # Convert result
        result = "Churn" if prediction == 1 else "No Churn"

        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)})

# Run app
if __name__ == "__main__":
    app.run(debug=True)