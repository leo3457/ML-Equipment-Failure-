from flask import Flask, request, jsonify
import numpy as np
import joblib

# Load trained model and scaler
model = joblib.load('models/random_forest_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from request
        data = request.get_json()
        features = np.array([[data['temperature'], data['vibration'], data['pressure']]])
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)
        return jsonify({'failure_prediction': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

