# Predicting Equipment Failure Using Sensor Data

## Project Overview
This project aims to predict equipment failures using historical sensor data. By training a machine learning model on time-series sensor readings, we can identify potential failures before they happen, allowing for proactive maintenance and minimizing downtime.

## Features
- **Data Preprocessing**: Handles missing values, normalizes data, and engineers features (rolling averages, lag features).
- **Machine Learning Model**: Uses a Random Forest classifier with hyperparameter tuning and cross-validation.
- **Model Evaluation**: Measures accuracy, precision, recall, F1-score, and ROC-AUC.
- **Deployment**: Provides a Flask API for real-time failure prediction.

## Installation
### Prerequisites
- Python 3.8+
- pip

### Setup
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/equipment-failure-prediction.git
   cd equipment-failure-prediction
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure the dataset is available in `data/sensor_data.csv`.
4. Train the model:
   ```bash
   python src/train_model.py
   ```

## Usage
### Predict Equipment Failure
Start the Flask API:
```bash
python src/app.py
```
Send a prediction request:
```bash
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"temperature": 75, "vibration": 0.5, "pressure": 30}'
```

## Project Structure
```
├── data/               # Folder for datasets
├── models/             # Saved trained models
├── notebooks/          # Jupyter notebooks for exploration
├── src/                # Source code
│   ├── train_model.py  # Model training script
│   ├── app.py          # Flask API for predictions
├── README.md           # Project documentation
├── requirements.txt    # Dependencies
```

## Future Improvements
- Integrate real-time streaming data
- Experiment with deep learning models (LSTMs)
- Improve feature selection for better model performance

## Contributors
- **Leo Liao**

