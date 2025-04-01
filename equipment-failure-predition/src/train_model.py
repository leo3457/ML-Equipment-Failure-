import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

# Load dataset (replace with actual file path)
data_path = 'data/sensor_data.csv'
df = pd.read_csv(data_path)

# Handle missing values (forward-fill, median imputation as fallback)
df.fillna(method='ffill', inplace=True)
df.fillna(df.median(), inplace=True)

# Feature Engineering: Rolling averages and lag features
window_size = 5
df['temp_avg'] = df['temperature'].rolling(window=window_size).mean()
df['vibration_avg'] = df['vibration'].rolling(window=window_size).mean()
df['pressure_avg'] = df['pressure'].rolling(window=window_size).mean()
df.fillna(df.median(), inplace=True)  # Fill NaN values after rolling

# Define features and target variable
features = ['temperature', 'vibration', 'pressure', 'temp_avg', 'vibration_avg', 'pressure_avg']
target = 'failure'
X = df[features]
y = df[target]

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Random Forest Classifier with hyperparameter tuning
param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='recall')  # Optimizing for recall
grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_

# Cross-validation to check model stability
cv_scores = cross_val_score(best_rf, X_train, y_train, cv=5, scoring='recall')
print(f'Cross-validation recall scores: {cv_scores}')
print(f'Mean cross-validation recall: {cv_scores.mean():.2f}')

# Save the trained model
os.makedirs('models', exist_ok=True)
joblib.dump(best_rf, 'models/random_forest_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

# Model Evaluation
y_pred = best_rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

# Print evaluation metrics
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1-Score: {f1:.2f}')
print(f'ROC-AUC Score: {roc_auc:.2f}')

