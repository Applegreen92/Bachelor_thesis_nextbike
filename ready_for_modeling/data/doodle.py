import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# Load the data
df = pd.read_csv('combined_city_data.csv')

# Define feature matrix X and target vector y
X = df.drop(columns=['bikes_available', 'datetime', 'lon', 'lat', 'year','station_name'])
y = df['bikes_available']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the models
gbm = GradientBoostingRegressor()
rf = RandomForestRegressor()
xgb = XGBRegressor()

# Train the models
gbm.fit(X_train, y_train)
rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)

# Make predictions
gbm_pred = gbm.predict(X_test)
rf_pred = rf.predict(X_test)
xgb_pred = xgb.predict(X_test)

# Calculate performance metrics
def evaluate_model(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2

gbm_metrics = evaluate_model(y_test, gbm_pred)
rf_metrics = evaluate_model(y_test, rf_pred)
xgb_metrics = evaluate_model(y_test, xgb_pred)

# Print the performance metrics
print("GBM Performance Metrics:")
print(f"RMSE: {gbm_metrics[0]}")
print(f"MAE: {gbm_metrics[1]}")
print(f"R²: {gbm_metrics[2]}")

print("\nRandom Forest Performance Metrics:")
print(f"RMSE: {rf_metrics[0]}")
print(f"MAE: {rf_metrics[1]}")
print(f"R²: {rf_metrics[2]}")

print("\nXGBoost Performance Metrics:")
print(f"RMSE: {xgb_metrics[0]}")
print(f"MAE: {xgb_metrics[1]}")
print(f"R²: {xgb_metrics[2]}")
