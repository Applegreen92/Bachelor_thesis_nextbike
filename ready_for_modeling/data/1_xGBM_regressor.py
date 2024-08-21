import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# Load the training data
train_df = pd.read_csv('test_data/new_combined_city_data.csv')

# Define feature columns and target column
selected_features = ['lat', 'lon', 'hour', 'weekday', 'precipitation']
target_col = 'bikes_available'

# Separate features and target variable from training data
X_train = train_df[selected_features]
y_train_reg = train_df[target_col]  # Cap the target values at 40

# Initialize XGBoost Regressor with given hyperparameters
reg_xgb = XGBRegressor(
    colsample_bytree=0.6977924525180372,
    gamma=4.897754201908785,
    learning_rate=0.03899301447596341,
    max_depth=9,
    min_child_weight=5,
    n_estimators=265,
    random_state=42
)

# Fit the regressor on the training data
reg_xgb.fit(X_train, y_train_reg)

# List of test CSV files
test_files = [
    #'2022_combined_city_data.csv',
    #'2022_complete_dresden.csv',
    #'2022_complete_heidelberg.csv',
    #'2022_complete_essen.csv',
    '2022_complete_nürnberg.csv',
]

# Initialize lists to store results for plotting
cities = []
reg_rmses = []

# Loop through each test file, evaluate the model, and store results
for test_file in test_files:
    print(f"\nEvaluating on {test_file}...")

    # Load the testing data
    test_df = pd.read_csv(test_file)

    # Separate features and target variable from testing data
    X_test = test_df[selected_features]
    y_test_reg = test_df[target_col]

    # Make predictions on the testing data
    reg_xgb_predictions = reg_xgb.predict(X_test)

    # Calculate the RMSE for the regressor
    reg_xgb_rmse = np.sqrt(mean_squared_error(y_test_reg, reg_xgb_predictions))

    # Store the results
    city_name = test_file.split('/')[-1].split('.')[0]
    cities.append(city_name)
    reg_rmses.append(reg_xgb_rmse)

    # Print results
    print(f'Results for {test_file}:')
    print(f'XGBoost Regressor RMSE: {reg_xgb_rmse:.4f}')

