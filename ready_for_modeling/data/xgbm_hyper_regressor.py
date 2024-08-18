import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import uniform, randint

# Load the training data
train_df = pd.read_csv('combined_city_data.csv')

# Define feature columns and target column
selected_features = ['city_lat','city_lng','bike_racks','lon', 'lat', 'hour', 'month', 'weekday', 'is_weekend', 'is_holiday', 'temperature', 'sfcWind',
                     'precipitation']
target_col = 'bikes_available'

# Separate features and target variable from training data
X_train = train_df[selected_features]
y_train = train_df[target_col]

# Initialize XGBoost Regressor
reg_xgb = XGBRegressor(
    random_state=3
)

# Define the parameter grid to search
param_dist = {
    'n_estimators': randint(50, 300),
    'learning_rate': uniform(0.01, 0.2),
    'max_depth': randint(3, 15),
    'subsample': uniform(0.5, 0.49),
    'colsample_bytree': uniform(0.5, 0.4),
    'gamma': uniform(0, 5),
    'min_child_weight': randint(1, 10),
}

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=reg_xgb,
    param_distributions=param_dist,
    n_iter=150,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    cv= 3,
    verbose=1,
    random_state=42
)

# Fit the RandomizedSearchCV
random_search.fit(X_train, y_train)

# Print the best parameters and the best score
print(f"Best Parameters: {random_search.best_params_}")
print(f"Best Cross-Validation MSE: {-random_search.best_score_:.4f}")

# Use the best estimator to predict on the test set
best_reg_xgb = random_search.best_estimator_

# List of test CSV files
test_files = [
    '2022_complete_nürnberg.csv',
]

# Loop through each test file, evaluate the model, and print results
for test_file in test_files:
    print(f"\nEvaluating on {test_file}...")

    # Load the testing data
    test_df = pd.read_csv(test_file)

    # Separate features and target variable from testing data
    X_test = test_df[selected_features]
    y_test = test_df[target_col]

    # Make predictions on the testing data using the best model
    reg_xgb_predictions = best_reg_xgb.predict(X_test)

    # Calculate the mean squared error for the regressor
    reg_xgb_mse = mean_squared_error(y_test, reg_xgb_predictions)

    # Print results
    print(f'XGBoost Regressor MSE with Tuned Hyperparameters: {reg_xgb_mse:.4f}')
