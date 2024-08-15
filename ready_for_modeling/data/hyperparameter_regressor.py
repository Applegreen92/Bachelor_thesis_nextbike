import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the training data
train_df = pd.read_csv('combined_city_data.csv')

# Define feature columns and target column
selected_features = ['lon', 'lat', 'hour', 'month', 'weekday', 'is_weekend', 'is_holiday', 'temperature', 'sfcWind',
                     'precipitation']
target_col = 'bikes_available'

# Separate features and target variable from training data
X_train = train_df[selected_features]
y_train_reg = train_df[target_col].clip(upper=40)  # Cap the target values at 40

# Define the parameter distribution for hyperparameter tuning
param_dist = {
    'n_estimators': np.arange(50, 201, 20),  # Range from 50 to 200 with step size of 10
    'max_depth': [None] + list(np.arange(10, 31, 5)),  # None or between 10 and 30 with step size of 5
    'min_samples_split': np.arange(2, 11),  # Between 2 and 10
    'min_samples_leaf': np.arange(1, 5),  # Between 1 and 4
    'max_features': ['auto', 'sqrt', 'log2', None],
    'bootstrap': [True, False]
}

# Initialize the RandomForestRegressor
reg = RandomForestRegressor(random_state=42)

# Set up RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=reg,
                                   param_distributions=param_dist,
                                   n_iter=200,  # 150 different random combinations
                                   scoring='neg_mean_squared_error',
                                   n_jobs=-1,  # Use all available cores
                                   random_state=42,  # For reproducibility
                                   verbose=2)

# Fit RandomizedSearchCV to the training data
random_search.fit(X_train, y_train_reg)

# Get the best hyperparameters
best_params = random_search.best_params_
print(f"Best Hyperparameters: {best_params}")

# Train the RandomForestRegressor with the best hyperparameters
best_reg = RandomForestRegressor(**best_params, random_state=3)
best_reg.fit(X_train, y_train_reg)

# Evaluate the model on a test set (example)
test_df = pd.read_csv('2022_combined_city_data.csv')
X_test = test_df[selected_features]
y_test_reg = test_df[target_col].clip(upper=40)

# Make predictions and calculate MSE
reg_predictions = best_reg.predict(X_test)
reg_mse = mean_squared_error(y_test_reg, reg_predictions)

print(f'Regressor Mean Squared Error with best hyperparameters: {reg_mse:.4f}')

# Save the best result to a CSV file
best_result = {
    'Best Hyperparameters': [best_params],
    'Test_MSE': [reg_mse]
}

best_result_df = pd.DataFrame(best_result)
best_result_df.to_csv('best_hyperparameter_result_random_search.csv', index=False)

print("Best result saved to 'best_hyperparameter_result_random_search.csv'.")
