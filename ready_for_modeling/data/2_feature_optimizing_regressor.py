import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the training data
train_df = pd.read_csv('test_data/new_combined_city_data.csv')

# Define all potential feature columns and the target column
all_features = ['lon', 'lat', 'hour', 'month', 'weekday', 'is_weekend',
                'is_holiday', 'temperature', 'sfcWind', 'precipitation']
target_col = 'bikes_available'

# Separate features and target variable from training data
X_train = train_df[all_features]
y_train = train_df[target_col]

# Load the validation dataset (second CSV file)
test_df = pd.read_csv('test_data/new_2022_combined_city_data.csv')
X_val = test_df[all_features]
y_val = test_df[target_col]

# Initialize variables for forward feature selection
selected_features = []
remaining_features = all_features.copy()
best_rmse = np.inf

# Forward feature selection loop
while remaining_features:
    best_feature = None
    for feature in remaining_features:
        # Try adding the current feature to the selected features
        current_features = selected_features + [feature]
        X_train_subset = X_train[current_features]
        X_val_subset = X_val[current_features]

        # Train the model on the current feature set
        reg_xgb = XGBRegressor(random_state=3)
        reg_xgb.fit(X_train_subset, y_train)

        # Evaluate on the validation set
        y_val_pred = reg_xgb.predict(X_val_subset)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))

        # Check if this feature improves the RMSE the most (i.e., decreases RMSE)
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_feature = feature

    # If we found a feature that improves the RMSE, add it to the selected features
    if best_feature is not None:
        selected_features.append(best_feature)
        remaining_features.remove(best_feature)
        print(f"Selected feature: {best_feature} with RMSE: {best_rmse:.4f}")
    else:
        # If no improvement, stop the process
        break

print(f"Final selected features: {selected_features}")
print(f"Best validation RMSE with selected features: {best_rmse:.4f}")

# Now you can use the selected features to train the final model or for further processing
