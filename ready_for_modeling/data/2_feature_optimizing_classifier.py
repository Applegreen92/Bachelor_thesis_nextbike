import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Load the training data
train_df = pd.read_csv('test_data/2023_dresden.csv')

# Define all potential feature columns and the target column
all_features = ['lon', 'lat', 'hour', 'month', 'weekday', 'is_weekend',
                'is_holiday', 'temperature', 'sfcWind', 'precipitation','precipitation','bikes_bookes','bikes_difference']
target_col = 'bikes_available'

# Transform the target variable for binary classification
train_df['binary_bikes_available'] = (train_df[target_col] > 0).astype(int)

# Separate features and binary target variable from training data
X_train = train_df[all_features]
y_train = train_df['binary_bikes_available']

# Load the validation dataset (second CSV file)
test_df = pd.read_csv('test_data/2022_complete_dresden.csv')
test_df['binary_bikes_available'] = (test_df[target_col] > 0).astype(int)
X_val = test_df[all_features]
y_val = test_df['binary_bikes_available']

# Initialize variables for forward feature selection
selected_features = []
remaining_features = all_features.copy()
best_accuracy = 0

# Forward feature selection loop
while remaining_features:
    best_feature = None
    for feature in remaining_features:
        # Try adding the current feature to the selected features
        current_features = selected_features + [feature]
        X_train_subset = X_train[current_features]
        X_val_subset = X_val[current_features]

        # Train the model on the current feature set
        clf_xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
        clf_xgb.fit(X_train_subset, y_train)

        # Evaluate on the validation set
        y_val_pred = clf_xgb.predict(X_val_subset)
        val_accuracy = accuracy_score(y_val, y_val_pred)

        # Check if this feature improves the accuracy the most
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_feature = feature

    # If we found a feature that improves the accuracy, add it to the selected features
    if best_feature is not None:
        selected_features.append(best_feature)
        remaining_features.remove(best_feature)
        print(f"Selected feature: {best_feature} with accuracy: {best_accuracy:.4f}")
    else:
        # If no improvement, stop the process
        break

print(f"Final selected features: {selected_features}")
print(f"Best validation accuracy with selected features: {best_accuracy:.4f}")

# Now you can use the selected features to train the final model or for further processing
