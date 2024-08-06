import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Load the training data
train_df = pd.read_csv('combined_city_data.csv')
test_df = pd.read_csv('2022_complete_dresden.csv')

# Define initial feature columns and target column
all_features = ['lon', 'lat','bikes_returned', 'bikes_difference', 'hour', 'month', 'weekday', 'is_weekend', 'is_holiday', 'temperature',
                'cloud_cover', 'sfcWind', 'precipitation']
target_col = 'bikes_available'

# Transform the target variable for binary classification
train_df['binary_bikes_available'] = (train_df[target_col] > 0).astype(int)
test_df['binary_bikes_available'] = (test_df[target_col] > 0).astype(int)

# Initialize the GradientBoostingClassifier
gbm = GradientBoostingClassifier(n_estimators=100, random_state=42)

# Recursive feature elimination
features = all_features.copy()
round_num = 1
while len(features) > 0:
    print(f"Round {round_num}:")
    print(f"Features: {features}")

    # Separate features and binary target variable from training and testing data
    X_train = train_df[features]
    y_train_clf = train_df['binary_bikes_available']
    X_test = test_df[features]
    y_test_clf = test_df['binary_bikes_available']

    # Fit the classifier on the training data
    gbm.fit(X_train, y_train_clf)

    # Make predictions on the testing data
    gbm_predictions = gbm.predict(X_test)

    # Calculate the accuracy for the classifier
    gbm_accuracy = accuracy_score(y_test_clf, gbm_predictions)
    print(f'Classifier Accuracy: {gbm_accuracy:.4f}')

    # Get feature importances
    importances = gbm.feature_importances_

    # Find the least important feature
    least_important_index = importances.argmin()
    least_important_feature = features[least_important_index]
    print(f'Removing least important feature: {least_important_feature}\n')

    # Remove the least important feature
    features.pop(least_important_index)

    round_num += 1

print("Feature elimination complete.")
