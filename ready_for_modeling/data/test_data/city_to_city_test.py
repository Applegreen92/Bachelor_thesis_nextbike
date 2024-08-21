import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# File names
city_files_2023 = {
    'dresden': '2023_dresden.csv',
    'essen': '2023_essen.csv',
    'heidelberg': '2023_heidelberg.csv',
    'nürnberg': '2023_nürnberg.csv'
}

city_files_2022 = {
    'dresden': '2022_complete_dresden.csv',
    'essen': '2022_complete_essen.csv',
    'heidelberg': '2022_complete_heidelberg.csv',
    'nürnberg': '2022_complete_nürnberg.csv'
}

# Define the target column
target_col = 'bikes_available'

# Initialize a dictionary to store the results
results = {}

# Forward feature selection for each city
for city, train_file in city_files_2023.items():
    print(f"\nPerforming forward feature selection for {city}...")

    # Load the training data
    train_df = pd.read_csv(train_file)

    # Transform the target variable for binary classification
    train_df['binary_bikes_available'] = (train_df[target_col] > 0).astype(int)
    y_train_clf = train_df['binary_bikes_available']

    # List of all possible features
    all_features = ['lon', 'lat', 'hour', 'month', 'weekday', 'is_weekend',
                    'is_holiday', 'temperature', 'sfcWind']

    # Initialize the best feature set and accuracy
    best_feature_set = []
    best_accuracy = 0

    # Forward feature selection loop
    while True:
        best_local_accuracy = 0
        best_feature = None

        # Test each feature not in the current best set
        for feature in all_features:
            if feature not in best_feature_set:
                current_features = best_feature_set + [feature]
                X_train = train_df[current_features]

                # Initialize and train the XGBoost Classifier
                clf_xgb = XGBClassifier(
                    colsample_bytree=1,
                    gamma=0,
                    learning_rate=0.3,
                    max_depth=6,
                    min_child_weight=1,
                    n_estimators=100,
                    subsample=0.9033844439161279,
                    random_state=42,
                    use_label_encoder=False,
                    eval_metric='logloss'
                )
                clf_xgb.fit(X_train, y_train_clf)

                # Evaluate on training data
                train_predictions = clf_xgb.predict(X_train)
                accuracy = accuracy_score(y_train_clf, train_predictions)

                # Select the best feature that improves accuracy
                if accuracy > best_local_accuracy:
                    best_local_accuracy = accuracy
                    best_feature = feature

        # If adding the best feature does not improve accuracy, break the loop
        if best_local_accuracy > best_accuracy:
            best_accuracy = best_local_accuracy
            best_feature_set.append(best_feature)
            print(f"Added feature {best_feature}, new accuracy: {best_accuracy:.4f}")
        else:
            break

    # Save the best feature set for this city
    results[city] = {
        'best_features': best_feature_set,
        'train_accuracy': best_accuracy
    }
    print(f"Best feature set for {city}: {best_feature_set}, with Accuracy: {best_accuracy:.4f}")

    # Load the corresponding 2022 dataset for evaluation
    test_df = pd.read_csv(city_files_2022[city])

    # Ensure test dataset has all necessary features
    missing_features = set(best_feature_set) - set(test_df.columns)
    for feature in missing_features:
        test_df[feature] = 0  # Fill missing features with a default value (e.g., 0)

    # Transform the target variable for binary classification
    test_df['binary_bikes_available'] = (test_df[target_col] > 0).astype(int)
    y_test_clf = test_df['binary_bikes_available']

    # Use the best feature set found for evaluation
    X_test = test_df[list(best_feature_set)]

    # Make predictions on the testing data
    clf_xgb_predictions = clf_xgb.predict(X_test)

    # Calculate the accuracy for the classifier
    clf_xgb_accuracy = accuracy_score(y_test_clf, clf_xgb_predictions)

    # Store the test results
    results[city]['test_accuracy'] = clf_xgb_accuracy
    print(f"Test accuracy for {city} on 2022 data: {clf_xgb_accuracy:.4f}")

# Display final results
print("\nSummary of results:")
for city, result in results.items():
    print(
        f"{city.capitalize()}: Best Features: {result['best_features']}, Train Accuracy: {result['train_accuracy']:.4f}, Test Accuracy: {result['test_accuracy']:.4f}")

# Optionally, save the results to a CSV file
results_df = pd.DataFrame.from_dict(results, orient='index')
results_df.to_csv('city_feature_optimization_results.csv')

# Plot the classifier accuracy results for the test set
plt.figure(figsize=(12, 6))
plt.bar(results_df.index, results_df['test_accuracy'], color='skyblue')
plt.title('XGBoost Classifier Test Accuracy for Different Cities Using Best Feature Set')
plt.xlabel('City')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.show()
