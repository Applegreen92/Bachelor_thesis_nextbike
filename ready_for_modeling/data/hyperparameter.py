import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
import time

# Load the training (2023) and testing (2022) data
train_df = pd.read_csv('combined_city_data.csv')
test_df = pd.read_csv('2022_combined_city_data.csv')

# Define feature columns and target column
selected_features = ['lon', 'lat', 'bikes_difference', 'hour', 'month', 'weekday', 'is_weekend', 'is_holiday', 'temperature', 'cloud_cover', 'sfcWind', 'precipitation']
target_col = 'bikes_available'

# Transform the target variable for binary classification
train_df['binary_bikes_available'] = (train_df[target_col] > 0).astype(int)
test_df['binary_bikes_available'] = (test_df[target_col] > 0).astype(int)

# Separate features and binary target variable from training and testing data
X_train = train_df[selected_features]
y_train_clf = train_df['binary_bikes_available']
X_test = test_df[selected_features]
y_test_clf = test_df['binary_bikes_available']

# Define the parameter distribution for RandomizedSearchCV
param_dist = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2', None],
    'bootstrap': [True, False]
}

# Initialize the Random Forest Classifier
rf_clf = RandomForestClassifier(random_state=42)

# Initialize the RandomizedSearchCV with 5-fold cross-validation
random_search_rf = RandomizedSearchCV(estimator=rf_clf, param_distributions=param_dist, n_iter=50, cv=5, scoring='accuracy', random_state=42, verbose=1, n_jobs=-1)

print("Starting hyperparameter tuning for RandomForest...")

start_time = time.time()

# Fit the random search to the training data
random_search_rf.fit(X_train, y_train_clf)

end_time = time.time()
elapsed_time_rf = end_time - start_time

# Get the best hyperparameters and best score from cross-validation
best_params_rf = random_search_rf.best_params_
best_score_rf = random_search_rf.best_score_

print(f'Best Hyperparameters for RandomForest: {best_params_rf}')
print(f'Best Cross-Validation Accuracy for RandomForest: {best_score_rf:.4f}')
print(f'Total tuning time for RandomForest: {elapsed_time_rf:.2f} seconds')

# Save the best parameters and best score to a file
with open('random_forest_best_params.txt', 'w') as f:
    f.write(f'Best Hyperparameters for RandomForest: {best_params_rf}\n')
    f.write(f'Best Cross-Validation Accuracy for RandomForest: {best_score_rf:.4f}\n')
    f.write(f'Total tuning time for RandomForest: {elapsed_time_rf:.2f} seconds\n')

# Display detailed results
cv_results_rf = pd.DataFrame(random_search_rf.cv_results_).sort_values(by='mean_test_score', ascending=False)

# Save all cross-validation results to a CSV file
cv_results_rf.to_csv('random_forest_cv_results.csv', index=False)

print("All Results for RandomForest:")
print(cv_results_rf[['mean_test_score', 'std_test_score', 'params']])

# Train the final model with the best hyperparameters on the entire training set
best_rf_clf = RandomForestClassifier(**best_params_rf, random_state=42)
best_rf_clf.fit(X_train, y_train_clf)

# Make predictions on the testing data
test_predictions_rf = best_rf_clf.predict(X_test)

# Calculate the accuracy for the classifier on the test set
test_accuracy_rf = accuracy_score(y_test_clf, test_predictions_rf)

print(f'Test Accuracy for RandomForest: {test_accuracy_rf:.4f}')

# Save the test accuracy to a file
with open('random_forest_test_accuracy.txt', 'w') as f:
    f.write(f'Test Accuracy for RandomForest: {test_accuracy_rf:.4f}\n')
