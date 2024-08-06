import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
import time

# Load the training (2023) and testing (2022) data
train_df = pd.read_csv('complete_dresden.csv')
test_df = pd.read_csv('2022_complete_dresden.csv')

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
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.1, 0.2, 0.3],
    'subsample': np.linspace(0.6, 1.0, 5)  # Subsampling ratios from 60% to 100%
}

# Initialize the Gradient Boosting Classifier
gb_clf = GradientBoostingClassifier(random_state=42)

# Initialize the RandomizedSearchCV with 5-fold cross-validation
random_search = RandomizedSearchCV(estimator=gb_clf, param_distributions=param_dist, n_iter=50, cv=5, scoring='accuracy', random_state=42, verbose=1, n_jobs=-1)

print("Starting hyperparameter tuning...")

start_time = time.time()

# Fit the random search to the training data
random_search.fit(X_train, y_train_clf)

end_time = time.time()
elapsed_time = end_time - start_time

# Get the best hyperparameters and best score from cross-validation
best_params = random_search.best_params_
best_score = random_search.best_score_

print(f'Best Hyperparameters: {best_params}')
print(f'Best Cross-Validation Accuracy: {best_score:.4f}')
print(f'Total tuning time: {elapsed_time:.2f} seconds')

# Display detailed results
cv_results = pd.DataFrame(random_search.cv_results_).sort_values(by='mean_test_score', ascending=False)
print("All Results:")
print(cv_results[['mean_test_score', 'std_test_score', 'params']])

# Train the final model with the best hyperparameters on the entire training set
best_gb_clf = GradientBoostingClassifier(**best_params, random_state=42)
best_gb_clf.fit(X_train, y_train_clf)

# Make predictions on the testing data
test_predictions = best_gb_clf.predict(X_test)

# Calculate the accuracy for the classifier on the test set
test_accuracy = accuracy_score(y_test_clf, test_predictions)

print(f'Test Accuracy: {test_accuracy:.4f}')
