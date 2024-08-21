import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import matplotlib.pyplot as plt

# Load the training data
train_df = pd.read_csv('test_data/new_combined_city_data.csv')

# Define feature columns and target column
selected_features = ['lat', 'lon', 'hour', 'temperature']
target_col = 'bikes_available'

# Transform the target variable for binary classification
train_df['binary_bikes_available'] = (train_df[target_col] > 0).astype(int)

# Separate features and binary target variable from training data
X_train = train_df[selected_features]
y_train_clf = train_df['binary_bikes_available']

# Define hyperparameter grid for the classifier
clf_param_grid = {
    'n_estimators': [100, 150, 200],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.1, 0.2, 0.3],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2],
}

# Initialize the RandomizedSearchCV for classifier
clf_xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
clf_random_search = RandomizedSearchCV(
    clf_xgb,
    param_distributions=clf_param_grid,
    n_iter=100,  # Number of parameter settings that are sampled
    scoring='accuracy',
    n_jobs=-1,
    cv=5,  # 5-fold cross-validation
    verbose=1,
    random_state=42
)

# Perform hyperparameter tuning for the classifier
print("Tuning XGBoost Classifier...")
clf_random_search.fit(X_train, y_train_clf)
best_clf_model = clf_random_search.best_estimator_
print(f"Best Classifier Parameters: {clf_random_search.best_params_}")

# List of test CSV files
test_files = [
    'test_data/new_2022_combined_city_data.csv',
    'test_data/2023_dresden.csv',
    'test_data/2023_heidelberg.csv',
    'test_data/2023_essen.csv',
    'test_data/2023_nürnberg.csv',
]

# Initialize lists to store results for plotting
cities = []
clf_accuracies = []

# Loop through each test file, evaluate the model, and store results
for test_file in test_files:
    print(f"\nEvaluating on {test_file}...")

    # Load the testing data
    test_df = pd.read_csv(test_file)

    # Transform the target variable for binary classification
    test_df['binary_bikes_available'] = (test_df[target_col] > 0).astype(int)

    # Separate features and binary target variable from testing data
    X_test = test_df[selected_features]
    y_test_clf = test_df['binary_bikes_available']

    # Make predictions on the testing data
    clf_xgb_predictions = best_clf_model.predict(X_test)

    # Calculate the accuracy for the classifier
    clf_xgb_accuracy = accuracy_score(y_test_clf, clf_xgb_predictions)

    # Store the results
    city_name = test_file.split('/')[-1].split('.')[0]
    cities.append(city_name)
    clf_accuracies.append(clf_xgb_accuracy)

    # Print results
    print(f'Results for {test_file}:')
    print(f'XGBoost Classifier Accuracy: {clf_xgb_accuracy:.4f}')

# Plot the classifier accuracy results
plt.figure(figsize=(12, 6))
plt.bar(cities, clf_accuracies, color='skyblue')
plt.title('XGBoost Classifier Accuracy for Different Cities')
plt.xlabel('City')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.show()
