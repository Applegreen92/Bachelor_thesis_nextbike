import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS

# Load the training data
train_df = pd.read_csv('test_data/new_combined_city_data.csv')

# Define feature columns and target column
selected_features = ['city_lat','city_lng','bike_racks','lon', 'lat', 'hour', 'month', 'weekday', 'is_weekend', 'is_holiday', 'temperature', 'sfcWind',
                     'precipitation']
target_col = 'bikes_available'

# Transform the target variable for binary classification
train_df['binary_bikes_available'] = (train_df[target_col] > 0).astype(int)

# Separate features and binary target variable from training data
X_train = train_df[selected_features]
y_train_clf = train_df['binary_bikes_available']

# Initialize XGBoost Classifier with given hyperparameters
clf_xgb = XGBClassifier(
    colsample_bytree=0.5765133157615585,
    gamma=2.8319073793101883,
    learning_rate=0.015406375121368878,
    max_depth=11,
    min_child_weight=1,
    n_estimators=70,
    subsample=0.9033844439161279,
    random_state=3,
    use_label_encoder=False,
    eval_metric='logloss'
)

# List of test CSV files
test_files = [
    'test_data/new_2022_combined_city_data.csv'
    # 'test_data/dresden.csv',
    # 'test_data/heidelberg.csv',
    # 'test_data/essen.csv',
    # 'test_data/nürnberg.csv',
]

# Initialize lists to store results for plotting
cities = []
clf_accuracies = []

# Function to perform Exhaustive Feature Selection
def perform_efs(X_train, y_train, X_test, y_test):
    efs = EFS(
        clf_xgb,
        min_features=1,
        max_features=len(selected_features),
        scoring='accuracy',
        print_progress=True,
        n_jobs=-1
    )
    efs = efs.fit(X_train, y_train)

    # Best feature subset
    best_features = list(efs.best_feature_names_)

    # Train and evaluate on the best feature subset
    clf_xgb.fit(X_train[best_features], y_train)
    predictions = clf_xgb.predict(X_test[best_features])
    accuracy = accuracy_score(y_test, predictions)

    return best_features, accuracy

# Loop through each test file, perform feature selection, and evaluate the classifier
for test_file in test_files:
    print(f"\nEvaluating on {test_file}...")

    # Load the testing data
    test_df = pd.read_csv(test_file)

    # Transform the target variable for binary classification
    test_df['binary_bikes_available'] = (test_df[target_col] > 0).astype(int)

    # Separate features and binary target variable from testing data
    X_test = test_df[selected_features]
    y_test_clf = test_df['binary_bikes_available']

    # Perform Exhaustive Feature Selection
    best_features, clf_xgb_accuracy = perform_efs(X_train, y_train_clf, X_test, y_test_clf)

    # Store the results
    city_name = test_file.split('/')[-1].split('.')[0]
    cities.append(city_name)
    clf_accuracies.append(clf_xgb_accuracy)

    # Print results
    print(f'Results for {test_file}:')
    print(f'Best Features: {best_features}')
    print(f'XGBoost Classifier Accuracy: {clf_xgb_accuracy:.4f}')

# Plot the classifier accuracy results
plt.figure(figsize=(12, 6))
plt.bar(cities, clf_accuracies, color='skyblue')
plt.title('XGBoost Classifier Accuracy with Best Feature Subset for Different Cities')
plt.xlabel('City')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.show()
