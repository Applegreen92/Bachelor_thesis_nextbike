import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the training data
train_df = pd.read_csv('combined_city_data.csv')

# Define feature columns and target column
selected_features = ['lon', 'lat', 'hour', 'month', 'weekday', 'is_weekend',
                'is_holiday', 'temperature', 'sfcWind', 'precipitation']
target_col = 'bikes_available'

# Transform the target variable for binary classification
train_df['binary_bikes_available'] = (train_df[target_col] > 0).astype(int)

# Separate features and binary target variable from training data
X_train = train_df[selected_features]
y_train_clf = train_df['binary_bikes_available']

# Initialize XGBoost Classifier with given hyperparameters
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

# Fit the classifier on the training data
clf_xgb.fit(X_train, y_train_clf)

# List of test CSV files
test_files = [
    '2022_combined_city_data.csv',
    #'2022_complete_dresden.csv',
    #'2022_complete_heidelberg.csv',
    #'2022_complete_essen.csv',
    #'2022_complete_nürnberg.csv',
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
    clf_xgb_predictions = clf_xgb.predict(X_test)

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

# Feature Importance Plot
# Get feature importances from the trained model
feature_importances = clf_xgb.feature_importances_

# Create a DataFrame for better plotting
importance_df = pd.DataFrame({
    'Feature': selected_features,
    'Importance': feature_importances
})

# Sort the DataFrame by importance
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot the feature importance
plt.figure(figsize=(10, 8))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.title('Feature Importance - XGBoost')
plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature on top
plt.savefig('')
plt.show()
