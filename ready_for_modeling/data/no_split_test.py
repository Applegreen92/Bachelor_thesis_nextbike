import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

# Load the training data
train_df = pd.read_csv('combined_city_data.csv')

# Define feature columns and target column
selected_features = ['lon', 'lat', 'hour', 'month', 'weekday', 'is_weekend', 'is_holiday', 'temperature', 'sfcWind',
                     'precipitation']
target_col = 'bikes_available'

# Transform the target variable for binary classification
train_df['binary_bikes_available'] = (train_df[target_col] > 0).astype(int)

# Separate features and binary target variable from training data
X_train = train_df[selected_features]
y_train_clf = train_df['binary_bikes_available']
y_train_reg = train_df[target_col].clip(upper=40)  # Cap the target values at 40

# Initialize the RandomForestClassifier and RandomForestRegressor
clf = RandomForestClassifier(
    n_estimators=50,
    min_samples_split=5,
    min_samples_leaf=4,
    max_features=None,
    max_depth=10,
    bootstrap=False,
    random_state=3
)

reg = RandomForestRegressor(
    n_estimators=50,
    min_samples_split=5,
    min_samples_leaf=4,
    max_features=None,
    max_depth=10,
    bootstrap=False,
    random_state=3
)

# Fit the models on the training data
clf.fit(X_train, y_train_clf)
reg.fit(X_train, y_train_reg)

# List of test CSV files
test_files = [
    '2022_combined_city_data.csv'
]

# Loop through each test file, evaluate the model, and print results
for test_file in test_files:
    print(f"\nEvaluating on {test_file}...")

    # Load the testing data
    test_df = pd.read_csv(test_file)

    # Transform the target variable for binary classification
    test_df['binary_bikes_available'] = (test_df[target_col] > 0).astype(int)

    # Separate features and binary target variable from testing data
    X_test = test_df[selected_features]
    y_test_clf = test_df['binary_bikes_available']
    y_test_reg = test_df[target_col].clip(upper=40)  # Cap the target values at 40

    # Make predictions on the testing data
    clf_predictions = clf.predict(X_test)
    reg_predictions = reg.predict(X_test)

    # Calculate the accuracy for the classifier
    clf_accuracy = accuracy_score(y_test_clf, clf_predictions)

    # Calculate the mean squared error for the regressor
    reg_mse = mean_squared_error(y_test_reg, reg_predictions)

    # Print results
    print(f'Results for {test_file}:')
    print(f'Classifier Accuracy: {clf_accuracy:.4f}')
    print(f'Regressor Mean Squared Error: {reg_mse:.4f}')
