import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

# Load the training data
train_df = pd.read_csv('combined_city_data.csv')
test_df = pd.read_csv('2022_complete_dresden.csv')

# Define feature columns and target column
selected_features = ['lon', 'lat','bikes_returned', 'bikes_difference', 'hour', 'month', 'weekday', 'is_weekend', 'is_holiday', 'temperature', 'sfcWind', 'precipitation']
target_col = 'bikes_available'

# Transform the target variable for binary classification
train_df['binary_bikes_available'] = (train_df[target_col] > 0).astype(int)
test_df['binary_bikes_available'] = (test_df[target_col] > 0).astype(int)

# Separate features and binary target variable from training and testing data
X_train = train_df[selected_features]
y_train_clf = train_df['binary_bikes_available']
y_train_reg = train_df[target_col]
X_test = test_df[selected_features]
y_test_clf = test_df['binary_bikes_available']
y_test_reg = test_df[target_col]

# Initialize the RandomForestClassifier and RandomForestRegressor
clf = RandomForestClassifier(n_estimators=100, random_state=42)
#reg = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the models on the training data
clf.fit(X_train, y_train_clf)
#reg.fit(X_train, y_train_reg)

# Make predictions on the testing data
clf_predictions = clf.predict(X_test)
#reg_predictions = reg.predict(X_test)

# Calculate the accuracy for the classifier
clf_accuracy = accuracy_score(y_test_clf, clf_predictions)

# Calculate the mean squared error for the regressor
#reg_mse = mean_squared_error(y_test_reg, reg_predictions)

print(f'Classifier Accuracy: {clf_accuracy:.4f}')
#print(f'Regressor Mean Squared Error: {reg_mse:.4f}')
