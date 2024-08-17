import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

# Load the training data
train_df = pd.read_csv('test_data/new_combined_city_data.csv')

# Define feature columns and target column
selected_features = ['city_lat','city_lng','bike_racks','lon', 'lat', 'hour', 'month', 'weekday', 'is_weekend', 'is_holiday', 'temperature', 'sfcWind', 'precipitation']
target_col = 'bikes_available'

# Transform the target variable for binary classification
train_df['binary_bikes_available'] = (train_df[target_col] > 0).astype(int)

# Separate features and binary target variable from training data
X_train = train_df[selected_features]
y_train_clf = train_df['binary_bikes_available']

# Initialize the XGBoost classifier with the specified hyperparameters
model = XGBClassifier(
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

# Initialize Sequential Feature Selector
sfs = SFS(model,
          k_features=5,  # Number of features to select
          forward=True,  # Forward selection
          floating=False,  # No floating steps
          scoring='accuracy',  # Using accuracy as the evaluation metric
          cv=5,  # 5-fold cross-validation
          n_jobs=-1)  # Use all available CPUs

# Fit SFS on the training data
sfs = sfs.fit(X_train, y_train_clf)

# Get the selected features
selected_features_sfs = list(sfs.k_feature_names_)
print(f'Selected Features by SFS: {selected_features_sfs}')

# Transform the training data to only include the selected features
X_train_sfs = sfs.transform(X_train)

# List of test CSV files
test_files = [
    'test_data/new_2022_combined_city_data.csv'
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

    # Transform the test data to include only the selected features
    X_test_sfs = sfs.transform(X_test)
    y_test_clf = test_df['binary_bikes_available']

    # Fit the model using the selected features
    model.fit(X_train_sfs, y_train_clf)

    # Make predictions on the testing data
    predictions = model.predict(X_test_sfs)

    # Calculate and print the accuracy
    accuracy = accuracy_score(y_test_clf, predictions)
    print(f'Accuracy on {test_file}: {accuracy:.4f}')
