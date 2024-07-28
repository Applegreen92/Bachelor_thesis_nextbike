import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

# Load the data from the CSV file
csv_file = 'combined_city_data.csv'  # Replace with the path to your CSV file
original_df = pd.read_csv(csv_file)

# Function to convert 'bikes_available' to a binary target variable based on a threshold
def convert_target(data, threshold):
    return data['bikes_available'].apply(lambda x: 0 if x <= threshold else 1)

# List of thresholds to test
thresholds = [1, 2, 3, 4, 5]

# Directory to save the plots
output_dir = 'feature_importance_plots'
os.makedirs(output_dir, exist_ok=True)

# List of selected features (adjust this list to include/exclude features)
selected_features = ['feature1', 'feature2', 'feature3']  # Replace with the names of the features you want to include

# Initialize a dictionary to store accuracies and feature importances
results = {'Threshold': [], 'Accuracy': [], 'Feature Importances': []}

for threshold in thresholds:
    # Create a fresh copy of the original DataFrame for each threshold
    df = original_df.copy()

    # Convert 'bikes_available' to a binary target variable
    df['target'] = convert_target(df, threshold)

    # Drop the original 'bikes_available' column as it is now the target
    X = df[selected_features]
    y = df['target']

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the RandomForestClassifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)

    # Predict the target values for the test set
    y_pred = rf_classifier.predict(X_test)

    # Calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Extract feature importances
    feature_importances = rf_classifier.feature_importances_

    # Store the results
    results['Threshold'].append(threshold)
    results['Accuracy'].append(accuracy)
    results['Feature Importances'].append(feature_importances)

    # Print the accuracy
    print(f'Threshold: {threshold}, Accuracy: {accuracy:.2f}')

    # Create a DataFrame for feature importances
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': feature_importances
    })

    # Sort the DataFrame by importance
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # Plot the feature importance barplot
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title(f'Feature Importances for Threshold {threshold}')
    plt.savefig(os.path.join(output_dir, f'feature_importance_threshold_{threshold}.png'))
    plt.close()

# Convert results to DataFrame for easier plotting
results_df = pd.DataFrame(results)

# Plot the accuracy for different thresholds
plt.figure(figsize=(10, 6))
sns.barplot(x='Threshold', y='Accuracy', data=results_df)
plt.title('Accuracy for Different Thresholds')
plt.xlabel('Threshold')
plt.ylabel('Accuracy')
plt.savefig(os.path.join(output_dir, 'accuracy_for_different_thresholds.png'))
plt.show()
