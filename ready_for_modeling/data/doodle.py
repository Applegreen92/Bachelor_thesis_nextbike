import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load the data from the CSV file
csv_file = 'combined_city_data.csv'  # Replace with the path to your CSV file
df = pd.read_csv(csv_file)

# Assuming the target variable is named 'target', and is present in the dataframe
# If your target column has a different name, replace 'target' with the actual name
target_column = 'bikes_available'  # Replace with the actual target column name

# Separate features and target
X = df.drop(columns=[target_column])
y = df[target_column]

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the RandomForestRegressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)

# Extract feature importances
feature_importances = rf_regressor.feature_importances_

# Create a DataFrame for feature importances
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
})

# Sort the DataFrame by importance
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(importance_df.set_index('Feature').T, annot=True, cmap='YlGnBu', cbar=True)
plt.title('Feature Importances for RandomForestRegressor')
plt.show()
