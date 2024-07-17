import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data_ready_updated = pd.read_csv('ready_to_go.csv')

# Define features and target
features_updated = ['precipitation', 'sfcWind', 'cloud_cover', 'temperature', 'is_holiday', 'minute', 'hour', 'day', 'month', 'weekday', 'year']
target = 'bikes_available'

# Split the data into training and testing sets
X_updated = data_ready_updated[features_updated]
y_updated = data_ready_updated[target]
X_train_updated, X_test_updated, y_train_updated, y_test_updated = train_test_split(X_updated, y_updated, test_size=0.2, random_state=42)

# Initialize models
models = {
    'RandomForest': RandomForestRegressor(random_state=42),
    'GradientBoosting': GradientBoostingRegressor(random_state=42),
    'Lasso': Lasso(random_state=42),
    'DecisionTree': DecisionTreeRegressor(random_state=42),
    'XGBoost': XGBRegressor(random_state=42)
}

# Train models and get feature importances
feature_importances = {}
for model_name, model in models.items():
    model.fit(X_train_updated, y_train_updated)
    if model_name == 'Lasso':
        # Lasso doesn't have feature_importances_ attribute, we use coefficients instead
        importance = np.abs(model.coef_)
    else:
        importance = model.feature_importances_
    feature_importances[model_name] = importance

# Create a DataFrame for feature importances
importance_df = pd.DataFrame(feature_importances, index=features_updated)

# Plot feature importances for each model
fig, axes = plt.subplots(2, 3, figsize=(18, 12), sharey=True)
fig.suptitle('Feature Importances Across Different Models')

for ax, model_name in zip(axes.flatten(), models.keys()):
    sns.barplot(y=importance_df.index, x=importance_df[model_name], ax=ax)
    ax.set_title(model_name)
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')

# Remove the last empty subplot if models are less than the grid (2x3 here)
if len(models) < 6:
    fig.delaxes(axes.flatten()[len(models)])
corr_matrix = data_ready_updated[features_updated].corr()

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')
plt.show()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
