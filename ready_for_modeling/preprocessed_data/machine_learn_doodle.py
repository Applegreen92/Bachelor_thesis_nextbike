import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Path to the preprocessed CSV file
path_to_csv = 'ready_to_go.csv'

# Read the CSV file into a DataFrame
data = pd.read_csv(path_to_csv)

# Define features and target
features = ['precipitation', 'sfcWind', 'cloud_cover', 'temperature','precipitation', 'is_holiday', 'minute', 'hour', 'day', 'month','weekday', 'year']
target = 'bikes_available'

# Split the data into training and testing sets
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mean_error = (y_test - y_pred).mean()
r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Error: {mean_error}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R^2 Score: {r2}")

# Residual Plot
residuals = y_test - y_pred

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Bike Availability')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(residuals, bins=30, kde=True)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Distribution of Residuals')
plt.show()

# Example prediction
example_data = pd.DataFrame({
    'precipitation': [0.1],
    'sfcWind': [3.5],
    'cloud_cover': [0.6],
    'temperature': [22],
    'is_holiday': [0],
    'minute': [30],
    'hour': [14],
    'day': [15],
    'month': [7],
    'weekday': [2],
    'year': [2023]
})

prediction = model.predict(example_data)
print(f"Predicted Bike Availability: {prediction[0]}")
