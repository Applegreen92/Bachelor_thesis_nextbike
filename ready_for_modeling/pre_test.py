import holidays
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file into a DataFrame
data = pd.read_csv('precipitation_windSpeed_cloudCover_temp_bike_availability_essen.csv')

# Extract datetime features
data['datetime'] = pd.to_datetime(data['datetime'])
data['minute'] = data['datetime'].dt.minute
data['hour'] = data['datetime'].dt.hour
data['day'] = data['datetime'].dt.day
data['month'] = data['datetime'].dt.month
data['year'] = data['datetime'].dt.year

# Define holidays for North Rhine-Westphalia (NRW)
NRW_holidays = holidays.Germany(state='NW')

# Check if a date is a holiday and convert to integer (0 or 1)
data['is_holiday'] = data['datetime'].apply(lambda x: int(x.date() in NRW_holidays))

# Define features and target
features = ['precipitation', 'sfcWind', 'cloud_cover', 'temperature', 'is_holiday', 'minute', 'hour', 'day', 'month', 'weekday', 'year']
target = 'bikes_available'

# Split the data into training and testing sets
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Get feature importances
feature_importances = model.feature_importances_

# Create a DataFrame for feature importances
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})

# Sort the DataFrame by importance
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df, x='Importance', y='Feature')
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()
