import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('data/combined_city_data.csv')

# Extract the relevant columns
features = ['bikes_available', 'bikes_booked', 'bikes_returned', 'temperature', 'cloud_cover', 'sfcWind', 'precipitation']

# Group the data by 'hour' and calculate mean values for each group
hourly_data = df.groupby('hour')[features].mean()

# Plotting the frequency analysis
fig, axes = plt.subplots(3, 3, figsize=(18, 15))

# Plot bikes_available
axes[0, 0].plot(hourly_data.index, hourly_data['bikes_available'], marker='o')
axes[0, 0].set_title('Average Bikes Available per Hour')
axes[0, 0].set_xlabel('Hour of the Day')
axes[0, 0].set_ylabel('Bikes Available')

# Plot bikes_booked
axes[0, 1].plot(hourly_data.index, hourly_data['bikes_booked'], marker='o')
axes[0, 1].set_title('Average Bikes Booked per Hour')
axes[0, 1].set_xlabel('Hour of the Day')
axes[0, 1].set_ylabel('Bikes Booked')

# Plot bikes_returned
axes[0, 2].plot(hourly_data.index, hourly_data['bikes_returned'], marker='o')
axes[0, 2].set_title('Average Bikes Returned per Hour')
axes[0, 2].set_xlabel('Hour of the Day')
axes[0, 2].set_ylabel('Bikes Returned')

# Plot temperature
axes[1, 0].plot(hourly_data.index, hourly_data['temperature'], marker='o')
axes[1, 0].set_title('Average Temperature per Hour')
axes[1, 0].set_xlabel('Hour of the Day')
axes[1, 0].set_ylabel('Temperature (°C)')

# Plot cloud_cover
axes[1, 1].plot(hourly_data.index, hourly_data['cloud_cover'], marker='o')
axes[1, 1].set_title('Average Cloud Cover per Hour')
axes[1, 1].set_xlabel('Hour of the Day')
axes[1, 1].set_ylabel('Cloud Cover (%)')

# Plot sfcWind
axes[1, 2].plot(hourly_data.index, hourly_data['sfcWind'], marker='o')
axes[1, 2].set_title('Average Surface Wind Speed per Hour')
axes[1, 2].set_xlabel('Hour of the Day')
axes[1, 2].set_ylabel('Surface Wind Speed (m/s)')

# Plot precipitation
axes[2, 0].plot(hourly_data.index, hourly_data['precipitation'], marker='o')
axes[2, 0].set_title('Average Precipitation per Hour')
axes[2, 0].set_xlabel('Hour of the Day')
axes[2, 0].set_ylabel('Precipitation (mm)')

# Hide the empty subplot
axes[2, 1].axis('off')
axes[2, 2].axis('off')

output_path = 'pictures/avarage_daylie_frequency'
plt.savefig(output_path)

plt.tight_layout()
plt.show()
