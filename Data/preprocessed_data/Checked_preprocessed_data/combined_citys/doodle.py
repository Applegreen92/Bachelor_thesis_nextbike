import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('combined_city_data.csv')

# Convert the 'datetime' column to a datetime object
df['datetime'] = pd.to_datetime(df['datetime'])

# Extract the year, month, and day
df['year'] = df['datetime'].dt.year
df['month'] = df['datetime'].dt.month
df['day'] = df['datetime'].dt.day

# Group by year, month, and day, and calculate the mean for temperature and bikes_booked
daily_means = df.groupby(['year', 'month', 'day']).agg({'temperature': 'mean', 'bikes_available': 'mean'}).reset_index()

# Pivot the data to get monthly averages
monthly_means = daily_means.groupby(['year', 'month']).mean().reset_index()

# Plot the results
plt.figure(figsize=(14, 7))

# Temperature plot
plt.subplot(2, 1, 1)
plt.plot(monthly_means['month'], monthly_means['temperature'], marker='o', linestyle='-', color='b')
plt.title('Average Daily Temperature Over 12 Months')
plt.xlabel('Month')
plt.ylabel('Temperature (°C)')
plt.grid(True)

# Bikes booked plot
plt.subplot(2, 1, 2)
plt.plot(monthly_means['month'], monthly_means['bikes_available'], marker='o', linestyle='-', color='r')
plt.title('Average Bikes Available Over 12 Months')
plt.xlabel('Month')
plt.ylabel('Bikes available')
plt.grid(True)
plt.savefig('temp_abailable')
plt.tight_layout()
plt.show()
