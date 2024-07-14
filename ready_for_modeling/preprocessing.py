# This is a preliminary script for an ML algorithm, focusing on preprocessing the data.
import pandas as pd
import holidays

# Path to the CSV file
path_to_csv = 'precipitation_windSpeed_cloudCover_temp_bike_availability_essen.csv'

# Read the CSV file into a DataFrame
data = pd.read_csv(path_to_csv)

# Convert 'datetime' column to datetime object
data['datetime'] = pd.to_datetime(data['datetime'])

# Define holidays for North Rhine-Westphalia (NRW)
NRW_holidays = holidays.Germany(state='NW')

# Check if a date is a holiday and convert to integer (0 or 1)
data['is_holiday'] = data['datetime'].apply(lambda x: int(x.date() in NRW_holidays))

# Extract temporal features from 'datetime' column
data['minute'] = data['datetime'].dt.minute
data['hour'] = data['datetime'].dt.hour
data['day'] = data['datetime'].dt.day
data['month'] = data['datetime'].dt.month
data['year'] = data['datetime'].dt.year

# Drop the 'station_name' column since it does not seem relevant
data = data.drop(columns=['station_name', 'datetime'])

output_file_path = 'preprocessed_data/ready_to_go.csv'
data.to_csv(output_file_path, index=False)

