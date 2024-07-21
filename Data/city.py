import pandas as pd
import os
import glob
import gzip
from collections import defaultdict
from datetime import datetime
import holidays

def extract_timestamp_from_filename(filename):
    base_name = os.path.basename(filename)
    # Filename format: nextbike_YYYYMMDD_HHMM.json.gz
    date_str = base_name.split('_')[1] + base_name.split('_')[2].split('.')[0]
    timestamp = pd.to_datetime(date_str, format='%Y%m%d%H%M')
    return timestamp

def is_weekend(date):
    return date.weekday() >= 5

NRW_holidays = holidays.Germany(state='NW', years=2023)

def is_holiday(date):
    return date in NRW_holidays

# Directory containing the JSON.GZ files
directory_path = 'nb23/nextbike/'

# Dictionary to hold the bike IDs at each station over time
station_bike_ids = defaultdict(set)

# List to hold the data
data_records = []

# Iterate through all JSON.GZ files in the directory
for gz_file_path in sorted(glob.glob(os.path.join(directory_path, '*.json.gz'))):
    try:
        print(f"Processing file: {gz_file_path}")
        # Extract timestamp from filename
        timestamp = extract_timestamp_from_filename(gz_file_path)
        print(f"Extracted timestamp: {timestamp}")

        with gzip.open(gz_file_path, 'rt', encoding='utf-8') as json_file:
            # Read the JSON content
            data = pd.read_json(json_file)

            # Get the data for Essen
            for country in data['countries']:
                if country['country_name'] == 'Germany':
                    for city in country['cities']:
                        #print(city['name'])
                        #Plön, Nürnberg, Dresden, Erlangen
                        if city['name'] in ['Dresden']:
                            for place in city['places']:
                                station_name = place['name']
                                if station_name.startswith('BIKE'):
                                    continue
                                available_bikes = place['bikes_available_to_rent']
                                lat = place['lat']
                                lng = place['lng']
                                bike_numbers = set(place.get('bike_numbers', []))

                                # Calculate bikes booked and returned
                                if station_name in station_bike_ids:
                                    prev_bike_numbers = station_bike_ids[station_name]
                                    bikes_booked = len(prev_bike_numbers - bike_numbers)
                                    bikes_returned = len(bike_numbers - prev_bike_numbers)
                                else:
                                    bikes_booked = 0
                                    bikes_returned = 0

                                # Update the bike numbers for the station
                                station_bike_ids[station_name] = bike_numbers

                                data_records.append(
                                    [timestamp, station_name, available_bikes, bikes_booked, bikes_returned, lat, lng])
                                print(
                                    f"Added data: {[timestamp, station_name, available_bikes, bikes_booked, bikes_returned, lat, lng]}")

    except Exception as e:
        print(f"Error processing file {gz_file_path}: {e}")

# Convert the list of records into a DataFrame
df_bike_availability = pd.DataFrame(data_records,
                                    columns=['datetime', 'station_name', 'bikes_available', 'bikes_booked',
                                             'bikes_returned', 'lat', 'lon'])

# Add additional temporal features
df_bike_availability['minute'] = df_bike_availability['datetime'].dt.minute
df_bike_availability['hour'] = df_bike_availability['datetime'].dt.hour
df_bike_availability['day'] = df_bike_availability['datetime'].dt.day
df_bike_availability['month'] = df_bike_availability['datetime'].dt.month
df_bike_availability['year'] = df_bike_availability['datetime'].dt.year
df_bike_availability['weekday'] = df_bike_availability['datetime'].dt.weekday + 1
df_bike_availability['is_weekend'] = df_bike_availability['datetime'].apply(is_weekend)
df_bike_availability['is_holiday'] = df_bike_availability['datetime'].apply(is_holiday)

# Save the DataFrame to a CSV file
save_file_path = 'preprocessed_data/Checked_preprocessed_data/dresden/bike_station_data_dresden.csv'

df_bike_availability.to_csv(save_file_path, index=False)
print("Data saved to " + save_file_path)
