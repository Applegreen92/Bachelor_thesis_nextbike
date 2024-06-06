import pandas as pd
import numpy as np
import os
import glob
from collections import defaultdict



if __name__ == '__main__':
    # Directory containing the JSON files
    directory_path = 'nb23/test/'

    # Initialize a dictionary to store all the data
    all_stations_data = defaultdict(list)

    # Generate a date range with 30-minute intervals from 01-Jan-2023 00:00 to 31-Dec-2023 23:30
    date_range = pd.date_range(start='2023-01-01 00:00', end='2023-12-31 23:30', freq='30min')

    # Iterate through all JSON files in the directory
    for file_path in glob.glob(os.path.join(directory_path, '*.json')):
        # Extract data from each JSON file
        data = pd.read_json(file_path)
        stations_data = defaultdict(list)

        # Get me the names of the places and the number of available bikes at those places
        for country in data['countries']:
            for city in country['cities']:
                for place in city['places']:
                    station_name = place['name']
                    available_bikes = place['bikes_available_to_rent']
                    stations_data[station_name].append(available_bikes)
        # Append the data to the dictionary
        for station_name, bikes in stations_data.items():
            all_stations_data[station_name].extend(bikes)


    # Initialize the DataFrame with timestamps as index
    df_bike_availability = pd.DataFrame(index=date_range)

    # Populate the DataFrame with bike availability data
    for station_name, bike_counts in all_stations_data.items():
        
        # Ensure the bike_counts list is the same length as the date_range
        if len(bike_counts) != len(date_range):
            df_bike_availability[station_name] = None
        else:
            print(f"Data length mismatch for station {station_name}")

    # Display the DataFrame
    print(df_bike_availability.head())

    # Save the DataFrame to a CSV file
    df_bike_availability.to_csv('bike_availability.csv')
