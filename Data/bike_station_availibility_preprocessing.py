import pandas as pd
import os
import glob
import gzip
from collections import defaultdict



def extract_timestamp_from_filename(filename):
    base_name = os.path.basename(filename)
    # Filename format: nextbike_YYYYMMDD_HHMM.json.gz
    date_str = base_name.split('_')[1] + base_name.split('_')[2].split('.')[0]
    timestamp = pd.to_datetime(date_str, format='%Y%m%d%H%M')
    return timestamp


if __name__ == '__main__':
    # Directory containing the JSON.GZ files
    directory_path = 'nb23/test/'

    # List to hold the data
    data_records = []

    # Iterate through all JSON.GZ files in the directory
    for gz_file_path in glob.glob(os.path.join(directory_path, '*.json.gz')):
        # Extract timestamp from filename
        timestamp = extract_timestamp_from_filename(gz_file_path)

        with gzip.open(gz_file_path, 'rt', encoding='utf-8') as json_file:
            # Read the JSON content

            data = pd.read_json(json_file)

            # Get the data for Essen
            for country in data['countries']:
                for city in country['cities']:
                    if city['name'] == 'Essen':
                        for place in city['places']:
                            station_name = place['name']
                            if station_name.startswith('BIKE'):
                                continue
                            available_bikes = place['bikes_available_to_rent']
                            data_records.append([timestamp, station_name, available_bikes])

    # Convert the list of records into a DataFrame
    df_bike_availability = pd.DataFrame(data_records, columns=['time', 'station_name', 'bikes_available'])

    # Save the DataFrame to a CSV file
    df_bike_availability.to_csv('bike_availability_essen.csv', index=False)
