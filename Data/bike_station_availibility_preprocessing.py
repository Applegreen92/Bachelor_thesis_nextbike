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
    directory_path = 'nb23/nextbike/'

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
                            if station_name.startswith('BIKE') or station_name != 'Berliner Platz':
                                continue
                            available_bikes = place['bikes_available_to_rent']
                            lat = place['lat']
                            lng = place['lng']
                            data_records.append([timestamp, station_name, available_bikes, lat, lng])

    # Convert the list of records into a DataFrame
    df_bike_availability = pd.DataFrame(data_records, columns=['datetime', 'station_name', 'bikes_available', 'lon', 'lat'])

    # Save the DataFrame to a CSV filex
    df_bike_availability.to_csv('bike_availability_essen.csv', index=False)
