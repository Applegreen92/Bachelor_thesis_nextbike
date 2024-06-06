import json
import pandas as pd
import matplotlib as plt




if __name__ == '__main__':

    # Load JSON data from file
    file_path = 'Data/nb22/nextbike_20220101_0000.json'

    data = pd.read_json(file_path)

    # Initialize a list to store the data
    stations_data = []

    # Extract relevant information
    for country in data['countries']:
        for city in country['cities']:
            for place in city['places']:
                station_name = place['name']
                available_bikes = place['bikes_available_to_rent']
                stations_data.append({'station_name': station_name, 'available_bikes': available_bikes})

    # Create a DataFrame
    df_stations = pd.DataFrame(stations_data)

    # Display the DataFrame
    print(df_stations)

    # Save DataFrame to CSV if needed
    df_stations.to_csv('stations_data.csv', index=False)

