import os
import threading
import time
import netCDF4 as nc
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load the CSV file containing bike availability and station information

csv_file_path = 'preprocessed_data/bike_station_data_Essen.csv'

csv_df = pd.read_csv(csv_file_path)
csv_df['datetime'] = pd.to_datetime(csv_df['datetime'])

# Define the path template and months for the NetCDF files
months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
temperature_path_template = 'weather/temperature/temperature_{}.nc'
cloudcover_path_template = 'weather/Cloudcover/cloudcover_{}.nc'
windspeed_path_template = 'weather/wind_speed/wind_speed_{}.nc'

# Pre-load NetCDF datasets into dictionaries
temperature_datasets = {month: nc.Dataset(temperature_path_template.format(month)) for month in months}
cloudcover_datasets = {month: nc.Dataset(cloudcover_path_template.format(month)) for month in months}
windspeed_datasets = {month: nc.Dataset(windspeed_path_template.format(month)) for month in months}

def find_nearest_2d_with_mask(lon_array, lat_array, lon, lat, mask):
    lon_flat = lon_array.flatten()
    lat_flat = lat_array.flatten()
    mask_flat = mask.flatten()

    valid_indices = np.where(mask_flat)[0]
    lon_flat = lon_flat[valid_indices]
    lat_flat = lat_flat[valid_indices]

    dist = np.sqrt((lon_flat - lon) ** 2 + (lat_flat - lat) ** 2)
    nearest_idx = valid_indices[dist.argmin()]
    lat_idx, lon_idx = np.unravel_index(nearest_idx, lon_array.shape)
    return lat_idx, lon_idx

def get_weather_for_month(dataset, lon_array, lat_array, data_array, valid_mask, lon, lat, datetime_obj):
    lat_idx, lon_idx = find_nearest_2d_with_mask(lon_array, lat_array, lon, lat, valid_mask)
    time_idx = nc.date2index(datetime_obj, dataset.variables['time'], select='nearest')
    return data_array[time_idx, lat_idx, lon_idx]

def process_month(month, df, datasets, variable_name):
    dataset = datasets[month]
    lon_array = dataset.variables['lon'][:]
    lat_array = dataset.variables['lat'][:]
    data_array = dataset.variables[variable_name][:]
    valid_mask = ~np.isnan(data_array).all(axis=0)

    weather_data = []
    for index, row in df.iterrows():
        try:
            weather_value = get_weather_for_month(dataset, lon_array, lat_array, data_array, valid_mask, row['lon'], row['lat'], row['datetime'])
            weather_data.append(weather_value)
        except IndexError as e:
            weather_data.append(np.nan)

    df[variable_name] = weather_data
    return df

def process_all_months(csv_df, datasets, variable_name):
    monthly_dfs = {month: csv_df[csv_df['datetime'].dt.strftime('%b').str.lower() == month] for month in months}

    print(f"Starting to process {variable_name} data...")
    with ThreadPoolExecutor(max_workers=12) as executor:
        future_to_month = {executor.submit(process_month, month, df, datasets, variable_name): month for month, df in monthly_dfs.items()}
        results = []
        for future in as_completed(future_to_month):
            month = future_to_month[future]
            try:
                result = future.result()
                print(f"Finished processing {variable_name} for month: {month}")
                results.append(result)
            except Exception as exc:
                print(f"Exception occurred while processing {variable_name} for month {month}: {exc}")

    return pd.concat(results).sort_values(by='datetime')

start_time = time.time()

# Process temperature
csv_df = process_all_months(csv_df, temperature_datasets, 'tas')

# Process cloud cover
csv_df = process_all_months(csv_df, cloudcover_datasets, 'clt')

# Process wind speed
csv_df = process_all_months(csv_df, windspeed_datasets, 'sfcWind')

# Close all opened NetCDF datasets
for datasets in [temperature_datasets, cloudcover_datasets, windspeed_datasets]:
    for dataset in datasets.values():
        dataset.close()

print("Finished processing weather data.")

# Load the precipitation data
precipitation_file_path = 'weather/precipitation/preprocessed_precipitation_essen.csv'
precipitation_df = pd.read_csv(precipitation_file_path)
precipitation_df['datetime'] = pd.to_datetime(precipitation_df['datetime'])

# Merge precipitation data
csv_df = pd.merge(csv_df, precipitation_df[['datetime', 'precipitation']], on='datetime', how='left')

print("Finished merging precipitation data.")

# Save the updated DataFrame to a new CSV file
output_file_path = 'preprocessed_data/final_bike_availability_essen.csv'
csv_df.to_csv(output_file_path, index=False)

end_time = time.time()
execution_time = end_time - start_time
print(f"Total execution time: {execution_time:.2f} seconds")
print("First few rows of the final dataframe:")
print(csv_df.head())
