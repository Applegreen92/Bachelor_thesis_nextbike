import netCDF4 as nc
import numpy as np
import pandas as pd

# Load the CSV file containing bike availability and station information
csv_file_path = 'weather/bike_availability_essen_Berliner_Platz.csv'
csv_df = pd.read_csv(csv_file_path)

# Convert 'datetime' column to datetime objects
csv_df['datetime'] = pd.to_datetime(csv_df['datetime'])

def find_nearest_2d_with_mask(lon_array, lat_array, lon, lat, mask):
    """
    Find the indices of the nearest valid longitude and latitude values in 2D arrays.

    Parameters:
    lon_array (numpy array): 2D array of longitudes.
    lat_array (numpy array): 2D array of latitudes.
    lon (float): The longitude value to find the nearest to.
    lat (float): The latitude value to find the nearest to.
    mask (numpy array): 2D mask array indicating valid data points.

    Returns:
    tuple: Indices of the nearest valid longitude and latitude in the 2D arrays.
    """
    # Flatten the arrays and the mask to simplify the search
    lon_flat = lon_array.flatten()
    lat_flat = lat_array.flatten()
    mask_flat = mask.flatten()

    # Filter out invalid points
    valid_indices = np.where(mask_flat)[0]
    lon_flat = lon_flat[valid_indices]
    lat_flat = lat_flat[valid_indices]

    # Calculate the distance for each valid point
    dist = np.sqrt((lon_flat - lon)**2 + (lat_flat - lat)**2)

    # Find the index of the minimum distance
    nearest_idx = valid_indices[dist.argmin()]

    # Convert the flat index back to 2D indices
    lat_idx, lon_idx = np.unravel_index(nearest_idx, lon_array.shape)

    return lat_idx, lon_idx

def get_temperature_from_nc(file_path, lon, lat, datetime_obj):
    """
    Retrieve temperature from a NetCDF file for the nearest valid coordinates and time.

    Parameters:
    file_path (str): Path to the NetCDF file.
    lon (float): Longitude of the desired location.
    lat (float): Latitude of the desired location.
    datetime_obj (datetime): Datetime object representing the desired time.

    Returns:
    float: Temperature at the nearest valid coordinates and time.
    """
    # Open the NetCDF file
    dataset = nc.Dataset(file_path)

    # Extract longitude, latitude, and time arrays from the dataset
    lon_array = dataset.variables['lon'][:]
    lat_array = dataset.variables['lat'][:]

    # Extract temperature array and create a mask for valid data points
    tas_array = dataset.variables['tas'][:]
    valid_mask = ~np.isnan(tas_array).all(axis=0)

    # Debugging statements to check the shapes and contents
    print(f"Longitude array shape: {lon_array.shape}")
    print(f"Latitude array shape: {lat_array.shape}")
    print(f"Temperature array shape: {tas_array.shape}")
    print(f"Valid mask shape: {valid_mask.shape}")
    print(f"Valid mask sample: {valid_mask}")

    # Find the nearest valid indexes for the given longitude and latitude
    lat_idx, lon_idx = find_nearest_2d_with_mask(lon_array, lat_array, lon, lat, valid_mask)
    print(f"Nearest indices - lat_idx: {lat_idx}, lon_idx: {lon_idx}")

    # Handle time
    time_units = dataset.variables['time'].units
    time_calendar = dataset.variables['time'].calendar
    time_idx = nc.date2index(datetime_obj, dataset.variables['time'], select='nearest')
    print(f"Time index: {time_idx}")

    # Extract the temperature at the nearest coordinates and time index
    temperature = dataset.variables['tas'][time_idx, lat_idx, lon_idx]
    print(f"Extracted temperature: {temperature}")

    # Close the NetCDF dataset
    dataset.close()

    return temperature

# Define the path template and months for the NetCDF files
months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
file_path_template = 'temperature_{}.nc'

# Add temperature data to the CSV DataFrame
temperatures = []

for index, row in csv_df.iterrows():
    month = row['datetime'].strftime('%b').lower()  # Extract the month
    file_path = file_path_template.format(month)  # Determine the correct NetCDF file
    try:
        print(f"Processing row {index}: {row['datetime']}, {row['lon']}, {row['lat']}")
        temperature = get_temperature_from_nc(file_path, row['lon'], row['lat'], row['datetime'])  # Get the temperature
        temperatures.append(temperature)  # Store the temperature
    except IndexError as e:
        print(f"IndexError at row {index}: {e}")
        temperatures.append(np.nan)  # Append NaN if there is an error

# Add the temperature data as a new column to the CSV DataFrame
csv_df['temperature'] = temperatures

# Save the updated DataFrame with temperature data to a new CSV file
output_file_path = 'bike_availability_with_temperature.csv'
csv_df.to_csv(output_file_path, index=False)

# Display the first few rows of the updated DataFrame
print(csv_df.head())
