import netCDF4 as nc
import numpy as np
import pandas as pd

# Open the NetCDF file
dataset = nc.Dataset('weather/tas_1hr_HOSTRADA-v1-0_BE_gn_2023010100-2023013123.nc')

# Print basic information about the dataset
print(dataset)

# Access the variables
lon = dataset.variables['lon'][:]
lat = dataset.variables['lat'][:]
time = dataset.variables['time'][:]
temperature = dataset.variables['tas'][:]

print(time)

# Convert time to a readable format if needed
units = dataset.variables['time'].units
time_converted = nc.num2date(time, units)

# Flatten the spatial coordinates and repeat them for each time step
lon_flat = lon.flatten()
lat_flat = lat.flatten()
num_time_steps = temperature.shape[0]

lon_repeated = np.tile(lon_flat, num_time_steps)
lat_repeated = np.tile(lat_flat, num_time_steps)
time_repeated = np.repeat(time_converted, len(lon_flat))

# Flatten the temperature data
temperature_flat = temperature.reshape(-1)

# Create a DataFrame for easy manipulation
data = {
    'time': time_repeated,
    'temperature': temperature_flat,
    'lon': lon_repeated,
    'lat': lat_repeated
}

df = pd.DataFrame(data)

# Display the DataFrame
print(df.head())
df.to_csv('temperature', index=False)
# Close the dataset
dataset.close()
