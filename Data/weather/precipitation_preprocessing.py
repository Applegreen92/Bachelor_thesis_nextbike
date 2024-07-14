import os
import tarfile
import pandas as pd
import numpy as np
from pyproj import Proj, transform
from datetime import datetime, timedelta


def parse_asc(file_path):
    """Parse the ASC file to extract header information and data grid."""
    header_info = {}
    data_start_line = 0

    with open(file_path, 'r') as file:
        lines = file.readlines()

        # Parse header
        for i, line in enumerate(lines):
            if line.startswith(('ncols', 'nrows', 'xllcorner', 'yllcorner', 'cellsize', 'NODATA_value')):
                key, value = line.strip().split()
                header_info[key] = float(value)
            else:
                data_start_line = i
                break

        # Parse data
        data = np.loadtxt(lines[data_start_line:], dtype=float)

    return header_info, data


def find_nearest_grid_cell(header_info, data, lat, lon, projection, nodata_value):
    """Find the nearest grid cell to a specific latitude and longitude and return its value."""
    x, y = projection(lon, lat)
    col = int(round((x - header_info['xllcorner']) / header_info['cellsize']))
    row = int(round((y - header_info['yllcorner']) / header_info['cellsize']))

    if 0 <= col < header_info['ncols'] and 0 <= row < header_info['nrows']:
        value = data[row, col]
        if value != nodata_value:
            return value
    return None


def process_tar_files(tar_files_directory, output_csv, lat, lon):
    """Process all tar files in the specified directory and extract precipitation data for a specific location."""
    # Define the projection
    p_stereo = Proj(proj='stere', lat_0=90, lon_0=10, lat_ts=60, x_0=0, y_0=0, datum='WGS84')

    all_data = []

    # Iterate over each tar file in the directory
    for tar_filename in os.listdir(tar_files_directory):
        if tar_filename.endswith(".tar") or tar_filename.endswith(".gz"):
            tar_path = os.path.join(tar_files_directory, tar_filename)
            print(f"Processing TAR file: {tar_filename}")
            with tarfile.open(tar_path, "r:gz") as tar:
                for member in tar.getmembers():
                    if member.isfile() and member.name.endswith(".asc"):
                        print(f"  Extracting and processing ASC file: {member.name}")
                        # Extract the file
                        extracted_file = tar.extractfile(member)
                        with open(member.name, "wb") as f:
                            f.write(extracted_file.read())

                        # Process the .asc file
                        datetime_str = member.name.split('_')[1].replace('-', '').replace('.asc',
                                                                                          '')  # Extract datetime part and remove .asc
                        dt = datetime.strptime(datetime_str, '%Y%m%d%H%M')
                        dt = dt.replace(minute=0)  #+ timedelta(hours=dt.minute // 30)

                        header_info, data = parse_asc(member.name)
                        value = find_nearest_grid_cell(header_info, data, lat, lon, p_stereo, nodata_value=-1)
                        if value is not None:
                            all_data.append((dt, value))
                            all_data.append((dt.replace(minute=30), value))

                        # Clean up the extracted file
                        os.remove(member.name)

    # Convert to DataFrame
    df = pd.DataFrame(all_data, columns=['datetime', 'precipitation'])

    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"Data extracted and saved to {output_csv}")


# Example usage
tar_files_directory = 'precipitation/raw_data/'  # Replace with your directory
output_csv = 'preprocessed_precipitation_essen.csv'  # Replace with your output path
lat_essen = 51.458744
lon_essen = 7.004194
process_tar_files(tar_files_directory, output_csv, lat_essen, lon_essen)
