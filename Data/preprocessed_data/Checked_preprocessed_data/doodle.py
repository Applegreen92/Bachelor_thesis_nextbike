import pandas as pd

# Read the CSV file
csv_file_path = 'combined_citys/combined_city_data.csv'
df = pd.read_csv(csv_file_path)

# Round all numerical columns to 1 decimal places
df = df.round(1)

# Save the updated DataFrame back to a CSV file
df.to_csv(csv_file_path, index=False)