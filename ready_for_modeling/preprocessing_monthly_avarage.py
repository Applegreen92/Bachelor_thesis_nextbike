import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('data/combined_city_data.csv')

# Extract the relevant columns
features = ['bikes_available', 'bikes_booked', 'bikes_returned', 'temperature', 'cloud_cover', 'sfcWind',
            'precipitation']

# Define the output directory
output_dir = 'pictures/'

for feature in features:
    # Count the frequency of each value in the feature column
    value_counts = df[feature].value_counts().sort_index()

    # Plot the value counts as a bar plot
    plt.figure(figsize=(18, 6))
    value_counts.plot(kind='bar')
    plt.title(f'Frequency of Each Value for {feature}')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(f'{output_dir}{feature}_value_counts.png')
    plt.close()
