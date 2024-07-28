# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset into a DataFrame
# Replace 'your_dataset.csv' with the path to your dataset file
df = pd.read_csv('combined_city_data.csv')
df = df.drop(columns='year')
# Calculate the correlation matrix
correlation_matrix = df.corr()

# Set up the matplotlib figure
plt.figure(figsize=(12, 10))

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

# Title for the heatmap
plt.title('Heatmap of Feature Correlations')
plt.savefig('heatmap all')
# Show the heatmap
plt.show()
