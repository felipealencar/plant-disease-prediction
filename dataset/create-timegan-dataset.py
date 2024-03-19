# Combine several .csv files into one .csv file
# Usage: python create-timegan-dataset.py

import os
import pandas as pd

# Set the paths to the directories containing the .csv files
paths = [
        r'C:\Users\flopes1\OneDrive - Saint Louis University\Desktop\Repos\plant-disease-prediction\dataset\LARGO1\08-30\VALUES\values.csv',
        r'C:\Users\flopes1\OneDrive - Saint Louis University\Desktop\Repos\plant-disease-prediction\dataset\LARGO1\09-24\VALUES\values.csv',
        r'C:\Users\flopes1\OneDrive - Saint Louis University\Desktop\Repos\plant-disease-prediction\dataset\LARGO1\10-05\VALUES\values.csv',
        ]

# Read the .csv files and store them in a list
dfs = [pd.read_csv(path) for path in paths]

# Normalize the column HEALTH_STA, it has HEALTHY, AVARAGE, and UNHEALTHY values


# Concatenate the .csv files into one .csv file
df = pd.concat(dfs, ignore_index=True)
df['HEALTH_STA'] = df['HEALTH_STA'].map({'HEALTHY': 0, 'AVERAGE': 1, 'UNHEALTHY': 2})

# Remove the 'Unnamed: 0' column
df = df.drop('Unnamed: 0', axis=1)

# Remove columns with '' or NaN values
df = df.replace('', pd.NA)
df = df.dropna()





# Save the concatenated .csv file
df.to_csv(r'C:\Users\flopes1\OneDrive - Saint Louis University\Desktop\Repos\plant-disease-prediction\dataset\timegan\plant_disease.csv', index=False)

# Print the shape of the concatenated .csv file
print(df.shape)

# Print the head of the concatenated .csv file
print(df.head())

