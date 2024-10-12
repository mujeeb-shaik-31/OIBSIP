# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Load the datasets (replace with the correct file path if needed)
df1 = pd.read_csv('Unemployment in India.csv')
df2 = pd.read_csv('Unemployment_Rate_upto_11_2020.csv')

# Display the first 5 rows of both datasets to check the structure
print(df1.head())
print(df2.head())

# Check for missing values in both datasets
print(df1.isnull().sum())
print(df2.isnull().sum())

# Data cleaning (dropping any rows with missing values if present)
df1 = df1.dropna()
df2 = df2.dropna()

# Data overview: Summary statistics of both datasets
print(df1.describe())
print(df2.describe())

# Merge the two datasets based on the 'region' and 'date' columns
df1['Region'] = df1['Region'].str.lower()
df2['Region'] = df2['Region'].str.lower()
merged_df = pd.merge(df1, df2, on=['Region', ' Date'], suffixes=('_df1', '_df2'))

# Display the merged dataset
print(merged_df.head())

# Visualization: Unemployment rate over time (from the merged dataset)
plt.figure(figsize=(12, 6))
sns.lineplot(data=merged_df, x=' Date', y=' Estimated Unemployment Rate (%)_df2', hue='Region')
plt.title('Estimated Unemployment Rate Over Time by Region')
plt.xticks(rotation=90)
plt.show()

# Visualization: Heatmap of Unemployment by Region
plt.figure(figsize=(10, 6))
heatmap_data = merged_df.pivot_table(values=' Estimated Unemployment Rate (%)_df1', index='Region', columns=' Date')
sns.heatmap(heatmap_data, cmap='coolwarm', linewidths=0.5)
plt.title('Heatmap of Estimated Unemployment Rate by Region and Date')
plt.show()

# Visualization: Unemployment rates in different regions (Using the second dataset)
fig = px.scatter_geo(df2,
                     lat='latitude', lon='longitude',
                     hover_name='Region',
                     hover_data=[' Estimated Unemployment Rate (%)', ' Estimated Unemployment Rate (%)'],
                     color=' Estimated Unemployment Rate (%)',
                     title='Unemployment Rate in India by Region (2020)',
                     color_continuous_scale='Viridis')

fig.update_layout(geo=dict(scope='asia', projection_type='equirectangular'))
fig.show()

# Visualization: Unemployment vs Employment rate in both rural and urban areas
plt.figure(figsize=(12, 6))
sns.barplot(x='Area', y=' Estimated Unemployment Rate (%)', data=df1)
plt.title('Unemployment Rate in Urban vs Rural Areas')
plt.show()

# Summary: Correlation matrix for understanding relationships between features
correlation_matrix = merged_df[[' Estimated Unemployment Rate (%)_df1', ' Estimated Employed_df1', ' Estimated Labour Participation Rate (%)_df1']].corr()
plt.figure(figsize=(8, 5))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation between Unemployment, Employment and Labour Participation Rates')
plt.show()