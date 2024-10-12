# 1. Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 2. Load the dataset (replace with your actual file path)
df = pd.read_csv('car data.csv', index_col=False)

df = pd.read_csv('car data.csv')
df.reset_index(drop=True, inplace=True)

# 3. Display the first few rows of the dataset
print(df.head())

# 4. Data Exploration: Check for missing values and data types
print(df.info())
print(df.describe())

print(df.columns)

# 5. Data Preprocessing
# Remove columns that may not be useful for price prediction (like 'car name')
df = df.drop(['Car_Name'], axis=1)

# Convert categorical variables into numerical format
df['Fuel_Type'] = df['Fuel_Type'].map({'Petrol': 0, 'Diesel': 1, 'CNG': 2})
df['Selling_type'] = df['Selling_type'].map({'Dealer': 0, 'Individual': 1})
df['Transmission'] = df['Transmission'].map({'Manual': 0, 'Automatic': 1})

# Feature Engineering: Create a new column for the age of the car
df['car_age'] = 2024 - df['Year']  # Assuming the current year is 2024
df = df.drop(['Year'], axis=1)  # Drop the 'year' column as we now have 'car_age'

# Check the processed data
print(df.head())

# 6. Data Splitting
# Define features (X) and target (y)
X = df.drop(['Selling_Price'], axis=1)  # Features (all except selling price)
y = df['Selling_Price']  # Target (selling price)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 7. Train the Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 8. Model Evaluation
# Predicting on the test set
y_pred = model.predict(X_test)

# Calculate the Mean Squared Error (MSE) and R-squared score (R2)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

# 9. Feature Importance (to see which features contribute the most)
feature_importance = pd.Series(model.feature_importances_, index=X.columns)
feature_importance.sort_values(ascending=False).plot(kind='bar', figsize=(10,6))
plt.title('Feature Importance in Car Price Prediction')
plt.show()

# 10. Predict the selling price of a new car
# Example: Predict the price for a car with given features
new_car = [[7.5, 50000, 0, 0, 1, 0, 1]]  # [present_price, driven_kms, fuel_type, selling_type, transmission, owner, car_age]
predicted_price = model.predict(new_car)
print(f"Predicted Selling Price for the new car: {predicted_price}")