# train_model.py

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset
df = pd.read_csv('bengaluru_house_prices.csv')

# Extract BHK from 'size' column
df['BHK'] = df['size'].str.extract(r'(\d+)').astype(float)

# Keep useful columns
df = df[['location', 'total_sqft', 'bath', 'BHK', 'price']]

# Drop rows with missing values
df.dropna(inplace=True)

# Convert 'total_sqft' to numeric
df['total_sqft'] = pd.to_numeric(df['total_sqft'], errors='coerce')
df.dropna(inplace=True)

# One-hot encode location
df = pd.get_dummies(df, columns=['location'], drop_first=True)

# Split data into X and y
X = df.drop('price', axis=1)
y = df['price']

# Train model
model = LinearRegression()
model.fit(X, y)

# Save the trained model
pickle.dump(model, open('house_price_model.pkl', 'wb'))

# Save the column names (for use in Flask input)
pickle.dump(X.columns, open('columns.pkl', 'wb'))

print("✅ Model trained and saved as .pkl files.")
