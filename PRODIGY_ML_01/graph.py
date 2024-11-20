# Step 1: Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Step 2: Load the Dataset
# Replace with the correct path to your downloaded dataset
data = pd.read_csv(r"C:\Users\SIDDHI\Documents\train.csv")

# Step 3: Data Exploration
# Display first few rows
print("Dataset Head:")
print(data.head())

# Focus on relevant columns for the task
columns = ['GrLivArea', 'BedroomAbvGr', 'FullBath', 'SalePrice']
data = data[columns]

# Step 4: Handle Missing Values (if any)
data = data.dropna()  # Drop rows with missing values

# Step 5: Feature and Target Separation
X = data[['GrLivArea', 'BedroomAbvGr', 'FullBath']]  # Features
y = data['SalePrice']  # Target variable

# Step 6: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Model Training
model = LinearRegression()
model.fit(X_train, y_train)

# Step 8: Model Evaluation
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("Mean Squared Error:", mse)
print("R-squared Value:", r2)

# Step 9: Display Coefficients
print("\nModel Coefficients:")
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Step 10: Prediction Example
sample_data = np.array([[2000, 3, 2]])  # Example input: 2000 sq ft, 3 bedrooms, 2 bathrooms
predicted_price = model.predict(sample_data)
print("\nPredicted House Price for the sample input:", predicted_price[0])

# Step 11: Visualization
plt.figure(figsize=(10,6))

# Plot actual vs predicted prices
plt.scatter(y_test, y_pred, color='blue', label='Predicted', alpha=0.6)
plt.scatter(y_test, y_test, color='yellow', label='Actual', alpha=0.4)

# Plot the ideal prediction line (where predicted = actual)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2, label='Ideal Prediction')

plt.xlabel('Actual Sale Price')
plt.ylabel('Predicted Sale Price')
plt.title('Actual vs Predicted House Prices')
plt.legend()

plt.show()

