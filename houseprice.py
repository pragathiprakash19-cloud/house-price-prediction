# House Price Prediction Project

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Load dataset
df = pd.read_csv("house_prices.csv")

print("First rows")
print(df.head())

print("\nDataset Shape")
print(df.shape)

print("\nDataset Info")
print(df.info())

print("\nMissing Values")
print(df.isnull().sum())

print("\nStatistical Summary")
print(df.describe())

# Features and target
X = df[['Area','Bedrooms','Age']]
y = df['Price']

print("\nFeature Sample")
print(X.head())

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

print("\nTraining Data:",X_train.shape)
print("Testing Data:",X_test.shape)

# Train model
model = LinearRegression()
model.fit(X_train,y_train)

print("\nModel Coeffients:",model.coef_)
print("Intercept:",model.intercept_)

# Prediction
y_pred = model.predict(X_test)

print("\nPredicted Prices")
print(y_pred[:5])

# Model evaluation
r2 = r2_score(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)

print("\nR2 Score:,{r2:rf}")
print("Mean Squared Error:{mse:2f}")

new_house=[[1400,3,5]]
predicted_price=model.predict(new_house)
print("Predicted price for new house:",predicted_price)

# Plot actual vs predicted
plt.scatter(y_test,y_pred)
plt.plot(y_test,y_test)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices")
plt.show()

print("\nHouse Price Prediction Completed")