import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("house_prices.csv")

# Features and target
X = df[['Area', 'Bedrooms', 'Age']]
y = df['Price']

# Train model
model = LinearRegression()
model.fit(X, y)

# App title
st.title("🏠 House Price Prediction App")

st.write("Enter details to predict house price")

# User inputs
area = st.number_input("Enter Area (sq ft)", min_value=500, max_value=5000, step=100)
bedrooms = st.number_input("Enter Number of Bedrooms", min_value=1, max_value=10, step=1)
age = st.number_input("Enter Age of House", min_value=0, max_value=50, step=1)

# Predict button
if st.button("Predict Price"):
    prediction = model.predict([[area, bedrooms, age]])
    st.success(f"Estimated House Price: ₹ {prediction[0]:,.2f}")