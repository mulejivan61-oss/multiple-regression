import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(page_title="House Price Predictor", layout="wide")

st.title("🏠 House Price Prediction App")

# ---------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------
df = pd.read_csv("multiple_linear_regression_house_dataset(1).csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

# ---------------------------------------------------
# TRAIN MODEL
# ---------------------------------------------------
X = df[['Area_sqft', 'Bedrooms', 'Bathrooms',
        'House_Age', 'Distance_to_City_km']]

y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

# ---------------------------------------------------
# PERFORMANCE
# ---------------------------------------------------
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

st.subheader("Model Performance")
st.write(f"Mean Absolute Error: ₹ {mae:,.2f}")
st.write(f"R² Score: {r2:.2f}")

# ---------------------------------------------------
# USER INPUT
# ---------------------------------------------------
st.subheader("Enter House Details")

col1, col2 = st.columns(2)

with col1:
    area = st.number_input("Area (sqft)", min_value=500, value=1500)
    bedrooms = st.number_input("Bedrooms", min_value=1, value=3)
    bathrooms = st.number_input("Bathrooms", min_value=1, value=2)

with col2:
    house_age = st.number_input("House Age", min_value=0, value=10)
    distance = st.number_input("Distance to City (km)", min_value=0.0, value=5.0)

# ---------------------------------------------------
# PREDICT
# ---------------------------------------------------
if st.button("Predict House Price"):

    input_data = pd.DataFrame([[
        area,
        bedrooms,
        bathrooms,
        house_age,
        distance
    ]], columns=[
        'Area_sqft',
        'Bedrooms',
        'Bathrooms',
        'House_Age',
        'Distance_to_City_km'
    ])

    prediction = model.predict(input_data)[0]

    st.success(f"Predicted House Price: ₹ {prediction:,.2f}")
