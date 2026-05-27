import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.set_page_config(page_title="House Price Prediction", layout="wide")

st.title("🏠 House Price Prediction App")

uploaded_file = st.file_uploader(
    "Upload House Dataset CSV",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    X = df[['Area_sqft', 'Bedrooms', 'Bathrooms',
            'House_Age', 'Distance_to_City_km']]

    y = df['Price']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    score = r2_score(y_test, predictions)

    st.write(f"Model Accuracy (R² Score): {score:.2f}")

    st.subheader("Enter House Details")

    area = st.number_input("Area_sqft", value=1500)
    bedrooms = st.number_input("Bedrooms", value=3)
    bathrooms = st.number_input("Bathrooms", value=2)
    house_age = st.number_input("House_Age", value=10)
    distance = st.number_input("Distance_to_City_km", value=5.0)

    if st.button("Predict Price"):

        input_data = pd.DataFrame([[
            area, bedrooms, bathrooms,
            house_age, distance
        ]], columns=[
            'Area_sqft',
            'Bedrooms',
            'Bathrooms',
            'House_Age',
            'Distance_to_City_km'
        ])

        prediction = model.predict(input_data)[0]

        st.success(f"Predicted House Price: ₹{prediction:,.2f}")

else:
    st.info("Please upload dataset CSV file")
