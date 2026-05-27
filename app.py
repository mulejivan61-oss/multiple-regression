# =========================================================
# HOUSE PRICE PREDICTION STREAMLIT APP
# =========================================================

# IMPORT LIBRARIES
import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# PAGE CONFIG
st.set_page_config(
    page_title="House Price Prediction",
    layout="wide"
)

# TITLE
st.title("🏠 House Price Prediction App")
st.write("Upload CSV, XLSX, or XLS dataset file")

# FILE UPLOADER
uploaded_file = st.file_uploader(
    "Upload Dataset",
    type=["csv", "xlsx", "xls"]
)

# =========================================================
# PROCESS FILE
# =========================================================

if uploaded_file is not None:

    try:

        # READ FILE
        file_name = uploaded_file.name.lower()

        if file_name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)

        elif file_name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file, engine="openpyxl")

        elif file_name.endswith(".xls"):
            df = pd.read_excel(uploaded_file, engine="xlrd")

        st.success("File uploaded successfully!")

        # DATA PREVIEW
        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        # DATASET INFO
        st.subheader("Dataset Information")
        st.write("Rows:", df.shape[0])
        st.write("Columns:", df.shape[1])

        # TARGET COLUMN
        st.subheader("Select Target Column")

        target_column = st.selectbox(
            "Choose House Price Column",
            df.columns
        )

        # FEATURES & TARGET
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Convert categorical columns
        X = pd.get_dummies(X)

        # TRAIN TEST SPLIT
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42
        )

        # MODEL
        model = LinearRegression()
        model.fit(X_train, y_train)

        # PREDICTIONS
        predictions = model.predict(X_test)

        # PERFORMANCE
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        st.subheader("Model Performance")
        st.write(f"Mean Absolute Error: {mae:.2f}")
        st.write(f"R2 Score: {r2:.2f}")

        # RESULTS
        results_df = pd.DataFrame({
            "Actual Price": y_test.values,
            "Predicted Price": predictions
        })

        st.subheader("Prediction Results")
        st.dataframe(results_df.head(20))

        # =====================================================
        # MANUAL PREDICTION
        # =====================================================

        st.subheader("Predict New House Price")

        col1, col2 = st.columns(2)

        with col1:
            bedrooms = st.number_input(
                "Bedrooms",
                min_value=1,
                value=3
            )

            bathrooms = st.number_input(
                "Bathrooms",
                min_value=1,
                value=2
            )

        with col2:
            house_age = st.number_input(
                "House_Age",
                min_value=0,
                value=10
            )

            distance_to_city = st.number_input(
                "Distance_to_City_km",
                min_value=0.0,
                value=5.0
            )

        # INPUT DATA
        input_data = pd.DataFrame([{
            "Bedrooms": bedrooms,
            "Bathrooms": bathrooms,
            "House_Age": house_age,
            "Distance_to_City_km": distance_to_city
        }])

        # Match columns
        input_data = input_data.reindex(
            columns=X.columns,
            fill_value=0
        )

        # PREDICT BUTTON
        if st.button("Predict House Price"):

            predicted_price = model.predict(input_data)

            st.success(
                f"Predicted House Price: ₹ {predicted_price[0]:,.2f}"
            )

    except Exception as e:
        st.error("Error while processing file")
        st.write(e)

else:
    st.info("Please upload CSV, XLSX, or XLS file")
