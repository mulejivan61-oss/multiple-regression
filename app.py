# =========================================================
# MANUAL HOUSE INPUT SECTIONS
# =========================================================

st.subheader("Enter House Details")

col1, col2 = st.columns(2)

with col1:
    bedrooms = st.number_input(
        "Bedrooms",
        min_value=1,
        max_value=20,
        value=3
    )

    bathrooms = st.number_input(
        "Bathrooms",
        min_value=1,
        max_value=20,
        value=2
    )

with col2:
    house_age = st.number_input(
        "House_Age",
        min_value=0,
        max_value=100,
        value=10
    )

    distance_to_city = st.number_input(
        "Distance_to_City_km",
        min_value=0.0,
        value=5.0
    )

# Create input dataframe
input_data = pd.DataFrame([{
    "Bedrooms": bedrooms,
    "Bathrooms": bathrooms,
    "House_Age": house_age,
    "Distance_to_City_km": distance_to_city
}])

# Ensure same columns as training data
input_data = input_data.reindex(
    columns=X.columns,
    fill_value=0
)
