import streamlit as st
import pandas as pd
import joblib

# Load model, scaler and columns
model = joblib.load("house_price_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

st.title("🏠 California House Price Prediction")

st.write("Enter the house details below:")

# Inputs
longitude = st.number_input(
    "Longitude",
    value=-122.23
)

latitude = st.number_input(
    "Latitude",
    value=37.88
)

housingMedianAge = st.number_input(
    "Housing Median Age",
    min_value=1.0,
    value=30.0
)

totalRooms = st.number_input(
    "Total Rooms",
    min_value=1.0,
    value=1000.0
)

totalBedrooms = st.number_input(
    "Total Bedrooms",
    min_value=1.0,
    value=200.0
)

population = st.number_input(
    "Population",
    min_value=1.0,
    value=500.0
)

households = st.number_input(
    "Households",
    min_value=1.0,
    value=150.0
)

medianIncome = st.number_input(
    "Median Income",
    min_value=0.0,
    value=4.0
)

oceanProximity = st.selectbox(
    "Ocean Proximity",
    [
        "<1H OCEAN",
        "INLAND",
        "ISLAND",
        "NEAR BAY",
        "NEAR OCEAN"
    ]
)

# Predict button
if st.button("Predict House Price"):

    # Create input dataframe
    input_data = pd.DataFrame({
        "longitude": [longitude],
        "latitude": [latitude],
        "housingMedianAge": [housingMedianAge],
        "totalRooms": [totalRooms],
        "totalBedrooms": [totalBedrooms],
        "population": [population],
        "households": [households],
        "medianIncome": [medianIncome],
        "oceanProximity": [oceanProximity]
    })

    # One-hot encode categorical column
    input_data = pd.get_dummies(
        input_data,
        columns=["oceanProximity"]
    )

    # Match training columns
    input_data = input_data.reindex(
        columns=columns,
        fill_value=0
    )

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)

    # Display result
    st.success(
        f"Predicted House Value: ${prediction[0]:,.2f}"
    )