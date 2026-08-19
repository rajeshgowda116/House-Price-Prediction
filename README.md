# House Price Prediction

This project is a machine learning app for predicting California house prices from housing details such as location, room count, population, median income, and ocean proximity.

The trained model is served through a Streamlit interface in `strem.py`.

## Project Files

- `strem.py` - Streamlit web app for entering house details and getting predictions.
- `House.ipynb` - Notebook used for data exploration and model training.
- `housing.csv` - Housing dataset.
- `house_price_model.pkl` - Trained house price prediction model.
- `scaler.pkl` - Saved feature scaler used before prediction.
- `columns.pkl` - Saved training columns used to align app inputs with the model.
- `house.txt` - Project notes and model-building guidance.

## Requirements

Install the required Python packages:

```bash
pip install streamlit pandas scikit-learn joblib
```

## Run the App

From the project folder, run:

```bash
streamlit run strem.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## How It Works

1. The app loads the trained model, scaler, and saved training columns.
2. The user enters house details in the Streamlit form.
3. The selected ocean proximity value is one-hot encoded.
4. The input columns are aligned to match the training data.
5. The scaler transforms the input data.
6. The model predicts the estimated house value.

## Inputs

The app currently accepts:

- Longitude
- Latitude
- Housing median age
- Total rooms
- Total bedrooms
- Population
- Households
- Median income
- Ocean proximity

## Notes

Keep `house_price_model.pkl`, `scaler.pkl`, and `columns.pkl` in the same folder as `strem.py`, because the app loads them using relative paths.
