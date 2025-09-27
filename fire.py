import streamlit as st
import pandas as pd
import numpy as np

try:
    import joblib
    scaler = joblib.load('scaler.pkl')
    model = joblib.load('fire_detection_model.pkl')
    #st.success("Model and Scaler loaded successfully. Ready for prediction.")
    
except FileNotFoundError:
    st.error("Model or Scaler files not found! Using mock objects for demonstration.")
    st.warning("To use your real model, place 'scaler.pkl' and 'model.pkl' in this script's directory.")
    
    
def predict_fire(temperature, ffmc, dmc, dc, isi):
    """
    Scales the user input and predicts the class (0 or 1).
    """

    input_data = pd.DataFrame([[temperature, ffmc, dmc, dc, isi]],
                              columns=['Temperature', 'FFMC', 'DMC', 'DC', 'ISI'])

    scaled_input = scaler.transform(input_data)
    
    prediction = model.predict(scaled_input)
    
    if prediction[0] == 1:
        return 1, "Fire 🔥"
    else:
        return 0, "No Fire 🌳"

# --- STREAMLIT UI SETUP ---

st.set_page_config(page_title="Forest Fire Prediction", layout="centered")


st.markdown("""
    <style>
    .header-font {
        font-size:30px !important;
        font-weight: bold;
        color: #B33A3A;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-top: 30px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .fire {
        background-color: #ffe0e0;
        border: 2px solid #ff4d4d;
    }
    .no-fire {
        background-color: #e0fff0;
        border: 2px solid #4dff99;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<p class="header-font">Forest Fire Risk Predictor</p>', unsafe_allow_html=True)
st.write("Enter the weather and fire danger index values to predict the risk of fire.")

with st.form("prediction_form"):
    st.subheader("Input Features")
    
    col1, col2 = st.columns(2)
    
    # Input fields
    temperature = col1.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)
    ffmc = col2.number_input("FFMC (Fine Fuel Moisture Code)", min_value=0.0, max_value=100.0, value=85.0)
    dmc = col1.number_input("DMC (Duff Moisture Code)", min_value=0.0, max_value=200.0, value=50.0)
    dc = col2.number_input("DC (Drought Code)", min_value=0.0, max_value=1000.0, value=600.0)
    isi = st.number_input("ISI (Initial Spread Index)", min_value=0.0, max_value=50.0, value=10.0)
    
    col_button_left, col_button_mid, col_button_right = st.columns([1, 2, 1]) 
    
    with col_button_mid:
        submitted = st.form_submit_button("Predict Risk")


if submitted:
    try:
        prediction_num, prediction_label = predict_fire(temperature, ffmc, dmc, dc, isi)
        
        class_style = "fire" if prediction_num == 1 else "no-fire"
        
        st.markdown(f"""
            <div class='result-box {class_style}'>
                <p>The predicted classification is:</p>
                <p style='font-size: 40px; font-weight: bold;'>{prediction_label}</p>
                <p style='font-size: small; color: gray;'>Raw Model Output: {prediction_num}</p>
            </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.exception(f"An error occurred during prediction. Please check your file paths and feature order: {e}")

