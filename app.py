import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load trained model and preprocessing artifacts
model = joblib.load("lightgbm_model.pkl")
scaler = joblib.load("scaler.pkl")
imputer = joblib.load("imputer.pkl")
X_test = joblib.load("X_test.pkl")
y_test = joblib.load("y_test.pkl")
feature_names = joblib.load("feature_names.pkl")

# Ensure X_test is a DataFrame with correct columns
if not isinstance(X_test, pd.DataFrame):
    X_test = pd.DataFrame(X_test, columns=feature_names)

# Streamlit UI setup
st.set_page_config(page_title="Where's the CAT?", page_icon="🧠", layout="wide")
st.title("🧠 Where's the CAT?")
st.markdown("This app predicts the axial position of a CT scan slice based on histogram features of bone and air inclusions.")

# Dropdown to select a sample
sample_idx = st.selectbox("🧬 Choose a sample index to test", options=X_test.index.tolist())

# Visualize the selected sample's features
sample = X_test.loc[sample_idx]
st.subheader("🔍 Feature Overview (Histogram)")
fig, ax = plt.subplots(figsize=(10, 3))
ax.bar(range(len(sample)), sample.values)
ax.set_xlabel("Feature Index")
ax.set_ylabel("Value")
st.pyplot(fig)

# Run prediction
if st.button("Run Prediction"):
    sample_input = sample.values.reshape(1, -1)
    prediction = model.predict(sample_input)[0]
    actual = y_test.iloc[sample_idx]
    error = abs(prediction - actual)

    st.subheader("📈 Prediction Results")
    st.markdown(f"- **Predicted Axial Position:** {prediction:.2f}")
    st.markdown(f"- **Actual Axial Position:** {actual:.2f}")
    st.markdown(f"- **Mean Absolute Error:** {error:.2f}")

    st.subheader("🩻 Position Indicator")
    st.progress(min(int((prediction / 180) * 100), 100))
    st.caption("0 = Top of Head ··· 180 = Bottom of Feet")
