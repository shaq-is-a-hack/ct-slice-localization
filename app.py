import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# --- Load Assets ---
model = joblib.load("lightgbm_model.pkl")
scaler = joblib.load("scaler.pkl")
imputer = joblib.load("imputer.pkl")
X_test = joblib.load("X_test.pkl")
y_test = joblib.load("y_test.pkl")
feature_names = joblib.load("feature_names.pkl")

# --- Streamlit Page Settings ---
st.set_page_config(page_title="I Scan Explain", layout="wide")
st.title("🧠 I Scan Explain")
st.markdown(
    "Predict the **axial location** of a CT scan slice using histogram features derived from bone and air structures."
)

# --- Sample Selector ---
index = st.selectbox("Choose a sample index to visualize and predict:", X_test.index.tolist())

sample = X_test.loc[index]
true = y_test[index]
predicted = model.predict([sample])[0]
mae = abs(predicted - true)

# --- Feature Visualization ---
st.subheader("Feature Histogram")
fig, ax = plt.subplots(figsize=(12, 4))
ax.bar(range(len(sample)), sample.values)
ax.set_title("Bone & Air Histogram Bins")
ax.set_xlabel("Feature Bin Index")
ax.set_ylabel("Normalized Value")
st.pyplot(fig)

# --- Output Section ---
st.subheader("Prediction Summary")
col1, col2, col3 = st.columns(3)
col1.metric("Predicted Axial Pos.", f"{predicted:.2f}")
col2.metric("True Position", f"{true:.2f}")
col3.metric("MAE", f"{mae:.2f}")

# --- Position Visualization ---
st.subheader("🧍 Human Body Axial Scale")
fig2, ax2 = plt.subplots(figsize=(10, 1))
ax2.plot([0, 180], [0, 0], color="lightgray", linewidth=10)
ax2.scatter([predicted], [0], color="blue", label="Predicted", s=120, zorder=3)
ax2.scatter([true], [0], color="green", label="Actual", s=120, zorder=3)
ax2.set_xlim(0, 180)
ax2.set_yticks([])
ax2.set_xlabel("0 = Head   |   180 = Feet")
ax2.legend(loc="upper right")
st.pyplot(fig2)
