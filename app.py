import pandas as pd
import numpy as np
import joblib
import streamlit as st
import matplotlib.pyplot as plt

# Load model and preprocessing tools
model = joblib.load("lightgbm_model.pkl")
scaler = joblib.load("scaler.pkl")
imputer = joblib.load("imputer.pkl")
X_test = joblib.load("X_test.pkl")
y_test = joblib.load("y_test.pkl")
feature_names = joblib.load("feature_names.pkl")

# Set up page
st.set_page_config(page_title="Where's the CAT?", layout="wide")
st.title("🧠 Where’s the CAT?")
st.markdown("""
This app predicts the **position of a CT slice** along the head-to-toe axis based on patterns of **bone density** and **air presence**.

Pick a sample, view the directional features, and check how close the model's guess is.
""")

# Sidebar sample selector
sample_idx = st.sidebar.selectbox("Pick a CT Slice Sample", range(len(X_test)), format_func=lambda i: f"Sample #{i}")

# Extract selected sample
sample_features = pd.Series(X_test[sample_idx], index=feature_names)
actual_position = y_test[sample_idx]

# Prepare histogram data
bone_bins = sample_features[:241]
air_bins = sample_features[241:]

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(range(1, 242), bone_bins, label="Bone (Density)", alpha=0.7)
ax.bar(range(242, 385), air_bins, label="Air (Presence)", alpha=0.7)
ax.set_xlabel("Direction Bin (Spread around CT slice)")
ax.set_ylabel("Feature Value")
ax.set_title("Direction-Based Features from CT Slice")
ax.legend()
st.pyplot(fig)

# Prediction button
if st.button("🔍 Run Prediction"):
    sample_array = np.array(sample_features).reshape(1, -1)
    predicted = model.predict(sample_array)[0]

    st.success(f"📍 **Predicted Axial Position:** {predicted:.2f}")
    st.info(f"📌 **Actual Axial Position:** {actual_position:.2f}")

    # Visual gauge
    st.markdown("### Estimated Position Along Body")
    st.progress(min(max(int((predicted / 180) * 100), 0), 100))

# Optional expandable info
with st.expander("ℹ️ What does this data represent?", expanded=False):
    st.markdown("""
Each CT scan is described using two sets of features:
- **Bone**: Measures where denser bone material appears around the slice (like skull or spine).
- **Air**: Measures presence of air in various directions (like sinuses or lungs).

These are captured as 384 directional readings around each CT slice. The model uses this to guess where the image was taken — top of the head is 0, bottom of the feet is 180.
""")
