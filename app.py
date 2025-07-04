import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import base64

# --- Load and encode logo image ---
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

logo_base64 = get_base64_image("logo.png")  # Make sure 'logo.png' is in the same folder

# --- Load assets ---
model = joblib.load("lightgbm_model.pkl")
scaler = joblib.load("scaler.pkl")
imputer = joblib.load("imputer.pkl")
X_test = joblib.load("X_test.pkl")
y_test = joblib.load("y_test.pkl")
feature_names = joblib.load("feature_names.pkl")

st.set_page_config(page_title="Where’s that CAT?", layout="centered")

# --- Header with Logo and GitHub Button ---
st.markdown(
    f"""
    <div style='display: flex; align-items: center; justify-content: space-between; margin-bottom: 1.5rem;'>
        <div style='display: flex; align-items: center; gap: 1rem;'>
            <img src="data:image/png;base64,{logo_base64}" width="50"/>
            <h1 style='margin: 0; font-size: 2rem;'>Where’s that CAT?</h1>
        </div>
        <div>
            <a href='https://github.com/shaq-is-a-hack/ct-slice-localization.git' target='_blank'>
                <button style='padding: 0.3rem 0.6rem; font-size: 0.8rem; background-color: #f0f0f0; border: none; border-radius: 5px;'>View on GitHub</button>
            </a>
        </div>
    </div>
    <div style='font-size: 0.95rem; color: #444; margin-bottom: 1rem;'>
        This tool predicts the axial location of a CAT scan slice along the head-to-toe body axis based on directional features of bone density and air presence.
    </div>
    """,
    unsafe_allow_html=True
)

# --- Initialize sample index in session state ---
if "random_idx" not in st.session_state:
    st.session_state.random_idx = 0

# --- Sample Selection ---
st.markdown("#### 🩻 Sample Selection")
sample_idx = st.selectbox(
    "Choose a CAT Scan sample:",
    options=range(len(X_test)),
    index=st.session_state.random_idx,
    format_func=lambda x: f"Image #{x}"
)

# --- Random Sample Button ---
if st.button("Pick a Random Sample"):
    st.session_state.random_idx = np.random.randint(0, len(X_test))
    st.rerun()


sample_features = X_test.iloc[sample_idx]
actual = y_test[sample_idx]

# --- Plotting ---
fig, ax = plt.subplots(figsize=(12, 4))

# Separate bin ranges
bone_bins = sample_features[:241]
air_bins = sample_features[241:]

# X-axis ranges to avoid overlap
x_bone = range(len(bone_bins))                     # 0 to 240
x_air = range(len(bone_bins), len(bone_bins) + len(air_bins))  # 241 to 241+143

# Plot without overlap
ax.bar(x_bone, bone_bins, label="Bone (Density)", color='#4a90e2', alpha=0.8)
ax.bar(x_air, air_bins, label="Air (Presence)", color='#f5a623', alpha=0.6)

ax.set_title("Direction-Based Features from CT Slice", fontsize=14)
ax.set_xlabel("Direction Bin (Spread around CT slice)")
ax.set_ylabel("Feature Value")
ax.legend()
st.pyplot(fig)


# --- Prediction Section ---
if st.button("🔍 Run Prediction", type="primary"):
    st.markdown("#### 📊 Model Prediction")

    # --- Progress bar: Estimated position ---
    predicted_position = model.predict([sample_features])[0]
    # --- Custom Horizontal Position Bar ---
    st.markdown("""
    <div style='margin-top: 0.1rem; margin-bottom: 0.5rem; font-size: 1rem; font-weight: 500;'>Estimated Position Along Body</div>
    <div style='position: relative; height: 30px; background: linear-gradient(to right, #dceefb, #fde2e4); border-radius: 15px;'>
        <div style='position: absolute; top: 0; bottom: 0; left: {left_pct}%; width: 4px; background-color: #1e88e5;'></div>
    </div>
    <div style='display: flex; justify-content: space-between; font-size: 0.85rem; margin-top: 0.3rem; margin-bottom: 0.5rem; color: #666;'>
        <span>Head (0)</span>
        <span>Feet (180)</span>
    </div>
    """.replace("{left_pct}", f"{(predicted_position / 180) * 100:.2f}"), unsafe_allow_html=True)

    mae_sample = abs(predicted_position - actual)
    rmse_sample = np.sqrt((predicted_position - actual) ** 2)

    st.markdown("""
    <div style='display: flex; gap: 1rem;'>
        <div style='flex: 1; background-color: #e9f7ef; padding: 1rem; border-radius: 10px; text-align: center;'>
            <div style='font-weight: bold; color: #2e7d32;'>Predicted Axial Position</div>
            <div style='font-size: 1.5rem; font-weight: bold;'>{pred:.2f}</div>
        </div>
        <div style='flex: 1; background-color: #e3f2fd; padding: 1rem; border-radius: 10px; text-align: center;'>
            <div style='font-weight: bold; color: #1565c0;'>Actual Axial Position</div>
            <div style='font-size: 1.5rem; font-weight: bold;'>{act:.2f}</div>
        </div>
    </div>
    """.format(pred=predicted_position, act=actual), unsafe_allow_html=True)

    st.markdown("#### 📌 Prediction Error")
    st.markdown(f"**MAE**: `{mae_sample:.4f}`")

# --- Info Expander ---
with st.expander("💡 What does this data represent?"):
    st.markdown(
        """
Each CT slice is converted into 384 values that represent **directional information**.  
- The first 241 values (blue bars) reflect **bone structures** around the slice.  
- The next 143 values (orange bars) reflect **air presence**, like lungs or air cavities.  
- The model uses these patterns to estimate where in the body the slice comes from.

Axial position values range from:  
- `0`: top of the head  
- `180`: bottom of the feet  

**Note:** "CAT" and "CT" are used interchangeably to refer to the medical imaging procedure of Computed Tomography.
        """
    )

