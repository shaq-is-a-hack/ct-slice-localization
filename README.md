# Where’s that CAT?

A Streamlit app that predicts the axial position of a CAT scan slice (from head to toe) using a LightGBM model.

Built as part of a machine learning assignment in undergrad mechanical engineering.

## 🧠 What it does

- Uses 384 features per CAT slice (bone + air histograms)
- Predicts position on a scale from 0 (head) to 180 (feet)
- Visualizes features, prediction, and error

## 🔗 Live app

Try it here:  
[https://i-scan-explain.streamlit.app](https://i-scan-explain.streamlit.app)

## 📁 Dataset

Data sourced from Kaggle:  
[CT Slice Localization Dataset – UCIML](https://www.kaggle.com/datasets/uciml/ct-slice-localization/data)

Each slice includes directional bone and air features, with labels based on estimated axial position.

## 🛠 Run locally

```bash
git clone https://github.com/shaq-is-a-hack/ct-slice-localization
cd ct-slice-localization
pip install -r requirements.txt
streamlit run app.py
