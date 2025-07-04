# Where’s that CAT?

A Streamlit app that predicts the axial position of a CAT scan slice (from head to toe) using a LightGBM model.

Built as part of a machine learning assignment in undergrad mechanical engineering.

## What it does

- Uses 384 features per CAT slice (bone + air histograms)
- Predicts position on a scale from 0 (head) to 180 (feet)
- Visualizes features, prediction, and error

## Live app

Try it here:  
[https://i-scan-explain.streamlit.app](https://i-scan-explain.streamlit.app)

## Dataset

Data sourced from Kaggle:  
[CT Slice Localization Dataset – UCIML](https://www.kaggle.com/datasets/uciml/ct-slice-localization/data)

Each slice includes directional bone and air features, with labels based on estimated axial position.

## 🛠 Run locally

```bash
git clone https://github.com/shaq-is-a-hack/ct-slice-localization
cd ct-slice-localization
pip install -r requirements.txt
streamlit run app.py
```

## Notes
The terms "CAT" and "CT" are used interchangeably in this project to refer to the same medical imaging procedure, Computed Tomography. We’re aware that there’s probably a technical difference, but it’s minor for the purposes of this project. We are not medical professionals.

If you are a rando on the internet who stumbled upon this group assignment for my uni coursework, here's a disclaimer:
This project is meant for educational and academic purposes only. It is not designed, validated or approved for clinical or diagnostic use. DO NOT use this to inform medical decisions despite the profile picture being Dr. Gregory House from the hit 2000s TV series House M.D., and definitely DO NOT sue us. This being an undergrad project should be enough to imply amount of money the team members have to their names.

