"""Streamlit demo: run breast_cancer_diagnosis.py first (creates outputs/final_model.joblib)."""
import joblib, pandas as pd, streamlit as st
from sklearn.datasets import load_breast_cancer

st.title("Breast Cancer Diagnosis Predictor")
st.caption("Learning project - not a medical device.")
model = joblib.load("outputs/final_model.joblib")
data = load_breast_cancer(as_frame=True).frame.drop(columns="target")
vals = {}
for c in data.columns[:10]:  # first 10 'mean' features are editable; the rest use dataset averages
    vals[c] = st.slider(c, float(data[c].min()), float(data[c].max()), float(data[c].mean()))
row = data.mean().to_dict(); row.update(vals)
if st.button("Predict"):
    p = model.predict_proba(pd.DataFrame([row])[data.columns])[0, 1]
    st.metric("Probability malignant", f"{p:.1%}")
