import sys
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.predict import predict_fit

st.title("Resume–Job Fit Classifier")

st.write(
    "Paste a resume and job description to predict whether the resume is a good fit."
)

resume_text = st.text_area("Resume Text", height=250)
job_description_text = st.text_area("Job Description Text", height=250)

if st.button("Predict Fit"):
    if not resume_text.strip() or not job_description_text.strip():
        st.warning("Please paste both a resume and a job description.")
    else:
        prediction, probabilities = predict_fit(resume_text, job_description_text)

        st.subheader("Prediction")
        st.write(f"**{prediction}**")

        if probabilities:
            st.subheader("Class Probabilities")
            st.json({
                label: round(float(prob), 4)
                for label, prob in probabilities.items()
            })