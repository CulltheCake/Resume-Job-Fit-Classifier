import sys
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.predict import predict_fit
from src.skills import compare_resume_to_job

st.title("Resume–Job Fit Classifier")

st.write(
    "Paste a resume and job description to predict whether the resume is a good, "
    "potential, or poor fit."
)

resume_text = st.text_area("Resume Text", height=250)
job_description_text = st.text_area("Job Description Text", height=250)

if st.button("Predict Fit"):
    if not resume_text.strip() or not job_description_text.strip():
        st.warning("Please paste both a resume and a job description.")
    else:
        if len(resume_text.split()) < 30 or len(job_description_text.split()) < 30:
            st.warning(
                "This prediction may be unreliable because the resume or job "
                "description is very short."
            )

        prediction, probabilities = predict_fit(resume_text, job_description_text)

        st.subheader("Prediction")
        st.write(f"**{prediction}**")

        if probabilities:
            st.subheader("Class Probabilities")

            for label, prob in probabilities.items():
                st.write(f"**{label}:** {prob * 100:.1f}%")
                st.progress(float(prob))

        skill_analysis = compare_resume_to_job(
            resume_text,
            job_description_text,
        )

        st.subheader("Skill Match Analysis")

        skill_match_percent = skill_analysis["skill_match_score"] * 100
        st.write(f"**Skill Match Score:** {skill_match_percent:.1f}%")
        st.progress(float(skill_analysis["skill_match_score"]))

        st.write("**Matched Skills:**")
        if skill_analysis["matched_skills"]:
            st.write(", ".join(skill_analysis["matched_skills"]))
        else:
            st.write("No matched skills found.")

        st.write("**Missing Skills from Job Description:**")
        if skill_analysis["missing_skills"]:
            st.write(", ".join(skill_analysis["missing_skills"]))
        else:
            st.write("No missing skills found.")

        with st.expander("View extracted skills"):
            st.write("**Resume Skills:**")
            st.write(", ".join(skill_analysis["resume_skills"]) or "None found")

            st.write("**Job Description Skills:**")
            st.write(", ".join(skill_analysis["job_skills"]) or "None found")