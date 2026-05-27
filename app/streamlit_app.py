import sys
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.predict import predict_fit
from src.skills import compare_resume_to_job


st.set_page_config(
    page_title="Resume–Job Fit Classifier",
    page_icon="📄",
    layout="wide",
)


def render_skill_chips(skills, color="#eef2ff"):
    """Render skills as small rounded chips."""
    if not skills:
        st.write("None found.")
        return

    chips_html = ""
    for skill in skills:
        chips_html += (
            f"<span style='"
            f"display:inline-block;"
            f"background-color:{color};"
            f"padding:6px 10px;"
            f"margin:4px;"
            f"border-radius:999px;"
            f"font-size:14px;"
            f"border:1px solid #d4d4d8;"
            f"'>"
            f"{skill}"
            f"</span>"
        )

    st.markdown(chips_html, unsafe_allow_html=True)


def load_example():
    st.session_state["resume_text"] = """
Computer science student with experience in Python, SQL, pandas, scikit-learn,
machine learning, data visualization, Git, and Streamlit. Built end-to-end
data science projects involving data cleaning, exploratory data analysis,
classification models, and interactive dashboards.
"""

    st.session_state["job_description_text"] = """
We are looking for a data analyst with experience in Python, SQL, Excel,
Tableau, statistics, machine learning, and data visualization. The ideal
candidate can clean data, build dashboards, communicate insights, and work
with cloud tools such as AWS.
"""


with st.sidebar:
    st.title("📄 Resume Fit Tool")

    st.write(
        "This app uses NLP and machine learning to estimate whether a resume "
        "matches a job description."
    )

    st.markdown("---")

    st.subheader("What it shows")
    st.write("- Predicted fit category")
    st.write("- Class probabilities")
    st.write("- Matched skills")
    st.write("- Missing job skills")
    st.write("- Skill match score")

    st.markdown("---")

    st.caption(
        "Model: TF-IDF + Logistic Regression. "
        "Skill analysis uses rule-based keyword extraction."
    )


st.title("Resume–Job Fit Classifier")
st.write(
    "Paste a resume and job description to predict whether the resume is a "
    "good, potential, or poor fit for the role."
)

top_col1, top_col2 = st.columns([1, 1])

with top_col1:
    st.button("Load Example", on_click=load_example)

with top_col2:
    st.caption(
        "Tip: Use full resume and job description text for better predictions."
    )

col1, col2 = st.columns(2)

with col1:
    resume_text = st.text_area(
        "Resume Text",
        key="resume_text",
        height=320,
        placeholder="Paste resume text here...",
    )

with col2:
    job_description_text = st.text_area(
        "Job Description Text",
        key="job_description_text",
        height=320,
        placeholder="Paste job description text here...",
    )


st.markdown("---")

if st.button("Predict Fit", type="primary"):
    if not resume_text.strip() or not job_description_text.strip():
        st.warning("Please paste both a resume and a job description.")
    else:
        if len(resume_text.split()) < 30 or len(job_description_text.split()) < 30:
            st.warning(
                "This prediction may be unreliable because the resume or job "
                "description is very short."
            )

        prediction, probabilities = predict_fit(resume_text, job_description_text)

        skill_analysis = compare_resume_to_job(
            resume_text,
            job_description_text,
        )

        st.subheader("Prediction Summary")

        prediction_col, skill_col, missing_col = st.columns(3)

        with prediction_col:
            st.metric("Predicted Fit", prediction)

        with skill_col:
            skill_match_percent = skill_analysis["skill_match_score"] * 100
            st.metric("Skill Match Score", f"{skill_match_percent:.1f}%")

        with missing_col:
            st.metric(
                "Missing Skills",
                len(skill_analysis["missing_skills"]),
            )

        st.progress(float(skill_analysis["skill_match_score"]))

        st.markdown("---")

        prob_col, skill_result_col = st.columns(2)

        with prob_col:
            st.subheader("Class Probabilities")

            if probabilities:
                sorted_probabilities = sorted(
                    probabilities.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )

                for label, prob in sorted_probabilities:
                    st.write(f"**{label}:** {prob * 100:.1f}%")
                    st.progress(float(prob))

        with skill_result_col:
            st.subheader("Skill Match Analysis")

            st.write("**Matched Skills**")
            render_skill_chips(
                skill_analysis["matched_skills"],
                color="#dcfce7",
            )

            st.write("**Missing Skills from Job Description**")
            render_skill_chips(
                skill_analysis["missing_skills"],
                color="#fee2e2",
            )

        with st.expander("View all extracted skills"):
            extracted_col1, extracted_col2 = st.columns(2)

            with extracted_col1:
                st.write("**Resume Skills**")
                render_skill_chips(skill_analysis["resume_skills"])

            with extracted_col2:
                st.write("**Job Description Skills**")
                render_skill_chips(skill_analysis["job_skills"])