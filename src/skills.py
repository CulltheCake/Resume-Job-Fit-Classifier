import re
from typing import Dict, List


SKILLS = [
    "python",
    "java",
    "javascript",
    "typescript",
    "c++",
    "c#",
    "sql",
    "html",
    "css",
    "r",
    "go",
    "rust",
    "scala",
    "pandas",
    "numpy",
    "scikit-learn",
    "sklearn",
    "matplotlib",
    "seaborn",
    "plotly",
    "tensorflow",
    "pytorch",
    "keras",
    "spark",
    "hadoop",
    "airflow",
    "dbt",
    "machine learning",
    "deep learning",
    "natural language processing",
    "nlp",
    "computer vision",
    "statistics",
    "data analysis",
    "data visualization",
    "data engineering",
    "etl",
    "a/b testing",
    "hypothesis testing",
    "regression",
    "classification",
    "clustering",
    "time series",
    "aws",
    "azure",
    "gcp",
    "docker",
    "kubernetes",
    "git",
    "github",
    "linux",
    "excel",
    "tableau",
    "power bi",
    "looker",
    "snowflake",
    "postgresql",
    "mysql",
    "mongodb",
    "fastapi",
    "flask",
    "streamlit",
    "react",
    "node.js",
]


def normalize_text(text: str) -> str:
    """Lowercase text and normalize spacing."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text


def extract_skills(text: str) -> List[str]:
    """Extract known technical skills from text."""
    normalized_text = normalize_text(text)
    found_skills = []

    for skill in SKILLS:
        pattern = r"\b" + re.escape(skill.lower()) + r"\b"

        if re.search(pattern, normalized_text):
            found_skills.append(skill)

    return sorted(set(found_skills))


def compare_resume_to_job(
    resume_text: str, job_description_text: str
) -> Dict[str, object]:
    """Compare resume skills against job description skills."""
    resume_skills = set(extract_skills(resume_text))
    job_skills = set(extract_skills(job_description_text))

    matched_skills = sorted(resume_skills.intersection(job_skills))
    missing_skills = sorted(job_skills.difference(resume_skills))

    if job_skills:
        skill_match_score = len(matched_skills) / len(job_skills)
    else:
        skill_match_score = 0.0

    return {
        "resume_skills": sorted(resume_skills),
        "job_skills": sorted(job_skills),
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "skill_match_score": skill_match_score,
    }