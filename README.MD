# Resume–Job Fit Classifier

This project uses NLP and machine learning to predict whether a resume is a good fit for a job description. It trains a baseline TF-IDF + Logistic Regression classifier on resume/job description pairs and provides a Streamlit app for interactive predictions.

## Features
- Loads resume-job description pairs from Hugging Face
- Combines resume and job description text for NLP classification
- Trains a TF-IDF + Logistic Regression baseline model
- Evaluates model performance on a held-out test set
- Provides an interactive Streamlit app for predictions

## Tech Stack
Python, pandas, scikit-learn, Hugging Face datasets, Streamlit

## Future Improvements
- Add skill extraction
- Add missing skill recommendations
- Compare TF-IDF with sentence-transformer embeddings
- Add SHAP/LIME explainability
- Deploy the app