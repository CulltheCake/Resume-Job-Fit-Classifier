import joblib

MODEL_PATH = "models/tfidf_logreg_resume_fit.pkl"

model = joblib.load(MODEL_PATH)

def predict_fit(resume_text: str, job_description_text: str):
    combined_text = (
        "Resume: " + resume_text +
        "\n\nJob Description: " + job_description_text
    )

    prediction = model.predict([combined_text])[0]

    if hasattr(model.named_steps["clf"], "predict_proba"):
        probabilities = model.predict_proba([combined_text])[0]
        classes = model.classes_
        probability_dict = dict(zip(classes, probabilities))
    else:
        probability_dict = {}

    return prediction, probability_dict