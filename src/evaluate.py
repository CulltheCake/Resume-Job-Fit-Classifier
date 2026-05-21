from pathlib import Path

import joblib
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "tfidf_logreg_resume_fit.pkl"
OUTPUT_DIR = ROOT_DIR / "outputs"
REPORT_PATH = OUTPUT_DIR / "evaluation_report.txt"

def main(): 
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Loading model...")
    model = joblib.load(MODEL_PATH)

    print("Loading dataset...")
    dataset = load_dataset("cnamuangtoun/resume-job-description-fit")
    test_df = pd.DataFrame(dataset["test"])

    test_df["combined_text"] = (
        "Resume: "
        + test_df["resume_text"].fillna("")
        + "\n\nJob Description: "
        + test_df["job_description_text"].fillna("")
    )

    X_test = test_df["combined_text"]
    y_test = test_df["label"]

    print("Runiing predictions...")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred, labels=model.classes_)

    
    output = f"""
Resume–Job Fit Classifier Evaluation Report

Model:
TF-IDF + Logistic Regression

Accuracy:
{accuracy:.4f}

Classes:
{list(model.classes_)}

Classification Report:
{report}

Confusion Matrix:
Rows = Actual Labels
Columns = Predicted Labels

{pd.DataFrame(matrix, index=model.classes_, columns=model.classes_)}
"""

    print(output)

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        file.write(output)

        print(f"Saved evaluation report to {REPORT_PATH}")


if __name__ == "__main__":
    main()