from pathlib import Path

import joblib
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "models"
MODEL_PATH = MODEL_DIR / "tfidf_logreg_resume_fit.pkl"


def main():
    MODEL_DIR.mkdir(exist_ok=True)

    print("Loading dataset...")
    dataset = load_dataset("cnamuangtoun/resume-job-description-fit")

    train_df = pd.DataFrame(dataset["train"])
    test_df = pd.DataFrame(dataset["test"])

    print("Train columns:", train_df.columns.tolist())

    train_df["combined_text"] = (
        "Resume: "
        + train_df["resume_text"].fillna("")
        + "\n\nJob Description: "
        + train_df["job_description_text"].fillna("")
    )

    test_df["combined_text"] = (
        "Resume: "
        + test_df["resume_text"].fillna("")
        + "\n\nJob Description: "
        + test_df["job_description_text"].fillna("")
    )

    X_train = train_df["combined_text"]
    y_train = train_df["label"]

    X_test = test_df["combined_text"]
    y_test = test_df["label"]

    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    max_features=20000,
                    ngram_range=(1, 2),
                ),
            ),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )

    print("Training model...")
    model.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    print(f"Saving model to {MODEL_PATH}")
    joblib.dump(model, MODEL_PATH)

    print("Done!")


if __name__ == "__main__":
    main()