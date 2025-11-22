import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

def main():
    data_path = os.path.join("data", "news.csv")
    print(f"Loading data from:{data_path}")
    df = pd.read_csv(data_path)
    print("Initial shape:", df.shape)
    df = df.dropna(subset=["text","label"])
    label_map = {
        "FAKE": 0,
        "TRUE": 1,
        "fake": 0,
        "true": 1,
        "REAL": 1,
        "real": 1,
    }
    df["label_num"] = df["label"].map(label_map)
    df = df.dropna(subset=["label_num"])
    df["label_num"] = df["label_num"].astype(int)

    X = df["text"].astype(str)
    y = df["label_num"]
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
    print("Vectorization text with TF-IDF...")
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1,2)
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf =vectorizer.transform(X_test)
    print("Training Logistic Regression model...")
    model = LogisticRegression(class_weight= "balanced")
    model.fit(X_train_tfidf, y_train)
    print("Evaluating model...")
    y_pred = model.predict(X_test_tfidf)
    acc= accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.4f}\n")
    print("classification report:\n")
    print(classification_report(y_test, y_pred))
    print("Saving model and vectorizer...")
    joblib.dump(model, "model.pkl") 
    joblib.dump(vectorizer, "vectorizer.pkl")
    print("Saved as model.pkl and vectorizer.pkl")

if __name__ == "__main__":
    main()

