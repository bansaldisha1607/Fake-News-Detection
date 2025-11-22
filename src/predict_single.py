import joblib

print("Loading model and vectorizer...")
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

print("\nEnter a news article or headline to classify.")
print("Type 'exit' to quit.\n")
while True:
    text = input("News text:")
    if text.strip().lower() == "exit":
        break
    X_tfidf = vectorizer.transform([text])
    pred = model.predict(X_tfidf)[0]
    if int(pred) == 1:
        label = "REAL"
    else:
        label = "FAKE"
    print("Prediction:", label)
    print()
