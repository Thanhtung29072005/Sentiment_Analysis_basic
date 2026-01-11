import joblib

# Load model
model=joblib.load("model/logistic_model.pkl")
vectorizer=joblib.load("model/vectorizer.pkl")

def predict_review_sentiment(review_text,vectorizer,model):
    X_review=vectorizer.transform([review_text])
    positive_prod=model.predict_proba(X_review)[0,1]
    return positive_prod
# Test
review = "This movie was absolutely fantastic"
prob = predict_review_sentiment(review,vectorizer,model)

print("Positive probability:", prob)