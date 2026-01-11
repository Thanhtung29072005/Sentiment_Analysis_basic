import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Load model
model=joblib.load("model/logistic_model.pkl")
vectorizer=joblib.load("model/vectorizer.pkl")

feature_name=vectorizer.get_feature_names_out()
weights=model.coef_[0]
word_sentiments=pd.DataFrame(
    {"word":feature_name,
     "weight":weights}
)
top_positive=word_sentiments.sort_values("weight",ascending=False).head(10)
top_negative=word_sentiments.sort_values("weight").head(10)

# Visualization
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.barh(top_positive["word"],top_positive["weight"],color='green')
plt.title("Top Positive Words")
plt.subplot(1,2,2)
plt.barh(top_negative["word"],top_negative["weight"],color='red')
plt.title("Top Negative Words")
plt.tight_layout()
plt.show()

