import pandas as pd
import joblib

df=pd.read_csv("IMDB_Dataset.csv")
df["sentiment"]=df["sentiment"].map({"positive":1,"negative":0})
# Load model
model=joblib.load("model/logistic_model.pkl")
vectorizer=joblib.load("model/vectorizer.pkl")

X=vectorizer.transform(df["review"])
# Phân tích review most positive,most negative
df["positive_prod"]=model.predict_proba(X)[:,1]
most_positive=df.sort_values("positive_prod",ascending=False).iloc[0]
most_negative=df.sort_values("positive_prod").iloc[0]
print("Most positive review probability: ",most_positive["positive_prod"])
print(most_positive["review"])
print()
print("Most negative review probability: ",most_negative["positive_prod"])
print(most_negative["review"])