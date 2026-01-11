import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df=pd.read_csv("IMDB_Dataset.csv")
df["sentiment"]=df["sentiment"].map({"positive":1,"negative":0})
X_text=df["review"]
y=df["sentiment"]

vectorizer=CountVectorizer(max_features=2000,stop_words="english")
X=vectorizer.fit_transform(X_text)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

# Train
model=LogisticRegression(max_iter=1000)
model.fit(X_train,y_train)

# Evaluate
y_pred=model.predict(X_test)
print("Accuracy: ",accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))

# Save model
joblib.dump(model,"model/logistic_model.pkl")
joblib.dump(vectorizer,"model/vectorizer.pkl")
