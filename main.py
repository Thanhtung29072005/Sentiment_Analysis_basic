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

# Phân tích review most positive,most negative
df["positive_prod"]=model.predict_proba(X)[:,1]
most_positive=df.sort_values("positive_prod",ascending=False).iloc[0]
most_negative=df.sort_values("positive_prod").iloc[0]
print("Most positive review probability: ",most_positive["positive_prod"])
print(most_positive["review"])
print("Most negative review probability: ",most_negative["positive_prod"])
print(most_negative["review"])