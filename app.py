import pandas as pd
import numpy as np
import re
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


df_fake = pd.read_csv("Fake.csv")
df_true = pd.read_csv("True.csv")


df_fake["class"] = 0
df_true["class"] = 1


df_merge = pd.concat([df_fake, df_true], axis=0)
df_merge = df_merge.sample(frac=1, random_state=42).reset_index(drop=True)


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text


if "text" not in df_merge.columns:
    raise Exception("❌ 'text' column missing in dataset!")

df_merge["text"] = df_merge["text"].apply(clean_text)


X = df_merge["text"]
y = df_merge["class"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


model = LogisticRegression()
model.fit(X_train_vec, y_train)


y_pred = model.predict(X_test_vec)


print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))


def predict_news(news):
    news = clean_text(news)
    vec = vectorizer.transform([news])
    prediction = model.predict(vec)

    return "REAL NEWS ✅" if prediction[0] == 1 else "FAKE NEWS ❌"


print("\n🧪 Test Example:")
print(predict_news("Breaking news: government announces free laptops for students"))