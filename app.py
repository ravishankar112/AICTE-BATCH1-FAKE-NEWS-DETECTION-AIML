import pandas as pd
import numpy as np
import re
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ===============================
# 1. LOAD DATASET (NO CHANGE)
# ===============================
df_fake = pd.read_csv("Fake.csv")
df_true = pd.read_csv("True.csv")

# ===============================
# 2. LABELS (NO NAME CHANGE)
# ===============================
df_fake["class"] = 0
df_true["class"] = 1

# ===============================
# 3. MERGE DATASET
# ===============================
df_merge = pd.concat([df_fake, df_true], axis=0)

# shuffle
df_merge = df_merge.sample(frac=1, random_state=42).reset_index(drop=True)

# ===============================
# 4. CLEAN TEXT FUNCTION
# ===============================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

# ===============================
# 5. APPLY CLEANING
# ===============================
df_merge["text"] = df_merge["text"].apply(clean_text)

# ===============================
# 6. FEATURES & LABEL
# ===============================
X = df_merge["text"]
y = df_merge["class"]

# ===============================
# 7. TRAIN TEST SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# 8. TF-IDF VECTORIZATION
# ===============================
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ===============================
# 9. MODEL TRAINING
# ===============================
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# ===============================
# 10. PREDICTION
# ===============================
y_pred = model.predict(X_test_vec)

# ===============================
# 11. EVALUATION
# ===============================
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ===============================
# 12. TEST FUNCTION
# ===============================
def predict_news(news):
    news = clean_text(news)
    vec = vectorizer.transform([news])
    prediction = model.predict(vec)
    
    if prediction[0] == 0:
        return "FAKE NEWS ❌"
    else:
        return "REAL NEWS ✅"

# ===============================
# 13. SAMPLE TEST
# ===============================
print("\nTest 1:", predict_news("Breaking news: government announces free laptops for all students"))