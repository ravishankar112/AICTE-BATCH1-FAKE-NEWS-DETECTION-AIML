import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Load data
df_fake = pd.read_csv('Fake.csv')
df_true = pd.read_csv('True.csv')

# Add class labels
df_fake["class"] = 0
df_true["class"] = 1

# Merge datasets
df = pd.concat([df_fake, df_true], axis=0)

# Keep only text and class columns
df = df[['text', 'class']]

# Random shuffle
df = df.sample(frac=1).reset_index(drop=True)

# Text preprocessing function
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

# Apply preprocessing
df["text"] = df["text"].apply(wordopt)

# Split features and labels
x = df["text"]
y = df["class"]

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

# Vectorize text
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

# Model 1: Logistic Regression
print("=" * 50)
print("MODEL 1: LOGISTIC REGRESSION")
print("=" * 50)
LR = LogisticRegression()
LR.fit(xv_train, y_train)
pred_lr = LR.predict(xv_test)
print(f"Accuracy: {LR.score(xv_test, y_test)}")
print(classification_report(y_test, pred_lr))

# Model 2: Linear Regression
print("\n" + "=" * 50)
print("MODEL 2: LINEAR REGRESSION")
print("=" * 50)
lin_reg = LinearRegression()
lin_reg.fit(xv_train, y_train)
lin_reg_pred = lin_reg.predict(xv_test)
print(f"Accuracy: {lin_reg.score(xv_test, y_test)}")

# Model 3: Decision Tree
print("\n" + "=" * 50)
print("MODEL 3: DECISION TREE")
print("=" * 50)
DT = DecisionTreeClassifier()
DT.fit(xv_train, y_train)
pred_dt = DT.predict(xv_test)
print(f"Accuracy: {DT.score(xv_test, y_test)}")
print(classification_report(y_test, pred_dt))

# Model 4: Random Forest
print("\n" + "=" * 50)
print("MODEL 4: RANDOM FOREST")
print("=" * 50)
RFC = RandomForestClassifier(random_state=0)
RFC.fit(xv_train, y_train)
pred_rfc = RFC.predict(xv_test)
print(f"Accuracy: {RFC.score(xv_test, y_test)}")
print(classification_report(y_test, pred_rfc))

# Function for manual testing
def output_label(n):
    return "Fake News" if n == 0 else "Not A Fake News"

def manual_testing(news):
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_xv_test = vectorization.transform(new_def_test["text"])
    
    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    pred_RFC = RFC.predict(new_xv_test)
    
    print("\n" + "=" * 50)
    print("PREDICTION RESULTS")
    print("=" * 50)
    print(f"LR Prediction: {output_label(pred_LR[0])}")
    print(f"DT Prediction: {output_label(pred_DT[0])}")
    print(f"RFC Prediction: {output_label(pred_RFC[0])}")
    print("=" * 50)

# Test with sample news
if __name__ == "__main__":
    test_news = "Your news text here"
    manual_testing(test_news)