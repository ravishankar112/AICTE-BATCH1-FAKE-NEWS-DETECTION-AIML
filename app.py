import pandas as pd
import numpy as np
import re
import string
import seaborn as sns
import matplotlib.pyplot as plt

# Sklearn Imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer  # Text ko numbers me badalne ke liye

# ==========================================
# 1. DATA LOADING (Same Paths & Names)
# ==========================================
print("--- Loading Datasets ---")
df_fake = pd.read_csv(r"C:\Users\hp\Downloads\archive\Fake.csv")
df_true = pd.read_csv(r"C:\Users\hp\Downloads\archive\True.csv")

# Class Target Feature Insert Karna
df_fake["class"] = 0
df_true["class"] = 1

# VS Code me shapes aur columns dekhne ke liye print use karna padega
print(f"Fake News Shape: {df_fake.shape}")
print(f"True News Shape: {df_true.shape}\n")

# ==========================================
# 2. DATA CLEANING & FIXING ISSUES
# ==========================================
print("--- Cleaning Data & Removing Duplicates/Empty Spaces ---")

# Dono dataframes ko merge (combine) karna
df = pd.concat([df_fake, df_true], axis=0).reset_index(drop=True)

# Issue 1 Fix: Jo text column me sirf spaces ("   ") hain, unhe remove karna
df['text'] = df['text'].str.strip()
df.replace("", np.nan, inplace=True)
df.dropna(subset=['text'], inplace=True)


# Issue 2 Fix: Reuters ka source tag hatana (Data Leakage se bachne ke liye)
def remove_reuters_prefix(text):
    # Yeh pattern "WASHINGTON (Reuters) - " jaisi cheezon ko remove karega
    return re.sub(r'^.*?\(\s*Reuters\s*\)\s*-\s*', '', text, flags=re.IGNORECASE)

df['text'] = df['text'].apply(remove_reuters_prefix)


# Issue 3 Fix: Punctuation, links aur numbers saaf karna
def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)                  # URLs hatana
    text = re.sub(r'<.*?>+', '', text)                                  # HTML tags hatana
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)   # Punctuation hatana
    text = re.sub(r'\n', ' ', text)                                     # Newlines ko space banana
    text = re.sub(r'\w*\d\w*', '', text)                                # Numbers wale words hatana
    return text

df['text'] = df['text'].apply(clean_text)
print(f"Final combined dataset shape: {df.shape}\n")

# ==========================================
# 3. TRAIN-TEST SPLIT
# ==========================================
print("--- Splitting Data into Train & Test sets ---")
X = df['text']
y = df['class']

# 80% Training aur 20% Testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training items: {len(X_train)}, Testing items: {len(X_test)}\n")

# ==========================================
# 4. TEXT VECTORIZATION (TF-IDF)
# ==========================================
print("--- Converting Text to Numbers (Vectorization) ---")
# Max features isliye set kiya taaki aapka VS Code crash na kare (Memory limit me rahe)
vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')

X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)
print("Vectorization complete! Ready for model training.")

# Iske niche aap apna koi bhi model (jaise RandomForestClassifier) fit kar sakte hain.