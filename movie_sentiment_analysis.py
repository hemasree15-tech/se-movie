########################################################################
# MOVIE REVIEW SENTIMENT ANALYSIS PROJECT
# College Mini Project - Machine Learning
# Simple Python Script (Beginner Level)
########################################################################

# =======================================================
# SECTION 1: IMPORT LIBRARIES
# =======================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report

from collections import Counter


# =======================================================
# SECTION 2: DATA LOADING
# =======================================================
# Dataset: "IMDB Dataset of 50K Movie Reviews" (Kaggle)
# File has 2 columns: "review" (the text) and "sentiment" (positive / negative).
# We rename "review" -> "text" and write sentiment as "Positive" / "Negative"
# so the rest of the code can stay the same.
#
# SAMPLE_SIZE: the full dataset has 50,000 reviews. SVM and Random Forest
# become very slow on all 50,000, so by default we use a balanced random
# sample (half Positive, half Negative). Set SAMPLE_SIZE = None to use
# the complete dataset (expect a long run time, especially for SVM).
SAMPLE_SIZE = 10000

df = pd.read_csv("IMDB Dataset.csv")
df = df.rename(columns={"review": "text"})
df['sentiment'] = df['sentiment'].str.strip().str.capitalize()   # positive -> Positive

print("Full dataset shape:", df.shape)

# remove duplicates first so they do not appear in both train and test sets
df = df.dropna().drop_duplicates()

if SAMPLE_SIZE is not None and SAMPLE_SIZE < len(df):
    df = df.groupby('sentiment', group_keys=False).sample(n=SAMPLE_SIZE // 2, random_state=42)
    df = df.reset_index(drop=True)
    print("Using a balanced sample of", SAMPLE_SIZE, "reviews for faster training.")

print("First 5 rows:")
print(df.head())

print("\nDataset shape:", df.shape)
print("\nColumn names:", df.columns.tolist())
print("\nSentiment counts:\n", df['sentiment'].value_counts())


# =======================================================
# SECTION 3: HANDLING MISSING VALUES
# =======================================================
print("\nMissing values before cleaning:\n", df.isnull().sum())

df = df.dropna()               # remove empty rows
df = df.drop_duplicates()      # remove duplicate rows

print("\nDataset shape after removing missing/duplicate rows:", df.shape)


# =======================================================
# SECTION 4: EXPLORATORY DATA ANALYSIS (EDA)
# =======================================================

# Sentiment distribution plot
plt.figure(figsize=(6, 4))
sns.countplot(x='sentiment', data=df, palette='Set2')
plt.title("Movie Review Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.savefig("1_sentiment_distribution.png")
plt.show()

# Review length column
df['text_length'] = df['text'].apply(len)

plt.figure(figsize=(6, 4))
sns.histplot(df['text_length'], bins=30, kde=True)
plt.title("Movie Review Length Distribution")
plt.xlabel("Length of Review")
plt.savefig("2_text_length_distribution.png")
plt.show()


# =======================================================
# SECTION 5: TEXT PREPROCESSING
# =======================================================
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()                                  # lowercasing
    text = re.sub(r'http\S+|www\S+', '', text)            # remove links
    text = re.sub(r'<.*?>', '', text)                     # remove HTML tags (common in IMDB reviews)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)  # remove punctuation
    text = re.sub(r'\d+', '', text)                       # remove numbers
    tokens = text.split()                                  # tokenization
    tokens = [word for word in tokens if word not in stop_words]  # remove stopwords
    tokens = [lemmatizer.lemmatize(word) for word in tokens]      # lemmatization
    return " ".join(tokens)

df['clean_text'] = df['text'].apply(clean_text)

print("\nSample before and after cleaning:")
print(df[['text', 'clean_text']].head())


# =======================================================
# SECTION 6: WORDCLOUD VISUALIZATION
# =======================================================
positive_text = " ".join(df[df['sentiment'] == 'Positive']['clean_text'])
negative_text = " ".join(df[df['sentiment'] == 'Negative']['clean_text'])

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

wc1 = WordCloud(width=600, height=400, background_color='white').generate(positive_text)
axes[0].imshow(wc1)
axes[0].axis('off')
axes[0].set_title("Positive Review Words")

wc2 = WordCloud(width=600, height=400, background_color='white').generate(negative_text)
axes[1].imshow(wc2)
axes[1].axis('off')
axes[1].set_title("Negative Review Words")

plt.savefig("3_wordclouds.png")
plt.show()


# =======================================================
# SECTION 7: MOST COMMON WORDS (POSITIVE / NEGATIVE)
# =======================================================
def get_top_words(text, n=10):
    words = text.split()
    common = Counter(words).most_common(n)
    return common

print("\nTop 10 Positive Words:", get_top_words(positive_text))
print("\nTop 10 Negative Words:", get_top_words(negative_text))


# =======================================================
# SECTION 8: FEATURE EXTRACTION (TF-IDF)
# =======================================================
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['clean_text'])   # sparse matrix (saves memory)
y = df['sentiment']

print("\nShape of feature matrix:", X.shape)


# =======================================================
# SECTION 9: TRAIN-TEST SPLIT
# =======================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# =======================================================
# SECTION 10: MODEL TRAINING (5 MODELS)
# =======================================================
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

naive_bayes = MultinomialNB()
naive_bayes.fit(X_train, y_train)

svm_model = SVC(probability=True)
svm_model.fit(X_train, y_train)

random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest.fit(X_train, y_train)

xgb_model = XGBClassifier(eval_metric='logloss')
# XGBoost needs numeric labels (binary: Negative=0, Positive=1)
y_train_num = y_train.map({'Negative': 0, 'Positive': 1})
y_test_num = y_test.map({'Negative': 0, 'Positive': 1})
xgb_model.fit(X_train, y_train_num)


# =======================================================
# SECTION 11: MODEL EVALUATION
# =======================================================
models = {
    "Logistic Regression": (log_reg, y_test),
    "Naive Bayes": (naive_bayes, y_test),
    "SVM": (svm_model, y_test),
    "Random Forest": (random_forest, y_test),
    "XGBoost": (xgb_model, y_test_num)
}

results = []

for name, (model, ytest_actual) in models.items():
    y_pred = model.predict(X_test)

    acc = accuracy_score(ytest_actual, y_pred)
    prec = precision_score(ytest_actual, y_pred, average='weighted')
    rec = recall_score(ytest_actual, y_pred, average='weighted')
    f1 = f1_score(ytest_actual, y_pred, average='weighted')

    results.append([name, acc, prec, rec, f1])

    print(f"\n----------------- {name} -----------------")
    print("Accuracy :", round(acc, 4))
    print("Precision:", round(prec, 4))
    print("Recall   :", round(rec, 4))
    print("F1 Score :", round(f1, 4))
    print("\nConfusion Matrix:\n", confusion_matrix(ytest_actual, y_pred))
    print("\nClassification Report:\n", classification_report(ytest_actual, y_pred))


# =======================================================
# SECTION 12: MODEL COMPARISON GRAPH
# =======================================================
results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score"])
print("\nFinal Comparison Table:\n", results_df)

plt.figure(figsize=(9, 5))
sns.barplot(x="Model", y="Accuracy", data=results_df, palette="viridis")
plt.title("Model Accuracy Comparison - Movie Sentiment Analysis")
plt.xticks(rotation=20)
plt.ylim(0, 1)
plt.savefig("4_model_comparison.png")
plt.show()


# =======================================================
# SECTION 13: CONFUSION MATRIX HEATMAP (BEST MODEL)
# =======================================================
best_model_name = results_df.sort_values("Accuracy", ascending=False).iloc[0]['Model']
print("\nBest Model:", best_model_name)

best_model, best_ytest = models[best_model_name]
y_pred_best = best_model.predict(X_test)

cm = confusion_matrix(best_ytest, y_pred_best)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f"Confusion Matrix - {best_model_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("5_confusion_matrix_best_model.png")
plt.show()


# =======================================================
# SECTION 14: ERROR ANALYSIS (WRONG PREDICTIONS)
# =======================================================
num_wrong = int((y_pred_best != np.array(best_ytest)).sum())
print("\nNumber of wrong predictions:", num_wrong)
print("Total test samples:", X_test.shape[0])
print("Error rate:", round(num_wrong / X_test.shape[0] * 100, 2), "%")


# =======================================================
# SECTION 15: PREDICTION SYSTEM (USER INPUT)
# =======================================================
def predict_sentiment(text):
    cleaned = clean_text(text)
    vector = tfidf.transform([cleaned])

    prediction = log_reg.predict(vector)[0]
    probability = log_reg.predict_proba(vector)[0]

    classes = log_reg.classes_
    prob_dict = dict(zip(classes, probability))

    print("\nReview:", text)
    print("Predicted Sentiment:", prediction)
    print("Probability Scores:")
    for cls, prob in prob_dict.items():
        print(f"   {cls}: {round(prob*100, 2)}%")

    return prediction


# Example predictions
predict_sentiment("This movie was an absolute masterpiece, the acting and direction were phenomenal!")
predict_sentiment("Terrible film, the plot made no sense and the acting was wooden.")
predict_sentiment("The cinematography was stunning but the story dragged on too long.")

# NOTE: the model is trained on full-length IMDB reviews (many sentences),
# so it is more reliable on longer reviews than on one-line comments.
predict_sentiment("I went in with low expectations, but this film completely won me over. "
                  "The story is moving, the lead performances feel genuine, and the final "
                  "act had me close to tears. One of the best movies I have seen this year, "
                  "and I would happily watch it again.")
predict_sentiment("What a waste of two hours. The script is full of cliches, the dialogue is "
                  "painful to listen to, and the so-called twist is obvious from the first "
                  "ten minutes. I regret watching this and would not recommend it to anyone.")


print("\n===================================================")
print("PROJECT COMPLETED SUCCESSFULLY")
print("===================================================")
