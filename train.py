import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

print("🔥 Training started...")

# Load dataset
data = pd.read_csv("data/feedback.csv", encoding="utf-8")

print("Dataset loaded:")
print(data.head())

# Split data
X = data['text']
y = data['sentiment']

# Convert text to numerical features
vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)

# Train model
model = MultinomialNB()
model.fit(X_vec, y)

# Save model and vectorizer
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ Model trained and saved successfully!")