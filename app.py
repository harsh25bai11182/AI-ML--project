import pickle

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def detect_issue(text):
    text = text.lower()
    if "fast" in text:
        return "Speed", "Reduce teaching pace"
    elif "difficult" in text or "confusing" in text:
        return "Difficulty", "Simplify explanations"
    elif "clear" in text:
        return "Clarity", "Maintain clarity"
    else:
        return "General", "Improve engagement"

# User input
feedback = input("Enter student feedback: ")

# Transform input
vec = vectorizer.transform([feedback])

# Predict sentiment
prediction = model.predict(vec)[0]
prob = max(model.predict_proba(vec)[0])

# Detect issue
issue, suggestion = detect_issue(feedback)

# Output
print("\n--- RESULT ---")
print("Sentiment:", prediction)
print("Confidence:", round(prob, 2))
print("Issue Detected:", issue)
print("Suggestion:", suggestion)