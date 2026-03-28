import pickle

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
    elif "excellent" in text or "good" in text:
        return "Positive Feedback", "Keep up the good work"
    else:
        return "General", "Improve engagement"
print("Choose input method:")
print("1. Type your own feedback")
print("2. Select from options")
choice = input("Enter choice: ")

#options
options = {
    "1": "Teacher is too fast",
    "2": "Class is confusing",
    "3": "Teaching is excellent",
    "4": "Subject is difficult"
}
#feedback
if choice == "2":
    print("\nOptions:")
    for k, v in options.items():
        print(f"{k}. {v}")
    opt = input("Select option: ")
    feedback = options.get(opt, "Average class")
else:
    feedback = input("Enter your feedback: ")

vec = vectorizer.transform([feedback])
prediction = model.predict(vec)[0]
prob = max(model.predict_proba(vec)[0])

issue, suggestion = detect_issue(feedback)

print("\n--- RESULT ---")
print("Feedback:", feedback)
print("Sentiment:", prediction)
print("Confidence:", round(prob, 2))
print("Issue:", issue)
print("Suggestion:", suggestion)
