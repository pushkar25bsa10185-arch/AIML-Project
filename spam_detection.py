import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Sample dataset of SMS labeled spam or ham
data = {
    'text': [
        "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005.",
        "U dun say so early hor... U c already then say...",
        "WINNER!! As a valued network customer you have been selected to receive a $900 prize reward!",
        "Had your mobile 11 months or more? You are entitled to update to the latest colour mobiles with camera for free!",
        "I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k?",
        "SIX chances to win CASH! From 100 to 20,000 pounds txt> CSH11 and send to 87575.",
        "Hey there! Are you free tonight to catch a movie?",
        "URGENT! You have won a 1 week FREE membership in our $100,000 Prize Jackpot!"
    ],
    'label': ['spam', 'ham', 'spam', 'spam', 'ham', 'spam', 'ham', 'spam']
}

df = pd.DataFrame(data)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.25, random_state=42)

# Vectorize text
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# Train classifier
clf = MultinomialNB()
clf.fit(X_train_counts, y_train)

# Save model and vectorizer
joblib.dump(clf, 'spam_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("Model and vectorizer saved.")

# Optional: Evaluate on test set
y_pred = clf.predict(X_test_counts)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(
"Classification Report:\n", classification_report(y_test, y_pred)
)

# Check if saved files exist and load model for new email prediction
if os.path.exists('spam_model.pkl') and os.path.exists('vectorizer.pkl'):
    clf_loaded = joblib.load('spam_model.pkl')
    vectorizer_loaded = joblib.load('vectorizer.pkl')

    # New email to check
    new_email = ["Congratulations! You won a free ticket to Bahamas! Claim now."]
    new_email_counts = vectorizer_loaded.transform(new_email)

    prediction = clf_loaded.predict(new_email_counts)
    print("The new email is classified as:", prediction[0])
else:
    print("Model files not found. Please train and save the model first.")