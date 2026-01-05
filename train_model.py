import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib

# Download stopwords (first time only)
nltk.download('stopwords')

# Load dataset
data = pd.read_csv("resume_data.csv")
data.columns = data.columns.str.strip().str.lower()

# Use only required columns
data = data[['resume_str', 'category']]
data.dropna(inplace=True)

# Clean text function
def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', str(text))
    text = text.lower()
    words = text.split()
    words = [w for w in words if w not in stopwords.words('english')]
    return " ".join(words)

# Apply cleaning
data['cleaned_resume'] = data['resume_str'].apply(clean_text)

# Features and labels
X = data['cleaned_resume']
y = data['category']

# Convert text to numbers
vectorizer = TfidfVectorizer(max_features=3000)
X_vec = vectorizer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Test accuracy
predictions = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, predictions))

# Save model
joblib.dump(model, "resume_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")


