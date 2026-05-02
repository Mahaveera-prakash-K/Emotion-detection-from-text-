import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from preprocess import preprocess


# Load dataset
df =pd.read_csv("data/emotion_model.pkl")
df['clean_text'] = df['text'].apply(preprocess)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['clean_text']).toarray()
y = df['emotion']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save model and vectorizer
joblib.dump(model, 'models/emotion_model.pkl')
joblib.dump(tfidf, 'models/tfidf_vectorizer.pkl')

# Evaluate
y_pred = model.predict(X_test)
print("✅ Model trained successfully!\n")
print("📊 Classification Report:")
print(classification_report(y_test, y_pred))
# Training script will go here
