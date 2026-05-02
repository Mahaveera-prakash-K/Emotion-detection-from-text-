import joblib
from preprocess import preprocess

# Corrected relative paths
model = joblib.load("models/emotion_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

def predict_emotion(text):
    cleaned = preprocess(text)
    vec = vectorizer.transform([cleaned])
    return model.predict(vec)[0]

if __name__ == "__main__":
    example = "I am feeling very sad and alone."
    predicted = predict_emotion(example)
    print(f"\n📢 Input: {example}")
    print(f"💬 Predicted Emotion: {predicted}")
