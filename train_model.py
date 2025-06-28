import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Load the CSV data
df = pd.read_csv('data/alerts.csv')
X = df['message']
y = df['label']

# Convert words to numbers using vectorizer
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# Train the model
model = LogisticRegression()
model.fit(X_vec, y)

# Save the trained model and vectorizer to files
with open('./model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('./vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("✅ Model trained and saved.")

import os

print("✅ Model training completed.")
print("Saved files:")
print(" - model.pkl: ", os.path.isfile('model.pkl'))
print(" - vectorizer.pkl: ", os.path.isfile('vectorizer.pkl'))
print("Saved in folder:", os.getcwd())

import os

save_path = os.path.join(os.getcwd(), "model.pkl")
with open(save_path, 'wb') as f:
    pickle.dump(model, f)

save_path = os.path.join(os.getcwd(), "vectorizer.pkl")
with open(save_path, 'wb') as f:
    pickle.dump(vectorizer,f)