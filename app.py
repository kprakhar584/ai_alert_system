from flask import Flask, render_template, request
import pickle
import pandas as pd
import smtplib
from email.mime.text import MIMEText

# Load model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Load contact emails from contacts.csv
contacts = pd.read_csv('contacts.csv')
emails = contacts['email'].tolist()

# Flask setup
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['message']
    vec_input = vectorizer.transform([user_input])
    prediction = model.predict(vec_input)[0]

    alert = ''
    if prediction == 'emergency':
        alert = "🚨 EMERGENCY DETECTED"
        message = f"🚨 Emergency Alert from AI System\n\nMessage: {user_input}"
        send_bulk_email(message)

    return render_template('index.html', prediction=prediction, alert=alert)

def send_bulk_email(message):
    sender = os.gentenv("EMAIL_USER")
    password = os.gentenv("EMAIL_PASS")

    for recipient in emails:
        msg = MIMEText(message)
        msg['Subject'] = '🚨 AI Emergency Alert'
        msg['From'] = sender
        msg['To'] = recipient

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender, password)
            server.send_message(msg)
            print(f"✅ Email sent to {recipient}")

if __name__ == '__main__':
    app.run(debug=True)