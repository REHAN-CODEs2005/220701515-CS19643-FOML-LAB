from flask import Flask, render_template, request, redirect, url_for, flash, send_file, jsonify
import pandas as pd
import joblib
import requests
import os
from werkzeug.utils import secure_filename
from io import BytesIO
import csv
from flask import render_template

app = Flask(__name__)
app.secret_key = 'superstructure'

# Load trained model
model = joblib.load('health_model.pkl')

# Global memory
health_report = {}
conversation_history = []

# API Key (DeepSeek)
DEEPSEEK_API_KEY = ''  # ✅ Replace with your actual DeepSeek API key

# Allowed file types
ALLOWED_EXTENSIONS = {'csv'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def generate_report(dataframe):
    report = {}
    for column in dataframe.columns:
        if dataframe[column].dtype in ['int64', 'float64']:
            report[column] = round(dataframe[column].mean(), 2)
    return report


@app.route('/dashboard')
def dashboard():
    report = {}

    # Open the CSV file and read its content
    with open('health_data.csv', 'r') as file:
        reader = csv.DictReader(file)  # This reads each row as a dictionary

        for row in reader:
            print(row)  # Debug: Check the row structure
            # Now we use the correct column names from your CSV
            report['Steps'] = row['Steps']
            report['Sleep Hours'] = row['SleepHours']
            report['BMI'] = row['BMI']
            report['Weight'] = row['Weight']
            report['Height'] = row['Height']
            report['Glucose'] = row['Glucose']
            report['Heart Rate'] = row['HeartRate']

    # Pass the report dictionary to the template
    return render_template('dashboard.html', report=report)
def generate_health_tips(report_data):
    """Generate personalized health tips using DeepSeek API"""
    if not DEEPSEEK_API_KEY:
        return "❗ API Key not configured properly."

    prompt = (
        "Generate 5 concise, practical health improvement tips based on these metrics:\n"
        f"{report_data}\n"
        "Format as numbered short bullet points (max 15 words each)."
    )

    try:
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={
                'Authorization': f'Bearer {DEEPSEEK_API_KEY}',
                'Content-Type': 'application/json'
            },
            json={
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "max_tokens": 300
            },
            timeout=15
        )

        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            return f"⚠️ Failed to generate tips: {response.text}"

    except Exception as e:
        return f"⚠️ Tips generation failed: {str(e)}"


@app.route('/', methods=['GET', 'POST'])
def index():
    global health_report
    health_report = {}
    conversation_history.clear()
    tips = None

    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part!', 'danger')
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            flash('No selected file!', 'danger')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join('uploads', filename)
            os.makedirs('uploads', exist_ok=True)
            file.save(filepath)

            try:
                df = pd.read_csv(filepath)
                health_report = generate_report(df)

                # Generate prediction
                input_features = [
                    health_report.get('Steps', 0),
                    health_report.get('SleepHours', 0),
                    health_report.get('BMI', 0),
                    health_report.get('Weight', 0),
                    health_report.get('Height', 0),
                    health_report.get('Glucose', 0),
                    health_report.get('HeartRate', 0)
                ]
                prediction = model.predict([input_features])
                health_report['Health Status'] = prediction[0]

                # Generate tips
                tips = generate_health_tips(health_report)

                os.remove(filepath)
                flash('Health data analyzed successfully!', 'success')
                return render_template('index.html', report=health_report, tips=tips)

            except Exception as e:
                flash(f'Error processing file: {str(e)}', 'danger')
                return redirect(request.url)

        else:
            flash('Only CSV files are allowed!', 'warning')
            return redirect(request.url)

    return render_template('index.html', report=None, tips=None)


@app.route('/chat', methods=['POST'])
def chat():
    global health_report, conversation_history

    if not DEEPSEEK_API_KEY:
        return {'reply': '❗ API Key not set properly.'}

    user_message = request.form['message']
    if not health_report:
        return {'reply': 'Please upload your health data first!'}

    conversation_history.append({"role": "user", "content": user_message})

    system_instruction = (
        "You are a professional health coach assistant. "
        "Only answer based on the user's uploaded health report: "
        f"{health_report}. "
        "Avoid discussing general health advice outside the provided data."
    )

    messages = [{"role": "system", "content": system_instruction}] + conversation_history

    try:
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={
                'Authorization': f'Bearer {DEEPSEEK_API_KEY}',
                'Content-Type': 'application/json'
            },
            json={
                "model": "deepseek-chat",
                "messages": messages,
                "temperature": 0.3,
                "max_tokens": 300
            }
        )

        if response.status_code == 200:
            reply = response.json()['choices'][0]['message']['content']
            conversation_history.append({"role": "assistant", "content": reply})

            with open('chat_log.txt', 'a', encoding='utf-8') as log_file:
                log_file.write(f"User: {user_message}\nBot: {reply}\n\n")

            return {'reply': reply}
        else:
            error_msg = response.json().get('error', {}).get('message', 'Unknown error')
            return {'reply': f"DeepSeek API Error: {error_msg}"}

    except Exception as e:
        return {'reply': f"Request Failed: {str(e)}"}


@app.route('/chat_history')
def chat_history():
    return render_template('chat_history.html', history=conversation_history)


@app.route('/clear_chat', methods=['POST'])
def clear_chat():
    global conversation_history
    conversation_history.clear()
    with open('chat_log.txt', 'w', encoding='utf-8') as log_file:
        log_file.write('')
    flash('Chat history cleared successfully!', 'success')
    return redirect(url_for('chat_history'))


@app.route('/download_report')
def download_report():
    global health_report
    if not health_report:
        flash('No health report available to download.', 'warning')
        return redirect(url_for('index'))

    df = pd.DataFrame([health_report])
    buffer = BytesIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)

    return send_file(buffer, as_attachment=True, download_name='health_report.csv', mimetype='text/csv')


@app.route('/download_chat_log')
def download_chat_log():
    if not os.path.exists('chat_log.txt'):
        flash('No chat log available to download.', 'warning')
        return redirect(url_for('chat_history'))
    return send_file('chat_log.txt', as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
