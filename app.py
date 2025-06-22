from flask import Flask, request, render_template_string
import numpy as np
import pickle
import os

app = Flask(__name__)

# Load only the model
BASE_DIR = os.path.dirname(__file__)
model_path = os.path.join(BASE_DIR, "rf_acc_68.pkl")

try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
except FileNotFoundError as e:
    print(f"❌ Model file not found: {e}")
    model = None

# HTML Templates
welcome_page = '''
<!DOCTYPE html>
<html>
<head>
    <title>Liver Cirrhosis Prediction</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            background: linear-gradient(to right, #83a4d4, #b6fbff);
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100vh;
        }
        .btn-start {
            font-size: 1.2rem;
            padding: 12px 30px;
        }
    </style>
</head>
<body>
    <h1 class="mb-4">Welcome to Liver Cirrhosis Prediction App</h1>
    <a href="/predict" class="btn btn-primary btn-start">Start Prediction</a>
</body>
</html>
'''

form_page = '''
<!DOCTYPE html>
<html>
<head>
    <title>Liver Cirrhosis Predictor</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            background: #f0f4f7;
            padding: 40px;
        }
        .container {
            background: #fff;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 0 15px rgba(0,0,0,0.1);
        }
        h2 {
            color: #2c3e50;
        }
    </style>
</head>
<body>
    <div class="container col-md-6 offset-md-3">
        <h2 class="mb-4">Liver Cirrhosis Prediction Form</h2>
        <form method="post">
            {% for feature in features %}
                <div class="mb-3">
                    <label class="form-label">{{ feature }}</label>
                    <input type="number" name="input{{ loop.index0 }}" step="any" class="form-control" required>
                </div>
            {% endfor %}
            <button type="submit" class="btn btn-success">Predict</button>
        </form>
        {% if prediction %}
            <div class="alert mt-4 {{ 'alert-danger' if prediction == 'Yes' else 'alert-success' }}">
                <strong>Prediction:</strong> Patient likely has Liver Cirrhosis: {{ prediction }}
            </div>
        {% endif %}
    </div>
</body>
</html>
'''

# Feature names (adjust if needed)
features = [
    "Age", "Gender (0=Female, 1=Male)", "Eosinophils", "Basophils",
    "Platelet Count (in lakhs)", "Total Bilirubin", "AST", "ASG",
    "Albumin", "A/G Ratio"
]

@app.route('/')
def welcome():
    return render_template_string(welcome_page)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        try:
            inputs = [float(request.form[f'input{i}']) for i in range(len(features))]
            if model is None:
                raise Exception("Model not loaded.")
            result = model.predict([inputs])[0]
            prediction = "Yes" if result == 1 else "No"
        except Exception as e:
            prediction = f"Error: {str(e)}"
    return render_template_string(form_page, features=features, prediction=prediction)

if __name__ == '__main__':
    print("📢 Server running at: http://127.0.0.1:5000")
    app.run(debug=True)
