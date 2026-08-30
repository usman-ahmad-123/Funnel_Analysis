from flask import Flask, render_template, request
import joblib
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load your trained model
model = joblib.load('my_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Example input fields from HTML form
    channel = request.form['channel']
    region = request.form['region']
    device = request.form['device']
    sessions = float(request.form['sessions'])

    # Create DataFrame for model input
    input_data = pd.DataFrame([[channel, region, device, sessions]],
                              columns=['Channel', 'Region', 'Device', 'Sessions'])

    # Predict conversion or revenue
    prediction = model.predict(input_data)[0]

    return render_template('index.html', prediction_text=f'Predicted Conversion: {prediction:.2f}%')

if __name__ == '__main__':
    app.run(debug=True)
