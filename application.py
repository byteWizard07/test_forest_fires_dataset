import pickle
from flask import Flask, request, render_template
import numpy as np
import pandas as pd

application = Flask(__name__)
app = application

# Load model and scaler
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == "POST":
        try:
            # Get input values
            Temperature = float(request.form.get('Temperature'))
            RH = float(request.form.get('RH'))
            Ws = float(request.form.get('Ws'))
            Rain = float(request.form.get('Rain'))
            FFMC = float(request.form.get('FFMC'))
            DMC = float(request.form.get('DMC'))
            ISI = float(request.form.get('ISI'))
            Classes = float(request.form.get('Classes'))
            Region = float(request.form.get('Region'))

            # Create DataFrame (fixes sklearn warning)
            input_data = pd.DataFrame(
                [[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]],
                columns=['Temperature','RH','Ws','Rain','FFMC','DMC','ISI','Classes','Region']
            )

            # Scale and predict
            new_data_scaled = standard_scaler.transform(input_data)
            prediction = ridge_model.predict(new_data_scaled)

            result = round(prediction[0], 2)

            return render_template('home.html', result=result)

        except Exception as e:
            return render_template('home.html', result=f"Error: {str(e)}")

    else:
        return render_template('home.html', result=None)

if __name__ == "__main__":
    app.run(host="0.0.0.0")