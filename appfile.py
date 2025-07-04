from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)
model = joblib.load('model.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None

    if request.method == 'POST':
        try:
            # Capture form inputs
            data = {
                "functional_group": request.form['functional_group'],
                "presence": int(request.form['presence']),
                "g": int(request.form['g']),
                "ng": int(request.form['ng']),
                "sampling_point": int(request.form['sampling_point']),
                "transect": request.form['transect']
            }

            # Make prediction
            df = pd.DataFrame([data])
            prediction = round(model.predict(df)[0], 2)

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)