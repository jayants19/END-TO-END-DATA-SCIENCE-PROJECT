from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model and columns
model = pickle.load(open('house_price_model.pkl', 'rb'))
columns = pickle.load(open('columns.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html',columns=columns)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        sqft = float(request.form['sqft'])
        bath = int(request.form['bath'])
        bhk = int(request.form['bhk'])
        location = request.form['location']

        # Create input array with correct column order
        input_data = np.zeros(len(columns))
        input_data[0] = sqft
        input_data[1] = bath
        input_data[2] = bhk

        if location in columns:
            loc_index = list(columns).index(location)
            input_data[loc_index] = 1

        prediction = model.predict([input_data])[0]
        output = round(prediction, 2)

        return render_template('index.html', prediction_text=f"Predicted Price: ₹ {output} Lakhs",columns=columns)

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
