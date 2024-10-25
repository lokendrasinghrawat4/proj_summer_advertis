from flask import Flask, render_template, request
import pickle
import numpy as np

# Create a Flask app
app = Flask(__name__)

# Load the pre-trained model
with open('advertising_sales_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle form submission and return prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user inputs from form
        tv_input = float(request.form['tv'])
        radio_input = float(request.form['radio'])
        newspaper_input = float(request.form['newspaper'])

        # Prepare the data for prediction
        input_data = np.array([[tv_input, radio_input, newspaper_input]])

        # Predict sales using the loaded model
        predicted_sales = model.predict(input_data)[0]

        # Return the prediction back to the frontend
        return render_template('index.html', prediction_text=f'Predicted Sales: {predicted_sales:.2f}')
    
if __name__ == '__main__':
    app.run(debug=True)
