import pickle
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == "POST":
        weight = [float(request.form['height'])]
        y_array = np.asarray(weight)
        final_features = y_array.reshape(-1, 1)
        my_prediction = model.predict(final_features)


    return render_template('result.html', prediction_text='Predicted Weight: {}'.format(my_prediction))


if __name__ == '__main__':
    app.run(debug=True)
