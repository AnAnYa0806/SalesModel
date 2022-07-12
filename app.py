import numpy as np
from flask import Flask, request, jsonify, render_template
from model import linear_regressor
import pickle

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route("/train", methods=["GET", "POST"])
def train_model():
    if request.method == 'GET':
        return render_template('train.html')
    else:
        data = dict(request.form)
        fit_intercept = True if data["fi"] == "true" else False 
        normalize = True if data["n"] == "true" else False

        linear_regressor(FI=fit_intercept, n=normalize)

        return render_template('train.html', done_text="Trained successfully!")

@app.route('/predict',methods=['POST'])
def predict():
    model = pickle.load(open('model.pkl', 'rb'))

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Sales should be $ {}'.format(output))

@app.route('/results',methods=['POST'])
def results():
    model = pickle.load(open('model.pkl', 'rb'))
    
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)