import pickle

from flask import Flask
from flask import request
from flask import jsonify
import pandas as pd

model_file = 'model_RF.bin'


with open(model_file, 'rb') as f_in_model:
    model = pickle.load(f_in_model)


app = Flask(__name__)


@app.route('/test', methods=['GET'])
def test():
    return jsonify(hello="world")


@app.route('/predict', methods=['POST'])
def predict():
    patient = request.get_json()
    patient_df = pd.DataFrame([patient])
    y_pred = model.predict(patient_df)
    prediction = bool(y_pred[0])

    result = {
        "death_event": prediction
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
