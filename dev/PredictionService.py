from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
import gzip
import os
HOME = os.environ['LIMA_HOME']
app = Flask(__name__)
CORS(app)
prediction_service=None

class PredictionService:
    def __init__(self) -> None:        
        with gzip.open(f'{HOME}/model/trained_models/RandomWalk.pgz', 'r') as f:
            self.model = pickle.load(f)

    def make_prediction(self,h):
        if self.model is None:
            return "No model"
        return self.model.forecast(h)

@app.route('/')
def index():
    return '42\n'

@app.route('/predict', methods=['POST'])
def predict():
    request_body = request.get_json()
    if request_body is None:
        return jsonify({'return': 'empty request_body'})    
    h=int(request_body['h'])
    result = prediction_service.make_prediction(h)
    return jsonify({'Prediction': str(result)})

def main():
    global prediction_service
    prediction_service=PredictionService()
    app.run(host='0.0.0.0', port=1234, debug=True)

if __name__ == '__main__':
    main()