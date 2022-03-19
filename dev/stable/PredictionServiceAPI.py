import datetime
import os

from flask import Flask, jsonify, request
from flask_cors import CORS

from PredictionService import PredictionService

HOME = os.environ['LIMA_HOME']
app = Flask(__name__)
CORS(app)


@app.route('/answer')
def index():
    return '42'


@app.route('/predict', methods=['POST'])
def predict():
    request_body = request.get_json()
    window_since = 5
    window_till = 0
    if 'start' in request_body:
        window_since = int(request_body['start'])
    if 'end' in request_body:
        window_till = int(request_body['end'])
    prediction_service = PredictionService()
    df_prediction = prediction_service.get_prev_prediction(
        window_start=window_since, window_end=window_till)
    df_prediction.reset_index(inplace=True)
    df_prediction["Date"] = df_prediction.Date.apply(
        lambda x: x.strftime("%Y-%m-%d"))
    df_prediction.set_index("Date", inplace=True)
    payload = {
        'now': datetime.datetime.utcnow(),
        'range': f'from: -{window_since}d to: {window_till}d',
        'Prediction': df_prediction.to_dict()
    }
    return jsonify(payload)

@app.route('/hist', methods=['POST'])
def hist_closing():
    request_body = request.get_json()
    window_since = 5
    window_till = 0
    if 'start' in request_body:
        window_since = int(request_body['start'])
    if 'end' in request_body:
        window_till = int(request_body['end'])
    prediction_service = PredictionService()
    df_hist = prediction_service.get_hist_quote(
        window_start=window_since, window_end=window_till)
    df_hist.reset_index(inplace=True)
    df_hist["Date"] = df_hist.Date.apply(
        lambda x: x.strftime("%Y-%m-%d"))
    df_hist.set_index("Date", inplace=True)
    payload = {
        'now': datetime.datetime.utcnow(),
        'range': f'from: -{window_since}d to: {window_till}d',
        'Hist Closing': df_hist.to_dict()
    }
    return jsonify(payload)


def main():
    app.run(host='0.0.0.0', port=1234, debug=True)


if __name__ == '__main__':
    main()
