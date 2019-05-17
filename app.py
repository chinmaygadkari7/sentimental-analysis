import json
import logging
from model import load_model, Predictor, tokenizer
from flask import Flask, request

app = Flask(__name__)

@app.route('/ping')
def ping():
    return 'pong'

@app.route('/predict', methods=['POST'])
def predict():
    text = request.data
    prediction = predictor(text.decode('utf-8'))
    out = 'positive' if prediction else 'negative'
    return json.dumps({'type': out}, indent=2)

def _get_predictor():
    model, vocab = load_model()
    predictor = Predictor(model, vocab)
    return predictor

if __name__ == "__main__":
    predictor = _get_predictor()
    app.run(port=5000)
