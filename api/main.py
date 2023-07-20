# save this as app.py
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
import tensorflow as tf
import pickle
import numpy as np

app = Flask(__name__)
model = tf.keras.models.load_model("H:/Python/MachineLearning/1/PROJETOS/CLASSIFICACAO_SENTIMENTOS/testmodel.h5")
with open("H:/Python/MachineLearning/1/PROJETOS/CLASSIFICACAO_SENTIMENTOS/vocabulario.pkl", "rb") as f:
    vocabulario = pickle.load(f, encoding='utf-8')

vectorizer = CountVectorizer(vocabulary=vocabulario)

@app.route("/")
def hello():
    return "Hello, World!"

@app.route("/predict", methods=['POST'])
def predictia():
    data = request.get_json()
    text = data['text']
    vectorized_text = vectorizer.transform([text])
    prediction = model.predict(vectorized_text)
    result = np.argmax(prediction)
    json_answer = ""
    match result:
        case 0:
            json_answer = "O sentimento é triste"
        case 1:
            json_answer = "O sentimento é neutro"
        case 2:
            json_answer = "O sentimento é positivo"

    return jsonify({'prediction': json_answer}), 200


if __name__ == "__main__":
    app.run(host='127.0.0.1')