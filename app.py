from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Cargar modelos
modelos = {
    "logistic": joblib.load("./models/model_logistic.h5"),
    "randomforest": joblib.load("./models/model_forest.h5"),
    "svm": joblib.load("./models/model_svm.h5"),
    "tree": joblib.load("./models/model_tree.h5")
}

# Diccionario de clases
clases_iris = {
    0: "setosa",
    1: "versicolor",
    2: "virginica"
}

# Función para hacer predicción
def hacer_prediccion(model, features):
    predict = model.predict([features])[0]
    return int(predict), clases_iris[int(predict)]

# Función para leer parámetros
def obtener_features(request):
    data = request.get_json() or request.args
    try:
        sepal_length = float(data['sepal_length'])
        sepal_width = float(data['sepal_width'])
        petal_length = float(data['petal_length'])
        petal_width = float(data['petal_width'])
    except (KeyError, ValueError):
        return None
    return [sepal_length, sepal_width, petal_length, petal_width]

# Rutas
@app.route('/', methods=['GET'])
def home():
    return """
    <h1>API de Predicción de la Flor de Iris</h1>
    <p>Utiliza los endpoints para predecir la clase de una flor de iris basándote en sus características.</p>
    """

@app.route('/predict/logistic', methods=['POST'])
def predict_logistic():
    features = obtener_features(request)
    if features is None:
        return jsonify({'error': 'Parámetros incorrectos'}), 400
    prediction, label = hacer_prediccion(modelos['logistic'], features)
    return jsonify({'modelo': 'Logistic Regression', 'prediccion': prediction, 'clase': label})

@app.route('/predict/randomforest', methods=['POST'])
def predict_randomforest():
    features = obtener_features(request)
    if features is None:
        return jsonify({'error': 'Parámetros incorrectos'}), 400
    prediction, label = hacer_prediccion(modelos['randomforest'], features)
    return jsonify({'modelo': 'Random Forest', 'prediccion': prediction, 'clase': label})

@app.route('/predict/svm', methods=['POST'])
def predict_svm():
    features = obtener_features(request)
    if features is None:
        return jsonify({'error': 'Parámetros incorrectos'}), 400
    prediction, label = hacer_prediccion(modelos['svm'], features)
    return jsonify({'modelo': 'SVM', 'prediccion': prediction, 'clase': label})

@app.route('/predict/tree_decision', methods=['POST'])
def predict_tree():
    features = obtener_features(request)
    if features is None:
        return jsonify({'error': 'Parámetros incorrectos'}), 400
    prediction, label = hacer_prediccion(modelos['tree'], features)
    return jsonify({'modelo': 'Decision Tree', 'prediccion': prediction, 'clase': label})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
