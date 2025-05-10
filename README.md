# Entorno Virtual - Local - Dataset Iris (Machine Learning)

Esta API permitirá realizar predicciones sobre la especie de una flor ingresando sus cuatro características principales: largo y ancho del sépalo, y largo y ancho del pétalo. Construiremos una API que utiliza modelos de Machine Learning entrenados sobre el conjunto de datos Iris Dataset, el cual contiene características de diferentes tipos de flores de iris. 

# Requerimientos:

1.0.- Entorno virtual de python(terminal)
2.0.- Software Postman

# Ejecutar entorno virtual (python)
Ej. Terminal IDE - Conda

1.0.- Crear entorno virtual

![1](https://github.com/user-attachments/assets/11204727-c1d7-4030-981e-19cc9ae353c6)

1.0.- Ejecutar entorno virtual

![image](https://github.com/user-attachments/assets/5d1d02f4-987d-4135-99f3-783b236f96a8)

# Dentro de Windows / Linux

1.0.- Crear una carpeta llamada machine_learning_api

Dentro de la carpeta machine_learning_api:
1.1.- Crear una carpeta llamada models

![image](https://github.com/user-attachments/assets/27fd9e76-7509-49b3-adb8-2876ce9c3993)

1.2.- Crear un archivo llamado requirements.txt y que contenga lo siguiente:

-flask
-scikit-learn
-joblib

Las librerías anteriores mencionadas dentro del archivo requirements.txt se instalaran y ejecutaran en conjunto al estar empaquetadas en dicho archivo de texto.

1.3.- Crear un archivo llamado iris_models.py y que contenga lo siguiente:

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib

# Cargar dataset
iris = load_iris()
X = iris.data
y = iris.target

# Separar en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Modelo Logistic Regression
logistic_model = LogisticRegression(max_iter=200)
logistic_model.fit(X_train, y_train)
accuracy_logistic = logistic_model.score(X_test, y_test)
print(f"Accuracy Logistic Regression: {accuracy_logistic:.2f}")
joblib.dump(logistic_model, './models/model_logistic.h5')

# 2. Modelo SVM
svm_model = SVC()
svm_model.fit(X_train, y_train)
accuracy_svm = svm_model.score(X_test, y_test)
print(f"Accuracy SVM: {accuracy_svm:.2f}")
joblib.dump(svm_model, './models/model_svm.h5')

# 3. Modelo Decision Tree
tree_model = DecisionTreeClassifier()
tree_model.fit(X_train, y_train)
accuracy_tree = tree_model.score(X_test, y_test)
print(f"Accuracy Decision Tree: {accuracy_tree:.2f}")
joblib.dump(tree_model, './models/model_tree.h5')

# 4. Modelo Random Forest
forest_model = RandomForestClassifier()
forest_model.fit(X_train, y_train)
accuracy_forest = forest_model.score(X_test, y_test)
print(f"Accuracy Random Forest: {accuracy_forest:.2f}")
joblib.dump(forest_model, './models/model_forest.h5')

1.4.- Crear un archivo llamado app.py y que contenga lo siguiente:

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


# Dentro del entorno virtual

1.0.- Ingresar a la ruta donde se encuentra la nuestros archivos que creamos

![image](https://github.com/user-attachments/assets/70f58aa1-7645-4dff-b793-b6cb4e83f515)

1.1.- instalar las blibliotecas contenidas en el archivo de requirements.txt

![image](https://github.com/user-attachments/assets/21d01f29-1a42-4fa4-ba69-4c067ac16d80)

1.2.- Ejecutar los archivos iris_models.py y app.py

![image](https://github.com/user-attachments/assets/138c1396-059d-4546-bf05-abcdc1005738)

1.3.- Al ejecutar iris_models.py - se crearán lo modelos de machine learning dentro de la carpeta models

![image](https://github.com/user-attachments/assets/10427fcf-b77a-4805-8f86-315db19b79eb)

![image](https://github.com/user-attachments/assets/a9930394-64c3-4797-ab7b-361116d19ba5)

1.3.- Al ejecutar app.py - se levantará el servidor necesario para realizar nuestras consultas

![image](https://github.com/user-attachments/assets/be6d68ec-6408-4370-8dd0-3841b9283b6d)

1.4.- Ingresar a la siguiente ruta http://127.0.0.1:5001/ para comprobar el funcionamiento del servidor

![image](https://github.com/user-attachments/assets/b331a0e1-3ae3-4e08-80c0-5aa78a841f0e)


