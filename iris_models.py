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
