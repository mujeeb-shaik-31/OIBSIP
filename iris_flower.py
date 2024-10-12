import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

iris = pd.read_csv("Iris.csv")
print(iris.head())
print(iris.describe())
print("Target Labels:", iris["Species"].unique())

fig = px.scatter(iris, x="SepalWidthCm", y="SepalLengthCm", color="Species")
fig.show()

# Separate features and target labels
x = iris.drop("Species", axis=1)
y = iris["Species"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:\n", classification_report(y_test, y_pred))
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

x_new = np.array([[5, 2.9, 1.4, 0.2, 2]])
prediction = knn.predict(x_new)
print(f"Prediction: {prediction}")