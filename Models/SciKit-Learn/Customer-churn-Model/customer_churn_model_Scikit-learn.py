from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
#Joblib = Job + Library, meaning a library for handling computational jobs efficiently. Although today it is widely known for saving and loading machine learning models.



import pandas as pd
data = pd.read_csv('customer_churn.csv')
print(data.head())
print(data.info())

data = data.dropna()

data = pd.get_dummies(data, drop_first=True)


X = data.drop('Churn', axis=1)
y = data['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f'Test accuracy: {accuracy}')

# Simplify model by limiting its maximum depth
pruned_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, max_features='sqrt') 

pruned_model.fit(X_train, y_train) 
pruned_predictions = pruned_model.predict(X_test) 
pruned_accuracy = accuracy_score(y_test, pruned_predictions) 
print(f'Pruned Test accuracy: {pruned_accuracy}')

joblib.dump(model, 'churn_model.pkl')