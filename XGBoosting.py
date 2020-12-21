import pandas as pd

from sklearn import model_selection

import xgboost

import sklearn

from xgboost.sklearn import XGBClassifier

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import accuracy_score

data = pd.read_csv('Food_Inspections.csv', header=0, low_memory =False)

print(data)

data = data.drop(['Inspection_ID', 'AKA_Name', 'Facility_Type', 'Address', 'Zip', 'Inspection_Date'], axis= 1)

data.info(verbose= True)

dataset = data.values

print(dataset)

x = dataset[:,0:6]

y = dataset[:,6]

print(x)

print(y)

x = x.astype(str)

y = y.astype(str)

label_encoder = LabelEncoder()

x[: , 2] = label_encoder.fit_transform(x[:, 2])

x[: , 3] = label_encoder.fit_transform(x[:, 3])

x[: , 4] = label_encoder.fit_transform(x[:, 4])

x[: ,5] = label_encoder.fit_transform(x[:, 5])

onehotencoder = OneHotEncoder(categories = 'auto')

x= onehotencoder.fit_transform(x)

label_encoder = label_encoder.fit(y)

label_encoded_y = label_encoder.transform(y)

seed = 7

test_size = 0.33

X_train, X_test,y_train, y_test = model_selection.train_test_split(x, label_encoded_y, test_size = test_size, random_state = seed)

classifier = xgboost.XGBClassifier()

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)

pd.DataFrame(y_pred).to_csv('y_pred.csv')

pd.DataFrame(cm).to_csv('confusion_matrix.csv')

prediction = [round(value) for value in y_pred]

accuracy = accuracy_score(y_test, prediction)

print(accuracy*100)

