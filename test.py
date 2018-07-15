import sys
import numpy as np
import metrics

# print(metric.accuracy_score([2, 3, 3], [1, 2, 3]))



# sys.exit()





import pandas as pd
from sklearn.model_selection import train_test_split

from ml.classification import DecisionTree

dataset = pd.read_csv('./datasets/titanic_data.csv')
data = dataset.drop(['Name', 'Ticket', 'PassengerId', 'Cabin'], axis = 1)
data.dropna(0, inplace = True)

data = pd.get_dummies(data, drop_first = True)

survived = data['Survived']
data = data.drop('Survived', axis = 1)
#print(data.head())

data = data.values
survived = survived.values

X_train, X_test, y_train, y_test = train_test_split(data, survived, random_state = 0)

tree = DecisionTree()
tree.fit(X_train, y_train)
y_train_pred = tree.predict(X_train)

print('accuracy train', metrics.accuracy_score(y_train, y_train_pred))

y_test_pred = tree.predict(X_test)
print('accuracy test', metrics.accuracy_score(y_test, y_test_pred))
