import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from ml.classification import DecisionTree

dataset = pd.read_csv('./datasets/titanic_data_adjusted.csv')
dataset.fillna(0, inplace = True)
survived = dataset['Survived'].values
data = dataset.drop(['Survived'], axis = 1).values
X_train, y_train, X_test, y_test = train_test_split(data, survived)





# X = np.array([
#     [0, 0],
#     [0, 0],
#     [1, 0],
#     [1, 0],
#     [0, 1]
# ])

# y = np.array([1, 0, 1, 0, 1])

tree = DecisionTree()
tree.fit(X_train, y_train)