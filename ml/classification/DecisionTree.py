import math
import numpy as np
import statistics as st
from collections import defaultdict

class DecisionTree:
    def __init__(self):
        self.__tree = None
        
    def fit(self, X, y):
        self.__tree = self.___make_split(X, y)

        # print('tree', self.__tree)
        
    def predict(self, X):
      res = []
      for row in X:
        # print('row', row)
        res.append(self.__predict_recursive(row, self.__tree))
        

      return np.array(res)
      

    def __predict_recursive(self, X_row, tree):
      #print('tree', tree)
      
      if dict != type(tree):
        return tree
      elif X_row[tree['col']] <= tree['threshold']:
        return self.__predict_recursive(X_row, tree['less_tree'])
      else:
        return self.__predict_recursive(X_row, tree['greater_tree'])
        
        

    def ___make_split(self, X, y):
      col, threshold, entropy = self.__get_best_split(X, y)

      less_X = []
      less_y = []
      greater_X = []
      greater_y = []

      #make split by threshold
      for row_idx in range(X.shape[0]):
        if X[row_idx][col] <= threshold:
          less_X.append(X[row_idx])
          less_y.append(y[row_idx])
        else:
          greater_X.append(X[row_idx])
          greater_y.append(y[row_idx])

      less_X = np.array(less_X)
      less_y = np.array(less_y)
      greater_X = np.array(greater_X)
      greater_y = np.array(greater_y)
      

      #no split done -> end this tree with the mode of all data
      if 0 == len(less_y):
        mode = st.mode_multi(greater_y)
        return mode[0]
      elif 0 == len(greater_y):
        mode = st.mode_multi(less_y)
        return mode[0]

      #split done->create new branches
      return {
        'col': col,
        'threshold': threshold,
        'less_tree': self.___make_split(less_X, less_y),
        'greater_tree': self.___make_split(greater_X, greater_y)
      }



      print(col, threshold, entropy)
      
     

    def __get_best_split(self, X, y):
        min_entropy = math.inf
        min_entropy_col = None
        min_entropy_threshold = None

        for col_idx in range(X.shape[1]):
            threshold, entropy = self.__get_split_by_entropy(X[:, col_idx], y)
            if(entropy < min_entropy):
                min_entropy = entropy
                min_entropy_col = col_idx
                min_entropy_threshold = threshold

        return (min_entropy_col, min_entropy_threshold, min_entropy)


    def __get_split_by_entropy(self, x, y):
        len_x = len(x)
        sorted_x_y = [(x, y) for x, y in sorted(zip(x, y), key = lambda pair: pair[0])]

        #initialze with no split
        #a split has to make it better
        min_entropy = self.__get_entropy(y)
        min_entropy_split = len_x

        for i in range(1, len_x):
          #just split if the x-values around the split are different
          #a split does not make sense if they are equal
          #ensures also that categorical data are just split in transitions between two categories
          if(sorted_x_y[i - 1][0] == sorted_x_y[i][0]):
            continue

          first_half = [y for _, y in sorted_x_y[0 : i]]
          second_half = [y for _, y in sorted_x_y[i : len_x]]
          entropy = (self.__get_entropy(first_half) + self.__get_entropy(second_half)) / 2.0
      
          if(entropy < min_entropy):
              min_entropy = entropy
              min_entropy_split = i


        #if no split happended, return results for the whole set
        if min_entropy_split == len_x:
          return (sorted_x_y[len_x - 1][0], min_entropy)

        #split happened, return mean of adjacent values and entropy
        split_threshold = (sorted_x_y[min_entropy_split][0] + sorted_x_y[min_entropy_split - 1][0]) / 2
        
        return (split_threshold, min_entropy)


    def __get_entropy(self, arr):
        arr_len = len(arr)
        counts = defaultdict(int)
        for item in arr:
            counts[item] +=1

        entropy = 0
        for item, count in counts.items():
            p = count / arr_len
            entropy -= p * math.log2(p)

        return entropy
        




