import math
from collections import defaultdict
class DecisionTree:
    def __init__(self):
        self.__tree = []
        
    def fit(self, X, y):
        self.__tree = []
        self.___make_split(X, y)
        

    def ___make_split(self, X, y):
        col, threshold, entropy = self.__get_best_split(X, y) 
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

        #initialze withno split
        #a split has to make it better
        min_entropy = self.__get_entropy(y)
        min_entropy_split = len_x

        for i in range(1, len_x):
            first_half = [y for _, y in sorted_x_y[0 : i]]
            second_half = [y for _, y in sorted_x_y[i : len_x]]
            entropy = (self.__get_entropy(first_half) + self.__get_entropy(second_half)) / 2.0
        
            if(entropy < min_entropy):
                min_entropy = entropy
                min_entropy_split = i


        split_threshold = (sorted_x_y[min_entropy_split][0] + sorted_x_y[min_entropy_split - 1][0]) / 2
        
        return (split_threshold, entropy)


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
        




