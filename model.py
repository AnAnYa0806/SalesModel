import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
import pickle

def get_X():
    dataset = pd.read_csv('sales.csv')

    dataset['rate'].fillna(0, inplace=True)

    dataset['sales_in_first_month'].fillna(dataset['sales_in_first_month'].mean(), inplace=True)

    X = dataset.iloc[:, :3]

    def convert_to_int(word):
        word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
                    'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0: 0}
        return word_dict[word]

    X['rate'] = X['rate'].apply(lambda x : convert_to_int(x))

    return X

def get_y():
    dataset = pd.read_csv('sales.csv')

    dataset['rate'].fillna(0, inplace=True)

    dataset['sales_in_first_month'].fillna(dataset['sales_in_first_month'].mean(), inplace=True)

    X = dataset.iloc[:, :3]

    def convert_to_int(word):
        word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
                    'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0: 0}
        return word_dict[word]
    y = dataset.iloc[:, -1]

    return y

from sklearn.linear_model import LinearRegression

def linear_regressor(X = get_X(), y = get_y(), FI=True, n=False):
    regressor = LinearRegression(fit_intercept=FI, normalize=n)

    regressor.fit(X, y)

    pickle.dump(regressor, open('model.pkl','wb'))

    model = pickle.load(open('model.pkl','rb'))
    print(model.predict([[4, 300, 500]]))