import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score



class myLinearRegressor():
    def __init__(self):
        self.m = None
        self.b = None
    
    def fit(self, X, Y):
        self.num=0
        self.den=0
        for i in range(X.shape[0]):
          self.num = self.num + (X[i] - X.mean())*(Y[i] - Y.mean())
          self.den = self.den + (X[i] - X.mean())*(X[i] - X.mean())

        self.m = self.num/self.den
        self.b = Y.mean() -(self.m*X.mean())

    def predict(self, X):
        return self.m*X +self.b

def train_test_split(data, test_ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    # print(shuffled)
    test_size = int(len(data)*test_ratio)
    test_indices = shuffled[-test_size:]
    train_indices = shuffled[:-test_size]

    return data.iloc[train_indices], data.iloc[test_indices]

df = pd.read_csv("placement.csv")
# print(df.head())
train_data, test_data = train_test_split(df, 0.3)
Train_x = train_data.iloc[: , 0].values
Train_y = train_data.iloc[: , 1].values
Test_x = test_data.iloc[: , 0].values
Test_y = test_data.iloc[: , 1].values

lr = myLinearRegressor()
lr.fit(Train_x, Train_y)
my_Y_pred=lr.predict(Test_x)

LR = LinearRegression()
LR.fit(pd.DataFrame(Train_x), pd.DataFrame(Train_y))
Y_pred= LR.predict(pd.DataFrame(Test_x))

print(f'Slope of my Model is: {lr.m}')
print(f'Slope of sklearn Model is: {LR.coef_}')
print(f'Intersept of my Model is: {lr.b}')
print(f'Intersept of sklearn Model is: {LR.intercept_}')
print(f'Rmse of my Model is: {np.sqrt(mean_squared_error(Test_y, my_Y_pred))}')
print(f'Rmse of sklearn Model is: {np.sqrt(mean_squared_error(Test_y, Y_pred))}')
print(f'R2 score of my Model is: {r2_score(Test_y, my_Y_pred)}')
print(f'R2 score of sklearn Model is: {r2_score(Test_y, Y_pred)}')



