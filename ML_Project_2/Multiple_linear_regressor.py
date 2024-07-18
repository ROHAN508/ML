import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split



class myMultipleLinearRegressor():
    def __init__(self):
        self.coefficients = None
        self.intersept = None
    
    def fit(self, X, Y):
       X= np.insert(X,0,1,axis=1)
       self.betas = np.linalg.inv(np.dot(X.T,X)).dot(X.T).dot(Y)
       self.intersept = self.betas[0]
       self.coefficients=self.betas[1:]

        

    def predict(self, X):
        return (np.dot(X,self.coefficients) + self.intersept)


X,y = load_diabetes(return_X_y=True)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)

mlr = myMultipleLinearRegressor()
mlr.fit(X_train, y_train)
my_y_pred = mlr.predict(X_test)

MLR = LinearRegression()
MLR.fit(X_train, y_train)
y_pred = MLR.predict(X_test)

print(f'coefficients of my Model is: {mlr.coefficients}')
print(f'coefficients of sklearn Model is: {MLR.coef_}')
print(f'Intersept of my Model is: {mlr.intersept}')
print(f'Intersept of sklearn Model is: {MLR.intercept_}')
print(f'Rmse of my Model is: {np.sqrt(mean_squared_error(y_test, my_y_pred))}')
print(f'Rmse of sklearn Model is: {np.sqrt(mean_squared_error(y_test, y_pred))}')
print(f'R2 score of my Model is: {r2_score(y_test, my_y_pred)}')
print(f'R2 score of sklearn Model is: {r2_score(y_test, y_pred)}')

