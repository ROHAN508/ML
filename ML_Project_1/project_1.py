import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import linear_model,metrics,tree,ensemble
from sklearn.preprocessing import PolynomialFeatures
from joblib import dump,load


def split_data(data, test_ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    # print(shuffled)
    test_size = int(len(data)*test_ratio)
    test_indices = shuffled[-test_size:]
    train_indices = shuffled[:-test_size]

    return data.iloc[train_indices], data.iloc[test_indices]

df = pd.read_csv('data.csv')
df.info()
df['RM']=df['RM'].fillna(df['RM'].median())
df.info()

# sns.scatterplot(data=df, x=df['RM'], y=df['MEDV'])
# plt.show()
# print(df.columns)

# checking coreletion
corr_matrix = df.corr()
print(corr_matrix['MEDV'].sort_values())

train_set, test_set = split_data(df, 0.3)

train_x = train_set.drop('MEDV', axis=1)
train_y = train_set['MEDV']
test_x = test_set.drop('MEDV', axis=1)
test_y = test_set['MEDV']


# MODEL 1 - LINEAR REGRESSION

model1 = linear_model.LinearRegression()
model1.fit(train_x,train_y)
pred_val1= model1.predict(test_x)

MAE1=metrics.mean_absolute_error(test_y,pred_val1)
print(MAE1)
RMSE1=np.sqrt(metrics.mean_squared_error(test_y,pred_val1))
print(RMSE1)
# MAE1 = 3.3630247954702592
# RMSE1 = 4.7171283147666685

# MODEL 2 - POLYNOMIAL REGRESSION

RSME_TRAIN = []
RSME_TEST = []
for d in range(2,6):
    poly_converter_test = PolynomialFeatures(degree=d, include_bias=False)
    polytest_train_x=poly_converter_test.fit_transform(train_x)
    polytest_test_x=poly_converter_test.fit_transform(test_x)

    model_test = linear_model.LinearRegression()
    model_test.fit(polytest_train_x,train_y)
    pred_val_test= model_test.predict(polytest_test_x)
    pred_val_train= model_test.predict(polytest_train_x)

    train_rsme = np.sqrt(metrics.mean_squared_error(train_y, pred_val_train))
    test_rsme = np.sqrt(metrics.mean_squared_error(test_y, pred_val_test))
    RSME_TEST.append(test_rsme)
    RSME_TRAIN.append(train_rsme)

    # print(f"{d}")
    # print(train_rsme)
    # print(test_rsme)
    # print('\n')

print(RSME_TRAIN)
print(RSME_TEST)

# rsme train = [2.406391327612731, 6.140289625901951e-07, 5.159458583601162e-10, 1.6111453986024074e-10]
# rsme test = [3.411159631190667, 282.66506617474954, 149.10115289054644, 137.7581584093889]
# As we can see the model is working best for second degree


poly_converter = PolynomialFeatures(degree=2, include_bias=False)
poly_train_x=poly_converter.fit_transform(train_x)
poly_test_x=poly_converter.fit_transform(test_x)

model2 = linear_model.LinearRegression()
model2.fit(poly_train_x,train_y)
pred_val2= model2.predict(poly_test_x)

MAE2=metrics.mean_absolute_error(test_y,pred_val2)
print(MAE2)
RMSE2=np.sqrt(metrics.mean_squared_error(test_y,pred_val2))
print(RMSE2)

#MAE2 = 2.509582569926805
#RSME2 = 3.411159631190667

# MODEL 3

model3 = tree.DecisionTreeRegressor()
model3.fit(train_x,train_y)
pred_val3= model3.predict(test_x)

MAE3=metrics.mean_absolute_error(test_y,pred_val3)
print(MAE3)
RMSE3=np.sqrt(metrics.mean_squared_error(test_y,pred_val3))
print(RMSE3)

#MAE3 = 2.959602649006623
#RSME3 = 4.196395488909232

# MODEL 4

model4 = ensemble.RandomForestRegressor()
model4.fit(train_x,train_y)
pred_val4= model4.predict(test_x)

MAE4=metrics.mean_absolute_error(test_y,pred_val4)
print(MAE4)
RMSE4=np.sqrt(metrics.mean_squared_error(test_y,pred_val4))
print(RMSE4)

#MAE4 = 2.1627417218543052
#RSME4 = 2.9578984664366597

# As we can see Model 4 the randomforestregressor model works best for this task

# Dumping final model

final_model = ensemble.RandomForestRegressor()
final_model.fit(df.drop('MEDV', axis=1),df['MEDV'])
dump(final_model, 'Final_Model.joblib')
loaded_model = load('Final_Model.joblib')
pred_val= loaded_model.predict(df.drop('MEDV', axis=1))

MAE=metrics.mean_absolute_error(df['MEDV'],pred_val)
print(MAE)
RMSE=np.sqrt(metrics.mean_squared_error(df['MEDV'],pred_val))
print(RMSE)

# Final MAE = 0.8007351778656119
# Final RSME = 1.1927246997887506
