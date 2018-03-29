#XGBregressor  
Implementation of XGBregressor on Kaggle house price data


import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
data = pd.read_csv('../input/train.csv')
data.dropna(axis=0, subset=['SalePrice'], inplace=True)
data1 = pd.read_csv('../input/test.csv')
train_y = data.SalePrice
train_X = data.drop(['SalePrice'], axis=1).select_dtypes(exclude=['object'])
test_X = data1.select_dtypes(exclude=['object'])
train_X, test1_X, train_y, test1_y = train_test_split(train_X.as_matrix(), train_y.as_matrix(), test_size=0.3)

my_imputer = Imputer()
train_X = my_imputer.fit_transform(train_X)
test1_X = my_imputer.transform(test1_X)
test_X = my_imputer.transform(test_X)
my_model = XGBRegressor(n_estimators = 1000,learning_rate = 0.04)
my_model.fit(train_X,train_y,early_stopping_rounds = 6,eval_set=[(test1_X, test1_y)], verbose=False)
predictions = my_model.predict(test_X)
