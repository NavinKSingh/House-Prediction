import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

house_price = fetch_california_housing()

print(house_price)

house_price_dataframe = pd.DataFrame(house_price.data)

house_price_dataframe.head()

house_price_dataframe['price'] = house_price.target

house_price_dataframe.head()

house_price_dataframe.shape

house_price_dataframe.isnull().sum()

house_price_dataframe.describe()

correlation = house_price_dataframe.corr()

plt.figure(figsize=(10,10))
sns.heatmap(correlation,cbar=True,square=True,fmt='.1f',annot=True,annot_kws={'size':8},cmap='Blues')

x = house_price_dataframe.drop(['price'],axis=1)
y = house_price_dataframe['price']

print(x)
print(y)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2)

print(x.shape,x_train.shape,x_test.shape)

model = XGBRegressor()

model.fit(x_train,y_train)

training_data_prediction = model.predict(x_train)

print(training_data_prediction)

score_1 = metrics.r2_score(y_train,training_data_prediction)

score_2 = metrics.mean_absolute_error(y_train,training_data_prediction)

print("R squared error: ",score_1)
print("Mean absolute error: ",score_2)

plt.scatter(y_train,training_data_prediction)
plt.xlabel("Actual prices")
plt.ylabel("Predicted prices")
plt.title("Actual price vs Predicted price")
plt.show()

test_data_prediction = model.predict(x_test)

score_1 = metrics.r2_score(y_test,test_data_prediction)
score_2 = metrics.mean_absolute_error(y_test,test_data_prediction)

print("R squared error: ",score_1)
print("Mean Absolute error: ",score_2)