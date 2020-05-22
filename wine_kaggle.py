Issues - need Seaborn (proxy didnt fix issue) and other ML packages
#loads the csv and headers!


#SEABORN NO MODULE NAMED
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

#Importing required packages.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing

import matplotlib.pyplot as plt  # To visualize
from scipy import stats
%matplotlib inline

#headers
wine = pd.read_csv('Downloads/winequality-red.csv')
wine.head()


#give rows with max value of quality 
wine[wine['quality']==wine['quality'].max()]

#convert dependednt variable (quality) as the index/first column
wine = pd.read_csv('Downloads/winequality-red.csv',index_col=[11])

#CREATE CHART 
wine.plot(figsize=(18,5))

#always check for missing data  --> should be false, raw data often will be true and thus needa  cleaning!
wine.isnull().values.any()

#remove outliers - essential when running regression 
wine.hist()

#remove outliers beyond 3 std dev 
std_dev = 3
wine = wine[(np.abs(stats.zscore(wine)) < float(std_dev)).all(axis=1)]
wine.plot(figsize=(18,5))

# 1 independent variable along with dependent (quality) **figure this ooout
pd.DataFrame(wine['fixed acidity'])

#scatter plot of two variables (regression)
X = wine.iloc[:, 0].values.reshape(-1, 1)  # values converts it into a numpy array
Y = wine.iloc[:, 1].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
linear_regressor = LinearRegression()  # create object for the class
linear_regressor.fit(X, Y)  # perform linear regression
Y_pred = linear_regressor.predict(X)  # make predictions
plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()

#Regression 
model = LinearRegression()


HOW TO SHOW AXES? HOW TO MAKE Y QUALITY ? 
x=wine.density

y=wine.quality
plt.scatter(x,y)
from scipy.stats import linregress

stats = linregress(x, y)
plt.plot(x, m * x + b)
#reshape data
X = wine['density'].values.reshape(-1,1)
y = wine['quality'].values.reshape(-1,1)

regressor = LinearRegression()
regressor.fit(X, y)
print(regressor.intercept_)
print(regressor.coef_)

HOW TO SHOW R VALUE? SHOE CORRELATION/HOW TO SHOW STRONGEST CORRELATION?

#regression on quality component/ strongest variable / column 