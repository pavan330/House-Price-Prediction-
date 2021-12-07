import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data=pd.read_csv("data.csv")

df=data.apply(lambda x:x.fillna(x.mean())
                       if x.dtype=='float' else
                       x.fillna(x.value_counts().index[0]))

from sklearn import preprocessing
from sklearn import utils

# Features
X = df.drop('price', axis =1).values
# Target
y= df.price.values

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.25, random_state=42)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(random_state=0)
rf_reg.fit(X_train, y_train)
print("Training accuracy: ", rf_reg.score(X_train, y_train))
print("Testing accuracy: ", rf_reg.score(X_test, y_test))

from xgboost import XGBRegressor
xgb_reg = XGBRegressor()
xgb_reg.fit(X_train, y_train)
print("Training accuracy: ", xgb_reg.score(X_train, y_train))
print("Testing accuracy: ", xgb_reg.score(X_test, y_test))

import pickle
filename = 'model.pkl'
pickle.dump(xgb_reg, open(filename, 'wb'))
