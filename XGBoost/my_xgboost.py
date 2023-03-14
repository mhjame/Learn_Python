# Phần 1. GBM
# Data: 

#dự đoán tình trạng vay 
import pandas as pd
import numpy as np

# Doc du lieu tu file
df = pd.read_csv(".\XGBoost\loan_data.csv")

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

is_Category = df.dtypes == object

category_column_list = df.columns[is_Category].tolist()

df[category_column_list] = df[category_column_list].apply(lambda col: le.fit_transform(col))

#chat gpt
#for col in category_column_list:
#    print(df[col].dtype)

df.dropna(inplace= True)
df.drop(columns=['Loan_ID'], inplace = True)
    
#chia dữ liệu train, test
from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size = 0.25, random_state=42)

x_train = train.drop(columns=['Loan_Status'])
y_train = train['Loan_Status']

x_test = test.drop(columns=['Loan_Status'])
y_test = test['Loan_Status']

# Train model

from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(learning_rate=0.01, random_state=42)
model.fit(x_train, y_train)

# Tinh toan acc tren test

acc = model.score(x_test, y_test)
print(acc*100)

# XgBoost

import xgboost as xgb

model_xgb = xgb.XGBClassifier(random_state=42, n_estimators = 100)
model_xgb.fit(x_train, y_train)

acc = model_xgb.score(x_test, y_test)
print(acc*100)