import seaborn as sns

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import warnings




warnings.filterwarnings("ignore")




diamonds = sns.load_dataset("diamonds")

diamonds.head()
diamonds.describe()

#After you are done with exploration, the first step in any project is framing the 
#machine learning problem and extracting the feature and target arrays based on the dataset.

#In this tutorial, we will first try to predict diamond prices using their physical measurements, so our target will be the price column.

#So, we are isolating the features into X and the target into y:


from sklearn.model_selection import train_test_split



# Extract feature and target arrays

X, y = diamonds.drop('price', axis=1), diamonds[['price']]

# Extract text features

cats = X.select_dtypes(exclude=np.number).columns.tolist()

cats

# Extract text features

cats = X.select_dtypes(exclude=np.number).columns.tolist()




# Convert to Pandas category

for col in cats:

   X[col] = X[col].astype('category')

X.dtypes

# Split the data

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

import xgboost as xgb




# Create regression matrices

dtrain_reg = xgb.DMatrix(X_train, y_train, enable_categorical=True)

dtest_reg = xgb.DMatrix(X_test, y_test, enable_categorical=True)

#The class accepts both the training features and the labels. To enable automatic 
#encoding of Pandas category columns, we also set enable_categorical to True.

#A note on the difference between a loss function and 
#a performance metric: A loss function is used by machine learning 
#models to minimize the differences between the actual (ground truth) values 
#and model predictions. On the other hand, a metric (or metrics) is 
#chosen by the machine learning engineer to measure the similarity between 
#ground truth and model predictions.

#In short, a loss function should be minimized while a metric 
#should be maximized. A loss function is used during training to guide 
#the model on where to improve. A metric is used during evaluation to measure 
#overall performance.

# Define hyperparameters

params = {"objective": "reg:squarederror", "tree_method": "gpu_hist"}

n = 100

model = xgb.train(

   params=params,

   dtrain=dtrain_reg,

   num_boost_round=n,

)

#During the boosting rounds, the model object has 
# learned all the patterns of the training set it 
# possibly can. Now, we must measure its performance
#  by testing it on unseen data. That's where our 
# dtest_reg DMatrix comes into play:

from sklearn.metrics import mean_squared_error

preds = model.predict(dtest_reg)

#This step of the process is called model 
# evaluation (or inference). Once you generate 
# predictions with predict, you pass them inside
#  mean_squared_error function of Sklearn to compare 
# against y_test:

rmse = mean_squared_error(y_test, preds, squared=False)
print(f"RMSE of the base model: {rmse:.3f}")

#we create a list of two tuples that each 
# contain two elements. The first element 
# is the array for the model to evaluate, 
# and the second is the array’s name.

evals = [(dtest_reg, "validation"), (dtrain_reg, "train")]

#When we pass this array to the evals 
# parameter of xgb.train, we will see the 
# model performance after each boosting round:

n = 10000
model = xgb.train(

   params=params,

   dtrain=dtrain_reg,

   num_boost_round=n,

   evals=evals,

   verbose_eval = 500, 
   #every ten rounds

   early_stopping_rounds = 50
)

#In real-world projects, you usually train for 
# thousands of boosting rounds, which means that 
# many rows of output. To reduce them, you can use 
# the verbose_eval parameter, which forces XGBoost 
# to print performance updates every vebose_eval rounds:

#We will use a technique called early stopping. 
# Early stopping forces XGBoost to watch the
#  validation loss, and if it stops improving 
# for a specified number of rounds, it automatically stops training.