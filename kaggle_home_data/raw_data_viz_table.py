# Encoding = UTF-8 

"""
Project for Kaggle's Machine Learning Introduction Course

Be sure to read the data description document for further information about the data used in this file. 

Expected Behavior of this program: 
    - Accurately predict the home prices in Ames, Iowa with various algorithms 
    - Further develop our model using a Random Forest Regressor 

This file: 
- Reads in "data/train.csv", saves all of the column names into a list, and return a visual correlelation table  
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv("data/train.csv")

# We are defining our target variable, "y" as the sale price of the home

y = ['SalePrice']

# Attempting to find values for X 
column_names = list(df.columns.values)

'''
Column Names: 
    ['Id', 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType', 'SaleCondition', 'SalePrice']
''' 


print(df.describe())

# Correlation Table: 

def build_corr_plot():
    """
    Builds our Correlation Plot 
    Inputs: DataFrame 
    Outputs: Framework for "plt.show()"
    """
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot
labels = [df[column_names[:]]]

plt.matshow(df.corr())
plt.show()

'''
A cursory look at our data shows that there is a good bit of "mapping" we need to do. First off, there are several columns of data that do not use numerical values, but instead use strings to quantify. We need to figure out a way to map those from a qualitative sense into a quantitative sense, and feed it into our random forest algorithm. 
'''

""" def cat_to_num(df):
    categories = unique(df)
    features = []
    for cat in categories:
        binary = (df == cat)
        features.append(binary.astype("int"))
    return features  """
