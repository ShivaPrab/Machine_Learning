# Encoding = UTF- 8 
"""
Attempting to find correlative X variables for Sale Price in our Ames, Iowa 
Home Data set. We will then use these correlative variables as inputs for our 
Machine Learning Algorithm. 
"""

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor 

import pandas as pd
home_data_path = 'data/train.csv'
home_train_data = pd.read_csv(home_data_path)

home_train_data.head()

y = home_train_data.SalePrice

features_model = ['OverallQual', 'LotArea', '1stFlrSF', 'FullBath',
                 'BedroomAbvGr', 'GarageArea', 'PoolArea']

X = home_train_data[features_model]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

model_test = DecisionTreeRegressor(random_state=1)

# fit Model 
model_test.fit(train_X, train_y)

# Predictions 
val_predictions = model_test.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)

print("Validation MAE when not specifying Max Leaf Nodes: {:,.0f}".format
    (val_mae)) 

# Specifying 'Best Value' for Leaf Nodes: 

model_test = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)

model_test.fit(train_X, train_y)
val_predictions = model_test.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE for best value of Max Leaf Nodes: {:,.0f}".format(val_mae))

# RF Model, set random state to 1: 
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)
rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)
print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))

header_output = ["id", "rf_val_predictions"]
home_train_data.to_csv('submission.csv', columns = header_output)