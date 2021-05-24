import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

#Reading in Data 

iowa_train_path = 'data/train.csv'
home_train_data = pd.read_csv(iowa_train_path)

# Target Object y = Sale Price; i.e. what we are attempting to predict 
y = home_train_data.SalePrice

# Creating X: 
features_example = ['LotArea', 'YearBuilt', '1stFlrSF', 'FullBath',
                    'BedroomAbvGr', 'TotRmsAbvGrd']

X = home_train_data[features_example]

#split "Train" Model into validation and train data:
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

#Specify Model 
iowa_model_example = DecisionTreeRegressor(random_state=1)
#Fit Model
iowa_model_example.fit(train_X, train_y)

#Predictions / Calculations on MAE (Mean Absolute Error)
val_predictions = iowa_model_example.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE when not specifying Max Leaf Nodes: {:,.0f}".format
    (val_mae))

# Using 'best' value for Max_Leaf_NOdes 

iowa_model_example = DecisionTreeRegressor(max_leaf_nodes=100,
                                           random_state=1)

iowa_model_example.fit(train_X, train_y)
val_predictions = iowa_model_example.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE for best value of Max Leaf Nodes: {:,.0f}".format
    (val_mae))

# Define the model, set random state to 1: 
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)
rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)
print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))