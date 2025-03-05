import pandas as pd
iowa_file_path = './melb_data.csv'
home_data=pd.read_csv(iowa_file_path)
home_data = home_data.dropna(axis=0)
y=home_data.Price
features=['Rooms','Bathroom','Landsize','Lattitude','Longtitude']
X=home_data[features]

#define model
from sklearn.tree import DecisionTreeRegressor
home_model=DecisionTreeRegressor(random_state=1)
home_model.fit(X, y)
print(home_data.groupby('Landsize').Landsize.count())
print("result of model")
#print(home_model.predict(X.head()))

#model validation
from sklearn.metrics import mean_absolute_error
predicted_home_prices=home_model.predict(X)
mean_absolute_error(y,predicted_home_prices)
#print(mean_absolute_error(y,predicted_home_prices))

#test-split validation
from sklearn.model_selection import train_test_split
train_x, val_x, train_y, val_y = train_test_split(X,y,random_state=0)
#define model
improved_model = DecisionTreeRegressor()
#fit model
improved_model.fit(train_x,train_y)
val_prediction = improved_model.predict(val_x)
#print(mean_absolute_error(val_y,val_prediction))
#randomforest model
from sklearn.ensemble import RandomForestRegressor
forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_x, train_y)
melb_preds = forest_model.predict(val_x)
print(mean_absolute_error(val_y, melb_preds))

output = pd.DataFrame({'Id': val_x.index,
                       'SalePrice': melb_preds})
output.to_csv('submission.csv', index=False)