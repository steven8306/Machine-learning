import pandas as pd
from sklearn.tree import DecisionTreeRegressor
iowa_file_path = './melb_data.csv'
home_data=pd.read_csv(iowa_file_path)
home_data = home_data.dropna(axis=0)
y=home_data.Price
features=['Rooms','Bathroom','Car','Regionname','Longtitude']
X=home_data[features]

home_model=DecisionTreeRegressor(random_state=1)
# home_model.fit(X, y)
print(X.describe())

# print("result of model")
# print(home_model.predict(X.head()))