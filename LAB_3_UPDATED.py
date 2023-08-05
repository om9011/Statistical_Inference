import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

warnings.filterwarnings('ignore')
dataset = pd.read_csv('melb_data.csv')
dataset.head(5)


Inuse = ['Suburb', 'Rooms', 'Type', 'Method', 'SellerG', 'Regionname', 'Propertycount',
         'Distance', 'CouncilArea', 'Bedroom2', 'Bathroom', 'Car', 'Landsize', 'BuildingArea', 'Price']
dataset = dataset[Inuse]

# dataset.isna().sum()

colToFill = ['Propertycount', 'Distance', 'Bedroom2', 'Bathroom', 'Car']  # These cols have missing values
dataset[colToFill] = dataset[colToFill].fillna(0)

dataset['Landsize'] = dataset['Landsize'].fillna(dataset.Landsize.mean())
dataset['BuildingArea'] = dataset['BuildingArea'].fillna(dataset.BuildingArea.mean())

dataset.dropna(inplace=True)

dataset = pd.get_dummies(dataset, drop_first=True)  # converts categorical variables into binary
print(dataset.head())
x = dataset.drop('Price', axis=1)
y = dataset['Price']


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
r_squared = r2_score(y_test, predictions)
print("R Squared Value Using Test Data", r_squared)

predictions = model.predict(X_train)
r_squared = r2_score(y_train, predictions)
print("R Squared Value Using Train Data" ,r_squared)



from sklearn import linear_model

lasso_reg = linear_model.Lasso(alpha =50, max_iter = 100, tol = 0.1)
lasso_reg.fit(X_train,y_train)

print("R Squared Value Using Train Data (Lasso)" ,lasso_reg.score(X_train,y_train))

print("R Squared Value Using Train Data (Lasso)" ,lasso_reg.score(X_test,y_test))