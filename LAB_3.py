# Linear Regression

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

data = pd.read_csv('USA_Housing.csv')

data.rename(columns={'Avg. Area Income': 'Income', 'Avg. Area House Age': 'Age', 'Avg. Area Number of Rooms': 'Rooms',
                     'Avg. Area Number of Bedrooms': 'Bedrooms', 'Area Population': 'Population'}, inplace=True)
data.head()

data['Price'] = data['Price'].astype('int64')

data.info()
data.drop(['Address'], axis=1, inplace=True)

# Splitting of data

x = data.iloc[:, :5]
x.head()

y = data.iloc[:, 5:]
y.head()

# Predict output Value

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

model = LinearRegression()
model.fit(X_train, y_train)

# R^2 Value by Train Data
predictions = model.predict(X_test)
r_squared = r2_score(y_test, predictions)
print("R squared Value for Test Data :")
print(r_squared)


# R^2 Value by Test Data
predictions = model.predict(X_train)
r_squared = r2_score(y_train, predictions)
print("R squared Value for Train Data :")
print(r_squared)
