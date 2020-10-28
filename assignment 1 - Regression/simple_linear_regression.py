#%%
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np


#%%
# Create pandas dataframe using the data from given csv
df = pd.read_csv('Salaries-Simple_Linear.csv')
print(df.keys())

#%%
# Split data into test and train data
X_train, X_test, y_train, y_test = train_test_split(df['Years_of_Expertise'].to_numpy().reshape(-1,1) ,df['Salary'].to_numpy().reshape(-1,1) , test_size=0.25, random_state=1)

# %%
# create regression model
regressor = linear_model.LinearRegression()

# fit regression model
regressor.fit(X_train, y_train)

# predict y values off of the x test data
y_pred = regressor.predict(X_test)

# Output regressor coefficients
print('Coefficients: \n', regressor.coef_)

# Output regressor mean absolute error
print('Mean absolute error: %.4f' % mean_absolute_error(y_test, y_pred))

# Output regressor coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.4f' % r2_score(y_test, y_pred))

# %%
# Make Predictions and output
predict_years = np.array([7.3]).reshape(-1,1)
sal_pred = regressor.predict(predict_years)
print("Years: ", predict_years, "Predicted Salary: ", sal_pred)

# %%
