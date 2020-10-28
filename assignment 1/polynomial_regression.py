#%%
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np

        
#%%
# Create pandas dataframe using the data from given csv
df = pd.read_csv('Propose-Salaries-Polynomial.csv')
print(df.keys())

# %%
# select x and y data
x_data = df["Level"].to_numpy()
y_data = df["Salary"].to_numpy()

# gives vector of coeficients for the polynomial function
p_coeffs = np.polyfit(x_data, y_data, 6)

# Create polynomial function
polynomial_model = np.poly1d(p_coeffs)

# predict y values off of the x test data
y_pred = polynomial_model(x_data)

# Output regressor coefficient of determination: 1 is perfect prediction
print('Coefficient of determination:%.4f' % r2_score(y_data, y_pred))
# %%
# Predict Salary using given level
predict_level = 6.5
predicted_sal = polynomial_model(predict_level)

# Output Predictions
print("Given level: %.4f" % predict_level, ", Predicted Salary is: %.2f" % predicted_sal)
# %%
