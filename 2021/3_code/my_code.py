# -*- coding: utf-8 -*-

import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Lasso

def z_score(df):
    # copy the dataframe
    df_std = df.copy()
    # apply the z-score method
    for column in df_std.columns:
        df_std[column] = (df_std[column] - df_std[column].mean()) / df_std[column].std()
        
    return df_std


df1 = pd.read_table("salary_vs_age_1.csv", sep=";") 

columns_titles = ["Salary","Age"]
df2=df1.reindex(columns=columns_titles)

df2['Salary'] = df2['Salary']/1000 

df2['Age2'] = df2['Age']**2
df2['Age3'] = df2['Age']**3
df2['Age4'] = df2['Age']**4
df2['Age5'] = df2['Age']**5

# call the z_score function
df2_standard = z_score(df2)
df2_standard['Salary'] = df2['Salary']

y = df2_standard['Salary']
X = df2_standard.drop('Salary',axis=1)

lr = LinearRegression()
lr.fit(X, y)

y_pred = lr.predict(X)

# The coefficients
print('Coefficients: \n', lr.coef_)
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(y, y_pred))


rr = Ridge(alpha=0.1, normalize=True) 
# higher the alpha value, more restriction on the coefficients; low alpha > more generalization,
# in this case linear and ridge regression resembles
rr.fit(X, y)
y_pred_r = rr.predict(X)

# The coefficients
print('Coefficients: \n', rr.coef_)
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(y, y_pred_r))

lsr = Lasso(alpha=.02, normalize=True, max_iter=1000000) 
# higher the alpha value, more restriction on the coefficients; low alpha > more generalization,
# in this case linear and ridge regression resembles
lsr.fit(X, y)

y_pred_lsr = lsr.predict(X)

# The coefficients
print('Coefficients: \n', lsr.coef_)
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(y, y_pred_lsr))

