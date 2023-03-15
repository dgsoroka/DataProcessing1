import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler


df = pd.read_excel('cancer patient data sets (v1).xlsx')
df.isnull().sum()
df1 = df.dropna()
df1 = df1.shape

df_p = df.interpolate(method='polynomial', order = 2)
# print(df_p)



ln_model = LinearRegression()
scaler = StandardScaler()
df_s = pd.DataFrame(scaler.fit_transform(df), columns = df.columns)

test = df_s[df_s['Dust Allergy'].notnull()]
train = df_s[df_s['Dust Allergy'].notnull()]

y_train = train['Dust Allergy']

X_train = train.drop(['Dust Allergy', 'Genetic Risk'], axis=1)
X_test = test.drop(['Dust Allergy', 'Genetic Risk'], axis=1)

print(ln_model.fit(X_train, y_train))



y_pred = ln_model.predict(X_test)
print(y_pred)

df5 = pd.concat([train, test])
df5.sort_index(inplace=True)
print(df5)

df5 = pd.DataFrame(scaler.inverse_transform(df5), columns=df5.columns)
with pd.option_context("display.max_rows", None, "display.max_columns", None):
    print(df5['Dust Allergy'])

scaler = StandardScaler()
df6 = pd.DataFrame(scaler.fit_transform(df), columns = df.columns)
enable_iterative_imputer = True
mice_imputer = IterativeImputer(initial_strategy = 'mean', estimator = LinearRegression())
mice = mice_imputer.fit_transform(df6)
mice = pd.DataFrame(scaler.inverse_transform(mice), columns = df6.columns)
print(mice)






