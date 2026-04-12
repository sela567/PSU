import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, max_error

df = pd.read_csv('cars_processed.csv')

#izbacujemo ime, kategoricke zadrzavamo
df = df.drop(['name', 'mileage'], axis=1)
df = df.dropna()

#rpogramiranje kategorickih velicina
df = pd.get_dummies(df, columns=['fuel', 'seller_type', 'transmission', 'owner'])

X = df.drop('selling_price', axis=1)
y = df['selling_price']

#podjela traina
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=300)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_s, y_train)

y_pred_train = model.predict(X_train_s)
y_pred_test = model.predict(X_test_s)

print("train skup:")
print("r2:", r2_score(y_train, y_pred_train))
print("mse:", mean_squared_error(y_train, y_pred_train))
print("mae:", mean_absolute_error(y_train, y_pred_train))
print("max error:", max_error(y_train, y_pred_train))

print("\ntest skup:")
print("r2:", r2_score(y_test, y_pred_test))
print("mse:", mean_squared_error(y_test, y_pred_test))
print("mae:", mean_absolute_error(y_test, y_pred_test))
print("max error:", max_error(y_test, y_pred_test))

#Bolji REZULTATI nego u zadatku 5 zbog KATEGORICKIH VARIJABLI

fig = plt.figure(figsize=[13, 10])
ax = sns.regplot(x=y_pred_test, y=y_test, line_kws={'color': 'green'})
ax.set(xlabel='predikcija', ylabel='stvarna vrijednost', title='rezultati na testnom skupu - s kategorickima')
plt.show()