import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

def non_func(x):
    y = 1.6345 - 0.6235*np.cos(0.6067*x) - 1.3501*np.sin(0.6067*x) - 1.1622 * np.cos(2*x*0.6067) - 0.9443*np.sin(2*x*0.6067)
    return y

def add_noise(y):
    np.random.seed(14)
    varNoise = np.max(y) - np.min(y)
    y_noisy = y + 0.1*varNoise*np.random.normal(0,1,len(y))
    return y_noisy
 
x = np.linspace(1,10,50)
y_true = non_func(x)
y_measured = add_noise(y_true)

x = x[:, np.newaxis]
y_measured = y_measured[:, np.newaxis]
 
plt.figure(figsize=(10, 6))
plt.plot(x, y_true, 'k--', label='pozadinska fja')

stupnjevi =[2,6,15]
np.random.seed(12)
perm=np.random.permutation(len(x))
for d in stupnjevi:
    poly=PolynomialFeatures(d)
    xpoly=poly.fit_transform(x)
    
    indexi = np.random.permutation(len(xpoly))
    indexi_train = indexi[0:int(np.floor(0.7*len(xpoly)))]
    indexi_test = indexi[int(np.floor(0.7*len(xpoly)))+1:len(xpoly)]

    xtrain = xpoly[indexi_train,]
    ytrain = y_measured[indexi_train]

    xtest = xpoly[indexi_test,]
    ytest = y_measured[indexi_test]

    linearModel = lm.LinearRegression()
    linearModel.fit(xtrain,ytrain)

    ytest_p = linearModel.predict(xtest)
    mse = mean_squared_error(ytest, ytest_p)
    print(" ", mse)
    plt.plot(x, linearModel.predict(xpoly),label=f"deg={d}")
    
plt.legend()
plt.show()