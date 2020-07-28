import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import numpy as np


rootDir = 'D:/SARP/SARP-Aerosol-ML-BrC/Data/'
procPath = rootDir + 'Processed/'
netPath = rootDir + 'Network/'

x = pd.read_csv(procPath+'input').drop('Time_Start', axis=1)
y = pd.read_csv(procPath+'output').drop('Time_Start', axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.67, random_state=4)

with open(netPath+'rf_v2', 'rb') as file:
    rf = pickle.load(file)

pred = rf.predict(x_test)
train_score = rf.score(x_train, y_train)
test_score = rf.score(x_test, y_test)
mse = mean_squared_error(y_test, pred)
result = permutation_importance(rf, x, y)
result_mean = result.importances_mean
result_std = result.importances_std
result_raw = result.importances

print('Train score is:', train_score, 'out of 1')
print('Test score is:', test_score, 'out of 1')
print('RMSE is:', sqrt(mse))

inputs = x.keys()

tempList = result_mean.copy()

topList = np.zeros(10, int)

for x in range(10):
    idx = np.argmax(tempList)
    topList[x] = idx
    tempList[idx] = 0

plt.figure()
plt.title('Feature Importance')
plt.bar(inputs[topList], result_mean[topList], color='r', yerr=result_std[topList], align='center')
plt.xticks(rotation=90)
plt.show()

y_graph = y_test.to_numpy()

plt.plot(pred, 'bo')
plt.plot(y_graph, 'ro')

plt.show()
