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
featurePath = rootDir + 'Features/'

x = pd.read_csv(procPath+'input')
y = pd.read_csv(procPath+'output')

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.67, random_state=4)

with open(netPath+'rf_v6', 'rb') as file:
    rf = pickle.load(file)

pred = rf.predict(x_test)
train_score = rf.score(x_train, y_train)
test_score = rf.score(x_test, y_test)
overall_score = rf.score(x, y)
mse = mean_squared_error(y_test, pred)
result = permutation_importance(rf, x, y, n_jobs=-1)
result_mean = result.importances_mean
result_std = result.importances_std

print('Train score is:', train_score, 'out of 1')
print('Test score is:', test_score, 'out of 1')
print('Overall score is:', overall_score, 'out of 1')
print('RMSE is:', sqrt(mse))

inputs = x.keys()

df = pd.DataFrame(list(zip(inputs, result_mean)))
df.to_csv(featurePath + 'feature_importance_rf_v6')

tempList = result_mean.copy()

topList = np.zeros(10, int)

for x in range(10):
    idx = np.argmax(tempList)
    topList[x] = idx
    tempList[idx] = 0

plt.figure()
plt.title('Feature Importance')
plt.bar(inputs[topList], result_mean[topList], color='r', yerr=result_std[topList], align='center')
plt.xticks(rotation=15)
plt.savefig(featurePath+'featurePlot')
plt.show()

y_graph = y_test.to_numpy()

plt.plot(pred, 'bo', label='Predicted')
plt.plot(y_graph, 'ro', label='Actual')
plt.legend(loc='upper left')
plt.savefig(featurePath+'predPlot')
plt.show()
