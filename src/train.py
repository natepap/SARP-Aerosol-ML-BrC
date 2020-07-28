from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle

rootDir = 'D:/SARP/SARP-Aerosol-ML-BrC/Data/'
procPath = rootDir + 'Processed/'
netPath = rootDir + 'Network/'

x = pd.read_csv(procPath+'input')
y = pd.read_csv(procPath+'output')

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.67, random_state=4)

model_2 = RandomForestRegressor(verbose=1)

model_2.fit(x_train, y_train)

with open(netPath + 'rf_v3', 'wb') as file:
    pickle.dump(model_2, file)
