from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import pickle
from src.data import getInputHeaders

rootDir = 'D:/SARP/SARP-Aerosol-ML-BrC/Data/'
dataDir = rootDir + 'Processed/'
savePath = rootDir + 'Network/'

inputHeaders = getInputHeaders()
x = pd.read_csv(dataDir+'input')
x = x[inputHeaders]
y = pd.read_csv(dataDir+'output')


