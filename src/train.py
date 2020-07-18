from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
import os
import pickle

rootDir = 'D:/SARP/SARP-Aerosol-ML-BrC/Data/'
dataDir = rootDir + 'Cleaned/'
savePath = rootDir + 'Network/'
dataList = os.listdir(dataDir)

dataSet = pd.read_csv(dataDir+dataList[0])

inputHeaders = []
outputHeaders = []

for header in dataSet:
    if "WEBER" in header:
        outputHeaders.append(header)
    else:
        if "YANG" not in header and "Unnamed" not in header:
            inputHeaders.append(header)

naList = []
for inpuT in inputHeaders:
    if dataSet[inpuT].isna().all():
        naList.append(inpuT)
    elif dataSet[inpuT].isna().any():
       dataSet[inpuT] = dataSet[inpuT].fillna(dataSet[inpuT].median())

dataSet.drop(columns=naList)


for inpuT in naList:
    inputHeaders.remove(inpuT)


for output in outputHeaders:
    dataSet[output] = dataSet[output].fillna(dataSet[output].median())


x = dataSet[inputHeaders].values
y = dataSet[outputHeaders[0]].values

lab_enc = preprocessing.LabelEncoder()
encoded = lab_enc.fit_transform(y)


model = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)

score = cross_val_score(model, x, encoded, cv=5)

print(f'Cross val score = {score.mean()}')

with open(savePath + "network", "wb") as file:
    pickle.dump(model, file)
