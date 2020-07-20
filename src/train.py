from sklearn.ensemble import RandomForestRegressor
import pandas as pd
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



model = RandomForestRegressor(max_depth=2, random_state=0)
model.fit(x,y)

score = model.score(x,y)

print(f'Score = {score}')

with open(savePath + "network", "wb") as file:
    pickle.dump(model, file)
