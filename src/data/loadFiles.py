import pandas as pd
import numpy as np
import os
from src.data.dataProcess import readFile
"""
Loads raw data and processes data into pandas dataframes
Saves to Data/Cleaned
"""

rootDir = 'D:/SARP/SARP-Aerosol-ML-BrC/Data/'
rawPath = rootDir + 'Raw/'
savePath = rootDir + 'Cleaned/'

def loadAuth(filePath):
    rawList = os.listdir(filePath)
    readDict = {}
    colHeaders = []

    for data in rawList:

        if ".ict" in data:
            print("Reading " + data)
            fileInfo = readFile(filePath + data, colHeaders)
        else:
            continue

        if type(fileInfo) != dict:
            print("Error", data, "has datatype NoneType")
        else:
            readDict.update(fileInfo)

    dfData = pd.DataFrame.from_dict(readDict, orient="index")
    return dfData

def main():
    dataList = os.listdir(rawPath)
    dfArray = []
    print(dataList)
    for dir in dataList:
        df = loadAuth(rawPath+dir+'/')
        dfArray.append(df)
        print(df)
        print("Read " + dir)

    print("Finished reading files, processing data...")

    for dfItem in dfArray:
        if df != dfItem:
            df.join(dfItem)

    print("Saving file...")
    df.to_pickle(savePath+"cleanedDF")

if __name__ == '__main__':
    main()