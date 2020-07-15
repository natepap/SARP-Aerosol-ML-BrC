import pandas as pd
import numpy as np
import os
import csv
from src.data.dataProcess import readFile
"""
Loads raw data and processes data into pandas dataframes
Saves to Data/Cleaned
"""

rootDir = 'D:/SARP/SARP-Aerosol-ML-BrC/Data/'
rawPath = rootDir + 'Raw/SAGAMERGE/'
cleanPath = rootDir + 'Cleaned/'
processPath = rootDir + 'Processed/'

def loadAuth(filePath):
    rawList = os.listdir(filePath)
    readPD = pd.DataFrame()

    for data in rawList:

        if ".ict" in data:
            print("Reading " + data)
            fileInfo = readFile(filePath + data)
        else:
            continue

        if type(fileInfo) != dict:
            print("Error", data, "has datatype " + type(fileInfo))
        else:
            dfTemp = pd.DataFrame.from_dict(fileInfo)
            readPD = readPD.append(dfTemp, ignore_index=True)

    return readPD

"""
This method takes the dataframes created from the .ict files and saves them
as .csv files containing data points and column headers, as well as replaces
no data vals with nan
"""
def cleanSave(dfList):

    dictNum = 0
    noDataVals = [-9999, -8888, -7777, -99999, -88888, -77777, -999999, -888888, -777777]

    for df in dfList:
        tempDF = df.mask(df.isin(noDataVals), np.nan)
        tempDF.to_csv(cleanPath+"dataFrame"+str(dictNum))
        dictNum+=1


def main():

    mergeList = os.listdir(rawPath)

    dfList = []

    for merge in mergeList:
        dfList.append(loadAuth(rawPath + merge + '/'))

    print("Finished reading files...")

    print("Saving files...")
    cleanSave(dfList)

if __name__ == '__main__':
    main()