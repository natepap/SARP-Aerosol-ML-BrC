import numpy as np
import os
import matplotlib.pyplot as ppl
import cartopy.crs as ccrs
import pandas as pd

"""
This file contains all functions for data preprocessing
In order to take .ict or .csv files from NASA FIREX-AQ MERGE Data and process it,
cleanSave(rawPath) >>> finalDataProcess(cleanPath)

Also contains plotFlightMap function to create a graph of flights, will either add more data
analysis methods for visualization later, or move to a different file for only those methods
"""

rootDir = 'D:/SARP/SARP-Aerosol-ML-BrC/Data/'
rawPath = rootDir + 'Raw/SAGAMERGE/'
cleanPath = rootDir + 'Cleaned/'
savePath = rootDir + 'Processed/'
networkPath = rootDir + 'Network/'

"""creates a dictionary of data from a file"""
def readFile(fileIn):
    lineNo = 0
    try:
        with open(fileIn, newline="") as file:
            notFoundTitle = True
            while notFoundTitle:
                title = file.readline()

                # list of some column headers
                titleSigns = ["Time_Start", "Time_start", "Time_mid", "Time_Mid", "Time_End", "Day_Of_Year",
                              "ext_dry_664", "nCPSPD_stdPT", "InletTemp_K", "nLAScold_stdPT", "nLAShot_stdPT",
                              "totSC450_stdPT", "TD_on"]

                # only the title will have at least two of the headers above
                # so we check if the line has at least two headers
                titleList = list(filter(lambda x: x in title, titleSigns))
                if len(titleList) > 1:
                    notFoundTitle = False

            title = title.strip().split(",")
            for i,t in enumerate(title):
                title[i]=t.strip()
            fileData = {}
            for i in title:
                fileData[i] = []
            for line in file:
                lineNo += 1
                line = line.strip().split(",")
                for i, j in enumerate(title):
                    fileData[j].append(line[i].strip())
        return fileData
    except OSError as err:
        print("Error opening file:", fileIn, err)
    except IndexError as err:
        print("Index error:", fileIn, err)


"""
Reads and loads all the files in a directory into a single dataframe
"""
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
            dfTemp = dfTemp.astype(float)
            readPD = readPD.append(dfTemp, ignore_index=True)

    return readPD


"""
This method takes the dataframes created from the .ict files and saves them
as .csv files containing data points and column headers, as well as replaces
no data vals with nan
"""
def cleanSave(path):

    mergeList = os.listdir(path)

    dfList = []

    for merge in mergeList:
        dfList.append(loadAuth(path+ merge + '/'))

    dictNum = 0
    noDataVals = [-9999, -8888, -7777, -99999, -88888, -77777, -999999, -888888, -777777]

    for df in dfList:
        tempDF = df.mask(df.isin(noDataVals), np.nan)
        tempDF.to_csv(cleanPath+"dataFrame"+str(dictNum))
        dictNum+=1

"""
Takes multiple dataframes created from file groups and combines them into
a single dataframe with the start time of the sample as the index
"""
def combineDF(dfList):
    tempDF = []
    for dF in dfList:
        df = pd.read_csv(cleanPath + dF)
        tempDF.append(df)
    df = pd.concat(tempDF)
    df = df.set_index("Time_Start")
    df.sort_index(axis=1)
    return df


"""takes dict data, isolates and cleans lat/long, and plots flight path"""
def plotFlightMap(dataList):

    # values indicating no data
    noDataVals = [-9999, -8888, -7777]

    # meta list of lats/longs of different flights
    lats = []
    longs = []

    # creates nan list of flight locations for a single data dict at a time
    for data in dataList:
        # converts list to numpy array
        latsArray = np.asarray(data["Latitude"], dtype=float)
        longsArray = np.asarray(data["Longitude"], dtype=float)

        # replaces no data vals with nan for clean plot
        for i, j in enumerate(latsArray):
            if j in noDataVals:
                latsArray[i] = np.nan

        for i, j in enumerate(longsArray):
            if j in noDataVals:
                longsArray[i] = np.nan

        # add to meta list
        lats.append(latsArray)
        longs.append(longsArray)

    # plot size and map type (easiest map to implement, no need for excessive accuracy)
    ppl.figure(figsize=(12,7))
    ax = ppl.axes(projection=ccrs.PlateCarree())

    # plot individual flights on the same graph
    for i, j in enumerate(lats):
        ax.plot(longs[i], lats[i])

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
    ax.coastlines()
    ax.set_xmargin(0.75)
    ax.set_ymargin(1)
    ppl.show()


"""
Takes absorption measurement data and retrieves the Angstrom Exponent of Absorption
for each time index and generates a column dataframe
"""
def angstromExponentAbs(df):
    lowerLambda = "WSAbs320_Aero"
    upperLambda = "WSAbs420_Aero"
    cols = ["", ""]
    for output in df:
        if lowerLambda in output:
            cols[0] = output
        elif upperLambda in output:
            cols[1] = output
    goalDF = df[cols].copy()

    denom = -np.log(320 / 420)

    goalDF["AEA"] = goalDF.apply(lambda x: (np.log(x[0] / x[1]) / denom), axis=1)

    goalDF['AEA'] = goalDF['AEA'].fillna(goalDF['AEA'].median())

    return goalDF['AEA']


"""
Finalizes input and output data for training and testing by filling NaN values, removing columns
that have no values, and excluding certain columns based on content
"""
inputHeaders = []
outputHeaders = []
def finalDataProcess(dfDir):

    dfList = os.listdir(dfDir)
    dataSet = combineDF(dfList)


    for header in dataSet:
        if header not in inputHeaders and header not in outputHeaders:
            if "WEBER" in header:
                outputHeaders.append(header)
            else:
                if "YANG" not in header and "Unnamed" not in header:
                    inputHeaders.append(header)
                elif "Relative_Humidity" in header or "Solar_Zenith_Angle" in header:
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

    outPutSet = angstromExponentAbs(dataSet)
    dataSet.to_csv(savePath + "input")
    outPutSet.to_csv(savePath + "output")

def getInputHeaders():
    return inputHeaders
