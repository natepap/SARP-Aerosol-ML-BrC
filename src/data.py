import numpy as np
import os
import matplotlib.pyplot as ppl
import cartopy.crs as ccrs
import pandas as pd

"""
v1_SARP

This file contains all functions for data preprocessing
In order to take .ict or .csv files from NASA FIREX-AQ MERGE Data and process it,
cleanSave(rawPath) >>> finalDataProcess(cleanPath)

Also contains plotFlightMap function to create a graph of flights, will either add more data
analysis methods for visualization later, or move to a different file for only those methods

Title Flags: ["Time_Start", "Time_End", "Day_Of_Year", "Fractional_Day"]

Removes: YANG - meteorological, Unnamed - index?, Fractional_Day - disruptive data + not useful, Time - ?
"""

"""
v2_thesis
10/2/2020

goals: reorganize data pipeline to exclude duplicate data, look at NaN handling and data exclusion, and
do more EDA, redownload data from NASA LaRC site https://www-air.larc.nasa.gov/missions.htm with different
parameters

*Redownloaded data from NASA LaRC: All PIs (check w Roya/Andreas about this but also train a model with
    everything included as a baseline)
    
*
"""

rootDir = 'D:/SARP/SARP-Aerosol-ML-BrC/Data/'
rawPath = rootDir + 'Raw/SAGAMERGE/'
dataPath = rootDir +'FIREX-AQ_5s/'
cleanPath = rootDir + 'Cleaned/'
procPath = rootDir + 'Processed/'
networkPath = rootDir + 'Network/'

noDataVals = [-9999, -8888, -7777, -99999, -88888, -77777, -999999, -888888, -777777]

"""creates a dictionary of data from a file -- See Jesse Bausell's lessons for template"""

'''
def readFile(fileIn):
    lineNo = 0
    try:
        with open(fileIn, newline="") as file:
            notFoundTitle = True
            while notFoundTitle:
                title = file.readline()

                # list of some column headers
                titleSigns = ["Time_Start", "Time_End", "Day_Of_Year", "Fractional_Day"]

                # only the title will have at least two of the headers above
                # so we check if the line has at least two headers
                titleList = list(filter(lambda x: x in title, titleSigns))
                if len(titleList) > 1:
                    notFoundTitle = False

            title = title.strip().split(",")
            for i, t in enumerate(title):
                title[i] = t.strip()
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
'''





'''
***UPDATED .ict to .csv DataFrame pipeline***

Reads data from multiple .ict files and saves to 'foldername.csv' to easily be converted to a
dataframe directly with pandas method read_csv()
'''


def collect(folderName):
    fileList = os.listdir(dataPath+folderName)
    dataDF = pd.DataFrame()
    titleList = ['Time_Start', 'Time_Stop', 'Day_Of_Year', 'Day_Of_Year_stdev', 'Day_Of_Year_points',
                 'Latitude']


    for file in fileList:
        if '.ict' in file:
            try:
                with open(dataPath+folderName+file, newline="") as data:
                    FoundData = False


                    while not FoundData:
                        line = data.readline()
                        lineSkip = 0

                        lineFilter = list(filter(lambda x: x in line, titleList))

                        if len(lineFilter) == len(titleList):
                            FoundData = True
                            break
                        
                        lineSkip += 1

                    if FoundData:
                        dataDF = pd.read_csv(dataPath+folderName+file, skiprows=lineSkip)

            except OSError as err:
                print("Error opening file:", file, err)




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
        dfList.append(loadAuth(path + merge + '/'))

    dictNum = 0

    for df in dfList:
        tempDF = df.mask(df.isin(noDataVals), np.nan)
        tempDF.to_csv(cleanPath + "dataFrame" + str(dictNum))
        dictNum += 1


"""
Takes multiple dataframes created from file groups and combines them into
a single dataframe
"""


def combineDF(dfList):
    tempDF = []
    for dF in dfList:
        df = pd.read_csv(cleanPath + dF)
        tempDF.append(df)
    df = pd.concat(tempDF, ignore_index=True)
    return df


"""takes dict data, isolates and cleans lat/long, and plots flight path"""


def plotFlightMap(dataList):
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
    ppl.figure(figsize=(12, 7))
    ax = ppl.axes(projection=ccrs.PlateCarree())

    # plot individual flights on the same graph
    for i, j in enumerate(lats):
        ax.plot(longs[i], lats[i])

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
    ax.coastlines()
    ax.set_xmargin(0.75)
    ax.set_ymargin(1)
    ppl.show()


'''
Retrieves Angstrom Exponent based on water-soluble & methanol soluble 
absorption at 320 and 420 nm
'''


# TODO: make label flags direct copies of title strings
def angstromExponentAbs(df):
    # need to sum across water+methanol soluble absorption
    lowerLambdaW = 'WSAbs320'
    lowerLambdaM = 'MSAbs320'
    upperLambdaW = 'WSAbs420'
    upperLambdaM = 'MSAbs420'

    # copy the full name to a list
    cols = ["", "", "", ""]
    for output in df:
        if lowerLambdaW in output:
            cols[0] = output
        elif lowerLambdaM in output:
            cols[1] = output
        elif upperLambdaW in output:
            cols[2] = output
        elif upperLambdaM in output:
            cols[3] = output

    # copy relevant columns to a new dataframe
    tempDF = df[cols].copy()

    # remove negative terms and avoid dividing by zero for log/division
    # (0.001 is small enough that the calculated result will go to)
    tempDF = tempDF.mask(tempDF < 0, 0.0001)

    # sum water+methanol
    sum_320 = tempDF[cols[0]] + tempDF[cols[1]]
    sum_420 = tempDF[cols[2]] + tempDF[cols[3]]

    # new dataframe
    data = [sum_320, sum_420]
    goalDF = pd.DataFrame(data).transpose()

    # AAE = -log(absorb_1/absorb_2)/log(lambda_1/lambda_2)
    denom = -np.log(320 / 420)

    goalDF['AAE'] = goalDF.apply(lambda x: (np.log(x[0] / x[1]) / denom), axis=1)

    goalDF['AAE'] = goalDF['AAE'].fillna(goalDF['AAE'].median())

    return goalDF['AAE']


"""
Takes absorption measurement data and retrieves the particle light absorption (bAP-BrC)
for each corresponding time index and generates a column dataframe
"""


# TODO: make label flags direct copies of title strings
def particleLightAbs(df):
    # BrC is best categorized at 365 nm (not enough mist chamber data so we stick
    # to Aero (filter)
    waterS = "WSAbs365"
    methanolS = "MSAbs365"
    cols = []
    for output in df:
        if (waterS in output or methanolS in output) and 'Aero' in output:
            cols.append(output)

    goalDF = df[cols].copy()

    goalDF['bAP'] = goalDF.sum(axis=1)

    return goalDF


"""
Finalizes input and output data for training and testing by filling NaN values, removing columns
that have no values, and excluding certain columns based on content
"""


# TODO: make label flags direct copies of title strings
def finalDataProcess(dfDir):
    dfList = os.listdir(dfDir)
    dataSet = combineDF(dfList)

    # List of columns to be included in final input/output data
    inputHeaders = []
    outputHeaders = []

    for header in dataSet:

        # no duplicates
        if header not in inputHeaders and header not in outputHeaders:

            # FIREXAQ 2019 MERGE data - WEBER has BrC goal measruements
            if "WEBER" in header:
                outputHeaders.append(header)
            else:

                # YANG contains largely positional data, exceptions in the elif, drop extra index col
                # and drop time
                if "YANG" not in header and "Unnamed" not in header and "Time" not in header and \
                        "Fractional_Day" not in header:
                    inputHeaders.append(header)
                elif "Relative_Humidity" in header:
                    inputHeaders.append(header)

    # list all columns that have no data whatsoever
    # to be dropped all at once
    naList = []
    for inpuT in inputHeaders:
        if dataSet[inpuT].isna().all():
            naList.append(inpuT)

        # fill nan values with median
        elif dataSet[inpuT].isna().any():
            dataSet[inpuT] = dataSet[inpuT].fillna(dataSet[inpuT].median())

    for inpuT in naList:
        inputHeaders.remove(inpuT)

    for output in outputHeaders:
        dataSet[output] = dataSet[output].fillna(dataSet[output].median())

    # if there are any nan values left for some reason, drop the datapoint with them
    x = dataSet.dropna(axis=1)

    outPutSetB = particleLightAbs(dataSet)
    outPutSetA = angstromExponentAbs(dataSet)

    x = x[inputHeaders]
    yB = outPutSetB['bAP']
    yA = outPutSetA

    x.to_csv(procPath + "input", index=False)
    yB.to_csv(procPath + "bap", index=False)
    yA.to_csv(procPath + "AAE", index=False)

# cleanSave(rawPath)
# finalDataProcess(cleanPath)
