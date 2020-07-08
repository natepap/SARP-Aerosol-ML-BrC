import numpy as np
import os
import matplotlib.pyplot as ppl
import cartopy
import cartopy.crs as ccrs

"""This file contains methods for cleaning and processing .ict and .csv files 
for data analysis and input"""

# creates a dictionary of data from a file
# TODO: Include no data values in dict
def readFile(filePath):
    with open(filePath, newline="") as file:
        while True:
            title = file.readline()
            # check if we have reached the data
            # note: maybe change this if not all files have lat/longs
            # TODO: make this more universal/modular
            if "Latitude,Longitude" in title:
                break
        title = title.strip().split(",")
        fileData = {}
        for i in title:
            fileData[i] = []
        for line in file:
            line = line.strip().split(",")
            for i, j in enumerate(title):
                fileData[j].append(line[i])
    return fileData


# takes dict data, isolates and cleans lat/long, and plots flight path
# currently only works for a single file dict, can change params to take
# multiple dicts/data to plot multiple flights

# TODO: take noDatVals from file in readFile
def plotFlightMap(data):

    noDataVals = []

    # converts list to numpy array
    latsArray = np.asarray(data["Latitude"], dtype=float)
    longsArray = np.asarray(data["Longitude"], dtype=float)

    # replaces no data vals with nan for clean plot
    for i,j in enumerate(latsArray):
        if j in noDataVals:
            latsArray[i] = np.nan

    for i,j in enumerate(longsArray):
        if j in noDataVals:
            longsArray[i] = np.nan

    ppl.figure(figsize=(12,7))
    ax = ppl.axes(projection=ccrs.PlateCarree())
    ax.plot(longsArray, latsArray)
    ax.gridlines(draw_labels=True)
    ax.coastlines()
    ax.set_xmargin(0.75)
    ax.set_ymargin(1)
    ppl.show()