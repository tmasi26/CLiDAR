"""
StarAODV4?
John E. Barnes
Jake P. Simoes
"""

import collections
import os
import csv
from math import log10, pi, sin, cos, radians

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
from matplotlib.widgets import Slider, Button, TextBox


class ErrorMessenger():
    """Provides various methods to signal errors to the user"""

    @staticmethod
    def testError(message):
        print(message)


class constants():
    """Contains constant values and or methods which return them"""

    MaxCfs = 30                     # Maximum number of confiurations read in
    NCfgInUse = 8              # Default configuration in use (from Config file)
    MaxStars = 20                    # Maximum number of APFs read in
    N181 = 181                      # Index for phase function arrays
    Deg = np.arange(N181)          # Create degree array
    Rad = Deg * pi / 180          # Create radian array
    # Physical constants
    CAir = 4.851E-23    # Conversion from g/m^3 to molec/m^3
    MIdeal = 7.24290e18     # Ideal gas constant


    @staticmethod
    def RefractIndex(Waveln):
        """ Index of refraction in air, Waveln in microns
            Ref:                """
        return 1.0002876 + 0.000001629/Waveln**2 + 0.0000000136/Waveln**4

    @staticmethod
    def RayScat(Waveln):
        """ Rayleigh scattering cross section not including King factor correction
            at STP,  Waveln (microns), see ResearchVarious.xls    """
        return 3.834E-28/Waveln**4.05

    @staticmethod
    def PRay(theta, phi):
        """  Molecular phase function, theta is observation angle,
            phi is polarization  """
        P = 1.5 * ((sin(radians(phi))**2 +
            (cos(radians(theta))**2 * (cos(radians(phi)))**2)))
        return P
    
    @staticmethod
    def CalculateAODs(SBIG: 'SBIG', starHi: 'Star', starLo: 'Star'):
        """Calculate AOD for pair of stars"""

        # If intensities are valid then calculate the magnitude?
        if starHi.IStar > 0:
            HiStarMag = -2.5 * log10(starHi.IStar)
        else:
            # Perhaps some kind of fallback value?
            HiStarMag = 1
        if starLo.IStar > 0:
            LowStarMag = -2.5 * log10(starLo.IStar)
        else:
            LowStarMag = 1

        # What is AA?
        AA = (starHi.AirMass - starLo.AirMass) * 1.086
        if AA != 0.0:
            TotalOD = ((HiStarMag - LowStarMag) -
                        (starHi.Mag - starLo.Mag))/AA
        else:
            TotalOD = 0
            
        MolOD = SBIG.Pressure/1013 * 0.00879 * SBIG.Wavelength**-4.09
        
        AOD = TotalOD - MolOD

        print('StarA  StarB     dAirmas    TotOD    MolOD     AOD')
        print('{0:6} {1:6}{2:10.3f}{3:10.3f}{4:10.4f}{5:10.4f}'.format(starHi.StarName, starLo.StarName, AA, TotalOD,
                    MolOD, AOD))


class SBIG():
    """A class which stores an SBIG file alongside its attributes and stars"""
    def __init__(self):
        self.Stars = [] # Stores individual star objects.
        self.data = 0 # Stores the SBIG file data.
        self.iFirst = 0
        self.iLast = 0
        self.File = ""
        self.StarFile = ""
        self.DarkFile = ""
        self.YEAR = 0
        self.MN = 0
        self.DAY = 0
        self.HOUR = 0
        self.MIN = 0
        self.SEC = 0
        self.Lat = 0
        self.Long = 0
        self.Alt = 0
        self.JDay = 0
        self.Wavelength = 0
        self.Pressure = 0
        self.TimeISO = ""

    def readCSV(self, fileName):
        """Reads in data from the csv file. Puts SBIG relevant information into the SBIG object. 
           Creates several star objects to hold star data."""
        
        if not fileName:
            ErrorMessenger.testError("No file provided.")
            return
        
        Cffile = open(fileName, "r")               # Open the csv.
        self.StarFile = os.path.basename(fileName) # Get the file name.
        
        with Cffile as csvfile:
            csvreader = csv.reader(csvfile)         # creating a csv reader object
            next(csvreader)       # Skip row of lables

            # extracting each data row one by one
            Crows = []
            for Crow in csvreader:           
                Crows.append(Crow)

            # Setting the SBIG file info. (filenames, date, etc...)
          
            self.File = Crows[0][0]
            self.DarkFile = Crows[0][1]
            self.YEAR = int(Crows[0][2])
            self.MN = int(Crows[0][3])
            self.DAY = int(Crows[0][4])
            self.HOUR = int(Crows[0][5])
            self.MIN = int(Crows[0][6])
            self.SEC = int(Crows[0][7])
            self.Lat = float(Crows[0][8])
            self.Long = float(Crows[0][9])
            self.Alt = float(Crows[0][10])
            self.JDay = float(Crows[0][11])
            self.Wavelength = float(Crows[0][12])
            self.Pressure = float(Crows[0][13])
            self.TimeISO = str(self.YEAR)+'-'+str(self.MN)+\
                '-'+str(self.DAY)+' '+str(self.HOUR)+':'+ \
                str(self.MIN)+':'+str(self.SEC)

            # Creating Star objects and filling in their data
            print('Read:  #   Name          Mag      AirMass  I(Star)    Time')
            for index, Crow in enumerate(Crows[0:csvreader.line_num+1]):
                # Making a new Star object and appending it to this SBIG objects array
                self.Stars.append(Star())

                # Filling in all the data...
                currentStar = self.Stars[index] 
                currentStar.parentSBIG = self
                currentStar.StarName = Crow[14]
                currentStar.Constallation = Crow[15]
                currentStar.Mag = float(Crow[16])
                currentStar.AppMag = float(Crow[17])
                currentStar.AirMass = float(Crow[18])
                currentStar.ColorIndex = float(Crow[19])
                currentStar.AzDeg = float(Crow[20])
                currentStar.AzMin = float(Crow[21])
                currentStar.AzSec = float(Crow[22])
                currentStar.xPix = int(Crow[23])
                currentStar.yPix = int(Crow[24])
                currentStar.AzAng = float(Crow[25])
                currentStar.IStar = float(Crow[26])
                print('Read:{0:3d}{1:12s}{2:10.3f} {3:8.3f} {4:12.3f} {5:20s}'.
                      format(index,'   '+currentStar.StarName, currentStar.Mag,
                      currentStar.AirMass, currentStar.IStar, self.TimeISO))
        Cffile.close()

    def readSBIGFile(self, fileName):
        self.data = SBIGData(fileName)

    def handleDark(this, darkFile):
        """   Subtract dark SBIG image pixel by pixel  """
    
        """The following steps are done to prevent overflow. The explanation is as follows.
            We first convert our light file data into int32. The smallest and largest number in int32 are magnitudes larger
            than those for our original uint16, so we will never overflow. Then we do our subtraction.
            After, we check for any negative values (because int32 is signed we allow them, unlike uint16).
            Any negative values mean that the dark file had a higher brightness than the actual photo.
            For our purposes, that means a value of zero, so we check for negative values and set them to zero.
            Afterwards we convert our file back into the original data type. All done!"""
        IP16 = this.data.array.astype(np.int32)
        IP16 = np.subtract(IP16, darkFile.data.array)
        IP16[IP16 < 0] = 0
        this.data.array = IP16.astype(np.uint16)
        

class Star():
    """A Star object which holds all relevant data and functions."""
    def __init__(self):
        # Fields which define a star.
        self.parentSBIG = SBIG()
        self.StarName = ""
        self.Constallation = ""
        self.Mag = 0
        self.AppMag = 0
        self.AirMass = 0
        self.ColorIndex = 0
        self.AzDeg = 0
        self.AzMin = 0
        self.AzSec = 0
        self.xPix = 0
        self.yPix = 0
        self.AzAng = 0
        self.IStar = 0

    def PlotStar(self, minBounding=4, maxBounding=18):
        """Plots the stars pixels, work in progress"""
        pass
        # It = np.arange(maxBounding)
        # AAvg = np.average(self.parentSBIG.data.array[minBounding:maxBounding+1])
        # SStd = np.std(self.parentSBIG.data.array[minBounding:maxBounding+1])
        # plt.ion()
        # plt.figure(60, figsize=(7, 8))
        # plt.subplot(2, 1, 1)
        # plt.title(self.StarName + ': IStar = ' +
        #           str(round(AAvg, 2)) + ' +/- ' + str(round(SStd/AAvg*100, 2)) + '%')
        # cp = plt.pcolormesh(self.parentSBIG.data.array[self.yPix-21:self.yPix+21, self.xPix-21:self.xPix+21])
        # plt.colorbar(cp)
        # plt.subplot(2, 1, 2)
        # plt.plot(It[minBounding:maxBounding], self.BackAvg[minBounding:maxBounding]*10)
        # plt.plot(It[minBounding:maxBounding], self.IStarArr[minBounding:maxBounding])
        # plt.plot(It[minBounding:maxBounding+1], self.IStarArr[minBounding:maxBounding+1], marker="D")
        # plt.xlabel('Side of Box / 2 - 1 (pixels)')
        # plt.ylabel('Total Star Counts, Background*10')
        # plt.grid(which='both', axis='both')
        # plt.show(block=False)
        # plt.pause(0.9)
        # plt.show()
    def calcStarSig(self, SBIG, minBounding=1, maxBounding=18):


        """Calculates the intensity of a star.

        Performs...

        Args:
            SBIG: The SBIG file to get data from.
            minBounding: The smallest size of the bounding box.
            maxBounding: The largest size of the bounding box.

        Returns:
            Some value.
        """

        # Local params for star coordinate
        xCoord = self.yPix
        yCoord = self.xPix
        
        print(' jj  nBord  nSig  nTot  Border  outerSum  Total  IStar')
        BackAvg = np.zeros(maxBounding)
        IStarArr = np.zeros(maxBounding)
        
        # Speculating what this does.
        # It seems like it creates an increasing size of bounding boxes to measure the signal of the star.
        for jj in range(minBounding, maxBounding):
            # Bounds for this iterations box.
            xLeftBound = xCoord-jj
            xRightBound = xCoord+jj+1       # np.sum uses i + 1 as upper limit
            yLeftBound = yCoord-jj
            yRightBound = yCoord+jj+1
            # Creating 2D arrays using these bounds
            innerBoxArray = SBIG.array[xLeftBound+1:xRightBound-1, yLeftBound+1:yRightBound-1]
            outerBoxArray = SBIG.array[xLeftBound:xRightBound, yLeftBound:yRightBound]
            # Getting their sums...
            innerSum = np.sum(innerBoxArray)
            outerSum = np.sum(outerBoxArray)
            borderSum = outerSum - innerSum
            nBord = outerBoxArray.size - innerBoxArray.size

            BackAvg[jj] = borderSum/nBord
            IStarArr[jj] = outerSum - BackAvg[jj]
            print('  {0:2d}{1:5d}{2:5d}{3:5d}{4:8.0f}{5:8.0f}{6:8.0f}{7:10.2f}{8:10.2f}'
                  .format(jj, nBord, outerBoxArray.size, innerBoxArray.size, borderSum, outerSum, innerSum,
                          IStarArr[jj], BackAvg[jj]))
           
        print("Average Intensity for star {} = ".format(self.StarName), np.average(IStarArr[minBounding:]))

class manualIntensityBounder():
        def __init__(self, Star, SBIG, maxBounding=18):
            self.zoom = 0
            self.horizontalShift = 0
            self.verticalShift = 0
            self.maxBounding = maxBounding
            self.Star = Star
            self.SBIG = SBIG

        def calcStarSig(self):
            def zoomIn(val):
                self.zoom += 1
                data = innerBoxArray.copy()
                data = data[self.zoom:data[0].size - self.zoom, self.zoom:data[0].size - self.zoom]
                img.set_array(data)
                text.set_text(data.mean())
                ax.imshow(img.get_array(), interpolation='none')
                fig.canvas.draw()
                plt.show()
            # Local params for star coordinate
            xCoord = self.Star.yPix
            yCoord = self.Star.xPix

            # Bounds for this iterations box.
            xLeftBound = xCoord - self.maxBounding
            xRightBound = xCoord + self.maxBounding + 1  # np.sum uses i + 1 as upper limit
            yLeftBound = yCoord - self.maxBounding
            yRightBound = yCoord + self.maxBounding + 1

            # Creating 2D arrays using these bounds
            innerBoxArray = self.SBIG.array[xLeftBound + 1:xRightBound - 1, yLeftBound + 1:yRightBound - 1]
            outerBoxArray = self.SBIG.array[xLeftBound:xRightBound, yLeftBound:yRightBound]
            fig, ax = plt.subplots()
            img = ax.imshow(innerBoxArray, interpolation='none')
            plt.title(self.Star.StarName)
            axZoomIn = plt.axes([0.03, 0.1, 0.1, 0.03])
            # axMean = plt.axes([0.03, 0.2, 0.1, 0.03])
            text = plt.text(.5, 5, outerBoxArray.mean())
            bnext = Button(axZoomIn, 'Next')
            bnext.on_clicked(zoomIn)
            # axSlider = Slider(
            #     ax=ax_slider,
            #     label='Threshold',
            #     valmin=innerBoxArray.min(),
            #     valmax=innerBoxArray.max(),
            #     valinit=1,
            #     orientation='vertical'
            # )
            # axSlider.on_changed(update)
            plt.show()


class SBIGData():
    def __init__(self, fileName, array=np.zeros((1)), headerDict=False):
        if array.any() and headerDict:
            self.headerDict = headerDict
            self.filename = True
            self.array = array
            self.height = headerDict["Height"]
            self.width = headerDict["Width"]
            self.type = "SBIG Image"
        else:
            self.filetitle = fileName.split("/")[-1]
            self.filename = fileName
            self.compressed = False
            self.headerDict = collections.OrderedDict()
            openFile = open(self.filename, "rb")
            headerTxt = str(openFile.read(2048))
            headerList = headerTxt.split("\\n\\r")
            self.type = headerList.pop(0)[2:]
            if self.type == ("SBIG Compressed Image") or ("ST-237 Compressed Image"):
                self.compressed = True
            if self.type == 'SBIG Model Image':
                self.compressed = False
            endTrail = headerList.pop(len(headerList)-1)
            end = headerList.pop(len(headerList)-1)
            for entry in headerList:
                a = entry.split(" = ", 1)
                self.headerDict[a[0]] = a[1]
            self.height = int(self.headerDict["Height"])
            self.width = int(self.headerDict["Width"])
            sMN = self.headerDict["Date"][0]
            i = 1
            while self.headerDict["Date"][i]!='/':
                sMN = sMN + self.headerDict["Date"][i]
                i = i + 1
            # Stars[0].MN = int(sMN)
            # i = i + 1
            # sDAY = self.headerDict["Date"][i]
            # while self.headerDict["Date"][i+1]!='/':
            #     sDAY = sDAY + self.headerDict["Date"][i+1]
            #     i = i + 1
            # Stars[0].DAY = int(sDAY)
            # i = i + 2
            # Stars[0].YEAR = int(self.headerDict["Date"][i:i+4])
            # if Stars[0].YEAR < 1000: Stars[0].YEAR = Stars[0].YEAR + 2000    # Fix abreviated year
            # Stars[0].HOUR = int(self.headerDict["Time"][0:2])
            # Stars[0].MIN = int(self.headerDict["Time"][3:5])
            # Stars[0].SEC = int(self.headerDict["Time"][6:8])
            # print('Read image:', Stars[0].YEAR, '/', Stars[0].MN, '/',
            #     Stars[0].DAY, ' ', Stars[0].HOUR,
            #     ':', Stars[0].MIN, ':', Stars[0].SEC)

            # Check if compressed or not###
            if self.compressed == True:###If compressed, follow compression scheme to decompress###
                pixArray = np.zeros((self.height, self.width), dtype=np.uint16)
                for i in range(0, self.height):
                    a = openFile.read(2)    # Check Row Length Bytes###
                    rowLen = (a[1]*256 + a[0])
                    if rowLen == (2*self.width):###(Uncompressed Row,read straight in)###
                        for j in range(0, self.width):
                            a = openFile.read(2)
                            pixArray[i, j] = (a[1]*256 + a[0])
                    else:
                        j = 0
                        a = openFile.read(2)
                        pixVal = (a[1]*256 + a[0])    # (Needed this time to establish previous pixel)###
                        pixArray[i, j] = pixVal     # First value###
                        while j < self.width-1:     # at != rowLen:##-2?  ###While j <= self.width:?
                            j += 1
                            a = openFile.read(1)
                            if a[0] != 128:
                                if a[0] < 128:
                                    pixVal = pixVal + a[0]
                                    pixArray[i, j] = pixVal
                                elif a[0] > 128:
                                    pixVal = pixVal + (a[0]-256)
                                    pixArray[i, j] = pixVal
                            else:   # a[0] = 128:
                                a = openFile.read(2)
                                pixVal = (a[1]*256+a[0])
                                pixArray[i, j] = pixVal
                self.type = "SBIG Image"
            elif self.compressed == False:  # If not Compressed, read straight from bytes###           
                pixArray = np.zeros((self.height, self.width), dtype=np.uint16)
                for i in range(0, self.height):
                    for j in range(0, self.width):
                        a = openFile.read(2)
                        byte2 = a[0]
                        byte1 = a[1]
                        pixVal = (byte1*256 + byte2)
                        pixArray[i, j] = pixVal
            openFile.close()
            self.array = pixArray

            
    
    def update_Header(self):
        headtxt = self.type + "\n\r"
        headerDict = self.headerDict
        headerDict["Height"] = self.height
        headerDict["Width"] = self.width
        # Normal Headers#
        for key in headerDict:
            headtxt += (key + " = " + str(headerDict[key]) + "\n\r")
        headtxt += "End\n\r\x1a"
        nul = "\x00"*(2048-len(headtxt))
        headtxt += nul
        return headtxt

    def getHeader(self):        # Needed?
        return self.headerDict

    def updateArray(self, array):
        # update height, width, # of arrays avged.#
        resolution = array.shape
        self.height = resolution[0]
        self.width = resolution[1]
        self.array = array
        # self.avgnum =

    def getPlot(self):
        plotArray = np.zeros((self.height, self.width))
        for i in range(0, self.height):
            for j in range(0, self.width):
                pixVal = self.array[i, j]/(65535)
                plotArray[i, j] = pixVal
        return plotArray

    def showImage(self): 
        """Currently converts the SBIG into a png file and displays it"""
        # Apparently modern computers only support 256 shades of grey. Tragic. So we must downscale our beautiful 16 bit data for 8 bit displays.
        # The default output from downscaling is awfully dim, so we square the result and clamp any overflows.
        # Accuracy isn't necessary, we're just trying to make it easy for humans to see.
        eightBitImage = (self.array//256)**2
        eightBitImage[eightBitImage > 256] = 256
        Image.fromarray(eightBitImage).convert("L").show()


# Making the file path relative just so it's easy to run for you...
dirname = os.path.dirname(__file__)

# Here we're creating SBIG objects, containers for information relevant to SBIG files.
lightFile = SBIG()
lightFile.readCSV(dirname + r"\Bahamas Dataset\Bahamas.csv") 
# lightFile.readSBIGFile(dirname + r"\Bahamas Dataset\Image30_30s_ccd1.SBIG")
#
# # This is an object for our dark file.
# darkFile = SBIG()
# darkFile.readSBIGFile(dirname + r"\Bahamas Dataset\darkframe_30s_ccd1.SBIG")
#
# # Subtracting the dark file...
# lightFile.handleDark(darkFile)
#
# # Just iterating through each star and calculating their intensity
# for star in lightFile.Stars:
#     manualIntensityBounder(star, lightFile.data).calcStarSig()

# Calculating AODs for reasonable star pairs...
constants.CalculateAODs(lightFile, lightFile.Stars[3], lightFile.Stars[4])
constants.CalculateAODs(lightFile, lightFile.Stars[3], lightFile.Stars[5])
constants.CalculateAODs(lightFile, lightFile.Stars[1], lightFile.Stars[4])
constants.CalculateAODs(lightFile, lightFile.Stars[1], lightFile.Stars[5])

# constants.CalculateAODs(lightFile, lightFile.Stars[14], lightFile.Stars[15])
# constants.CalculateAODs(lightFile, lightFile.Stars[14], lightFile.Stars[17])
# constants.CalculateAODs(lightFile, lightFile.Stars[14], lightFile.Stars[18])
# constants.CalculateAODs(lightFile, lightFile.Stars[16], lightFile.Stars[13])
# constants.CalculateAODs(lightFile, lightFile.Stars[16], lightFile.Stars[15])
# constants.CalculateAODs(lightFile, lightFile.Stars[16], lightFile.Stars[17])
# constants.CalculateAODs(lightFile, lightFile.Stars[16], lightFile.Stars[18])

# A good way to test if an image is actually being read correctly is to display it.
# lightFile.data.showImage()
