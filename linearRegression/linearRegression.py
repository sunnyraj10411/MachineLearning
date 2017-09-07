import numpy as np
import sys



dataStore = []
resultStore = []

def breakData(lineData):
    sepData = lineData.split()
    #print(sepData)
    pointData = []
    if(sepData[0] == 'M'):
        pointData.append(1)
    else:
        pointData.append(2)
    for i in range(1,8):
        #print(sepData[i])
        pointData.append(float(sepData[i]))
    #print(pointData)
    resultStore.append(float(sepData[8]))
    #print(sepData[8])
    dataStore.append(pointData)
    #print("*************")

def read(dataFile):
    print("In read function file:",dataFile)
    with open(dataFile,"r") as dataFileO:
        for line in dataFileO:
            breakData(line)
    numData = len(dataStore)
    #print(readData)
    print(numData,len(resultStore))



class LinearRegression():

    #data
    #result
    
    tDiv = 5
    e = 2.71828
    dDim = 8

    def Q(self,mean,sd,x):
        exp = (x-mean)^2
        exp = exp/(2*sd*sd)
        exp *= -1
        exp =  e^(exp)
        return exp

    def __init__(self,dataStore,result):
        #divide the dataStore into 5 parts
        self.data = []
        self.result = []
        dataItr = 0
        self.divSize = int(len(dataStore)/self.tDiv)
        #print(divSize,"for: ",divSize*5)
        for i in range(0,self.tDiv):
            divData = []
            divResult = []
            for j in range(0,self.divSize):
                divData.append(dataStore[i*self.divSize+j])
                divResult.append(resultStore[i*self.divSize+j])
            self.data.append(divData)
            self.result.append(divResult)
        #print(self.data[i])
        return


        #2 data points have been left out will take care of it later
        #print(len(self.data))
    def runRegression(self,valSet):
        #find mean for each dimension for populating Q
        self.npTrainingData = np.zeros([self.dDim,self.divSize*(self.tDiv-1)])
        self.npTrainingResults = np.zeros()
        self.npValidationData = np.zeros([1,self.divSize*(self.tDiv-1)])
        self.npValidationResults = np.zeros()
        for j in range(0,self.dDim):
            index = 0
            for i in range(0,self.tDiv):
                if i != valSet:
                    for k in self.data[i]:
                        self.npTrainingData[j][index] = k[j] 
                        index += 1

        #print(self.npTrainingData)
       


if len(sys.argv) != 2:
    print("Usage: ",sys.argv[0],"dataFile")
    exit()
print("The name of the script is",sys.argv[0])
read(sys.argv[1])
a = LinearRegression (dataStore,resultStore)
a.runRegression(1)




