import numpy as np
import sys



dataStore = []
resultStore = []

def breakData(lineData):
    sepData = lineData.split()
    #print(sepData)
    pointData = []
    if(sepData[0] == 'M'):
        #return
        pointData.append(1)
    elif( sepData[0] == 'F'):
        #return
        pointData.append(2)
    else:
        pointData.append(3)

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

    #def Q(self,mean,var,x):
        #exp = (x-mean)*(x-mean)
        #exp = exp/(2*var)
        #exp *= -1
        #exp =  np.power(self.e,exp)
        #return exp

    def Q(self,mean,var,x):
        return x

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
        #2 data points have been left out will take care of it later
        #print(len(self.data))

        self.npTrainingData = np.zeros([self.dDim,self.divSize*(self.tDiv-1)])
        self.npTrainingResults = np.zeros([self.divSize*(self.tDiv-1)])
        self.npValidationData = np.zeros([self.dDim,self.divSize])
        self.npValidationResults = np.zeros([self.divSize])

        self.npTrainingMean = np.zeros([self.dDim])
        self.npTrainingVariance = np.zeros([self.dDim])
        #self.npPredictDiff = np.zeros([self.divSize])

        return


    def findMeanAndVariance(self):
        self.npTrainingMean.fill(0)
        self.npTrainingVariance.fill(0)

        for i in range(0,len(self.npTrainingData)):
            self.npTrainingVariance[i] = np.var(self.npTrainingData[i])
            self.npTrainingMean[i] = np.mean(self.npTrainingData[i])
            #print(np.var(self.npValidationData[i]))

        #print("Variance: ",self.npTrainingVariance)
        #print("Mean: ",self.npTrainingMean)


    def populateNumpyMatrix(self):
        npTemArrayMat = np.zeros([len(self.npTrainingData[0]),len(self.npTrainingData)])
        print(len(self.npTrainingData[0]),len(self.npTrainingData))
        for i in range(0,len(self.npTrainingData[0])):
            for j in range(0,len(self.npTrainingData)):
                npTemArrayMat[i][j] = self.Q(self.npTrainingMean[j],self.npTrainingVariance[j],self.npTrainingData[j][i])
                #print(npTemArrayMat[i][j])
        self.npQ = np.matrix(npTemArrayMat)
        print("Matrix dim: ",self.npQ.shape)

        self.npT = self.npTrainingResults[:,None]

    def predict(self):
        #self.PredictDiff.fill(0)
        self.predictDiff = 0
        for i in range(0,len(self.npValidationData[0])):
            calculatedResult = 0.0
            for j in range(0,len(self.npValidationData)):
                calculatedResult +=  self.WML[j]*self.Q(self.npTrainingMean[j],self.npTrainingVariance[j],self.npValidationData[j][i])
            #print(calculatedResult," vs Actual Result",self.npValidationResults[i])
            #self.npPreditDiff[i] = np.power(caculatedResult-self.npValidationResults[i],2)
            self.predictDiff += np.power(calculatedResult-self.npValidationResults[i],2)

        self.predictDiff = np.power(self.predictDiff/len(self.npValidationData[0]),0.5)
        print("Deviation:",self.predictDiff)
        


    def runRegression(self,valSet):
        #Read data into validation and test set according to valSet
        print("Div size is:",self.divSize)
        self.npTrainingData.fill(0)
        self.npTrainingResults.fill(0)
        self.npValidationData.fill(0)
        self.npValidationResults.fill(0)

        for j in range(0,self.dDim):
            index = 0
            for i in range(0,self.tDiv):
                if i != valSet:
                    for k in range(0,len(self.data[i])):
                        self.npTrainingData[j][index] = self.data[i][k][j]
                        if j == 0:
                            self.npTrainingResults[index] =  self.result[i][k]
                        index += 1
                else:
                    for k in range(0,len(self.data[i])):
                        self.npValidationData[j][k] = self.data[i][k][j]
                        if j == 0:
                            self.npValidationResults[k] = self.result[i][k]

        #print(self.npTrainingResults)
        #print(self.npTrainingData[1])
        #print(self.npValidationResults)
        #print(self.npValidationData[1])

        #data read into validation and test sets

        #find out the average and variance of different dimensions to calculate mean and
        #variance for gaussian basis
        self.findMeanAndVariance()

        #calculate matrix
        self.populateNumpyMatrix() 

        #matrix stored in np.Q calculate 

        npTQ = np.transpose(self.npQ)
        npTQxQ = np.dot(npTQ,self.npQ)
        print("Shape after mul:",npTQxQ.shape)
        npTQxQInv = np.linalg.inv(npTQxQ)

        npTQxQInvxnpTQ = np.dot(npTQxQInv,npTQ) 
        print("Shape of Phi:",npTQxQInvxnpTQ.shape)

        self.WML = np.dot(npTQxQInvxnpTQ,self.npT)
        print(self.WML)
        #self.WML  = ([ 0 , 8.91404042,  12.22048227,  11.91342775,   8.72210978,
        #-21.56383663, -12.44905461,   6.33154266])
        self.predict()

if len(sys.argv) != 2:
    print("Usage: ",sys.argv[0],"dataFile")
    exit()
print("The name of the script is",sys.argv[0])
read(sys.argv[1])
a = LinearRegression (dataStore,resultStore)

for i in range(0,5):
    a.runRegression(i)





