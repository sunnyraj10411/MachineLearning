import numpy as np
import sys
np.set_printoptions(threshold=np.nan)



dataStore = []
resultStore = []


def polyKernel(vec_x1, vec_x2):
    return ( (zeta + gamma * np.dot(vec_x1, vec_x2.T)) ** Q )
        

def GaussianKernel(vec_x1, vec_x2):
    return ( np.exp(-gamma * (cdist(vec_x2, np.atleast_2d(vec_x1), 'euclidean').T ** 2)).ravel()  )


def linearKernel(vec_x1, vec_x2):
    return np.dot(np.transpose(vec_x1), vec_x2)

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
    Lambda = 0
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

    def predict(self,y,t):
        #print(y.shape,t.shape)
        self.predictDiff = 0
        calculatedResult = 0.0
        for i in range(y.shape[0]):
            self.predictDiff += np.power(y[i]-t[i],2)

        self.predictDiff = np.sqrt(self.predictDiff/y.shape[0])
        print("Deviation:",self.predictDiff)
       
    def polynomial(self,x,y):
        val = np.dot(x.T,y)
        val = self.g*val + self.r
        return(np.power(val,2))

    def createK(self,x,y):
        xPhi = np.zeros(x.shape)
        yPhi = np.zeros(y.shape)

        result = np.zeros([x.shape[1],y.shape[1]])
        print(result.shape)
    
        for i in range(x.shape[1]):
            for j in range(y.shape[1]):
                result[i][j]=self.polynomial(x.T[i],y.T[j])

        #print(x.T.shape,y.shape)
        #print(result.shape)
        #exit()
        return result

    def runRegression(self,valSet,gI,rI,lamb):
        #Read data into validation and test set according to valSet
        print("Div size is:",self.divSize)
        self.npTrainingData.fill(0)
        self.npTrainingResults.fill(0)
        self.npValidationData.fill(0)
        self.npValidationResults.fill(0)
        self.g = gI
        self.r = rI

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

        #data read into validation and test sets

        #find out the average and variance of different dimensions to calculate mean and
        #variance for gaussian basis
        #self.findMeanAndVariance()

        #calculate matrix
        self.populateNumpyMatrix() 


        # t matrix with dimension (3340,1)
        npTrainingResultsMat = np.matrix(self.npTrainingResults).T

        kx = self.createK(self.npTrainingData,self.npValidationData)
        K = self.createK(self.npTrainingData,self.npTrainingData)
        
        self.Lambda = lamb

        I = np.diag(np.ones(K.shape[0]))
        invPart = np.linalg.inv(K + self.Lambda*I)
        #inv_part = np.linalg.inv(K + I)
        invPartKT = np.dot(invPart, npTrainingResultsMat)
        self.vec_predicted_y = np.dot(kx.T, invPartKT)
        self.predict(self.vec_predicted_y,self.npValidationResults)
        #print (self.vec_predicted_y)

        #print self.vec_predicted_y.shape
        #exit()
        return self.predictDiff



if len(sys.argv) != 2:
    print("Usage: ",sys.argv[0],"dataFile")
    exit()
print("The name of the script is",sys.argv[0])
read(sys.argv[1])
a = LinearRegression (dataStore,resultStore)

result = np.zeros([100,100]) #this stores the results of all the experiments

for j in range(3,4):
    r = j
    if j%2 == 0:
        r = 1/j
    print (r)
    for k in range(5,100):
        k = k*10
        for lamb in range(1,2):
            diff = 0
            print(r,lamb)
            for i in range(0,1):
                diff += a.runRegression(i,r,k,lamb*50)
            result[j][lamb] = diff/1.0
            print(result[j][lamb],'j',j,'k',k,'lamb',lamb)
            sys.stdout.flush()
        
            

