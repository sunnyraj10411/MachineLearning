import pandas as pd
import sys
import numpy as np
import tempfile

irisD = 4
randD = 3

flagData = 0

class MixtureModelClassification:
    tDiv = 5
    e = 2.71828

    def __init__(self,dataFile,dim):
        self.dataFile = dataFile
        self.dDim = int(dim)#+1 #+1 is the added bias
        if(self.dDim == irisD ): #this is the bcd delete the first row as its identifier
            self.dataPd = pd.read_csv(self.dataFile,delimiter=',',names=['0','1','2','3','4'])
            self.dataPd = self.dataPd.sample(frac=1).reset_index(drop=True)
        else:
            self.dataPd = pd.read_csv(self.dataFile,delimiter=',',names=['0','1','2'])
            global flagData
            flagData = 1
        self.W = np.asmatrix(np.ones(self.dDim)).H


    def makePhiT(self,div):
        if(self.dDim == irisD ): #this is the bcd delete the first row as its identifier
            phi = self.dataPd[['0','1','2','3']].as_matrix()
            t1 = self.dataPd[['4']].as_matrix()
            t = np.ones([phi.shape[0],1])
            for i in range(len(t)):
                if t1[i] == 'Iris-setosa':
                    t[i] = int(0)
                elif t1[i] == 'Iris-versicolor':
                    t[i] = int(1)
                else:
                    t[i] = int(2)
        
        else:        
            phi = self.dataPd[['0','1']].as_matrix()
            t = self.dataPd[['2']].as_matrix()

        dataLen = t.shape[0]

        self.blockLen = t.shape[0]/self.tDiv
        delLen = t.shape[0]%self.tDiv

        if( self.dDim == irisD): #delete 3 datapoints
            phiD = phi
            tD = t
        else:
            phiD = phi
            tD = t
        
        phiArr = np.vsplit(phiD,5)
        tArr = np.vsplit(tD,5)

        self.Phi = phiArr[0]
        self.T = tArr[0]
            
        for i in range(1,5):
            self.Phi = np.concatenate((self.Phi,phiArr[i]))
            self.T = np.concatenate((self.T,tArr[i]))
                
        
        ones = np.ones(self.Phi.shape[0])
        a_2d = ones.reshape((self.Phi.shape[0],1))
        print(self.Phi.shape,'and',self.T.shape)

    def KMeans(self):
        

def main():

    if len(sys.argv) != 4:
        print("Usage: ",sys.argv[0],"dataFile dimensions expectedCluster")
        exit()
    a = MixtureModelClassification(sys.argv[1],sys.argv[2])

    a.makePhiT(1)
    a.KMeans()
    #a.train()
    #a.predict(a.PhiTest,a.TTest)
    
    #    W = a.train()
        #if ranD == 3:
        #    a.plot()
        #    exit()
    #we will plot the data



if __name__ == "__main__":main()
