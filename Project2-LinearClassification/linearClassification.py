import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import pylab

bcdD = 10
ranD = 3


class linearClassification:
    tDiv = 5
    e = 2.71828
    dDim = 0
    
    def logit(self,w,Phi):
        wT = w.H
        #print(wT,wT*Phi)
        t = 1/(1-np.power(self.e,-(wT*Phi)))
        #print(t)
        return t


    def __init__(self,dataFile,dim):
        self.dataFile = dataFile
        self.dDim = int(dim)+1 #+1 is the added bias
        #print(self.dDim)
        if(self.dDim == bcdD ): #this is the bcd delete the first row as its identifier
            self.dataPd = pd.read_csv(self.dataFile,delimiter=',',names=['0','1','2','3','4','5','6','7','8','9','10'])
        else:
            self.dataPd = pd.read_csv(self.dataFile,delimiter=',',names=['0','1','2'])
        #print(self.dataPd)
        #self.dataPd = pd.read_csv(self.dataFile,delimiter=',',header=None)
        self.W = np.asmatrix(np.ones(self.dDim)).H
        #self.logit(self.W,self.W)


    def makePhiT(self,div):
        if(self.dDim == bcdD ): #this is the bcd delete the first row as its identifier
            phi = self.dataPd[['1','2','3','4','5','6','7','8','9']].as_matrix()
            t = self.dataPd[['10']].as_matrix()
            #print("shape",t.shape[0])
        else:        
            phi = self.dataPd[['0','1']].as_matrix()
            t = self.dataPd[['2']].as_matrix()

        dataLen = t.shape[0]
        #print(t.shape[0])
        #divide into 5 parts

        for i in range(t.shape[0]):
            if t[i] == 2:
                t[i] = 0
            elif t[i] == 4:
                t[i] = 1
            else:
                pass
        #print(t)
        #exit()
        #print(phi)

        self.blockLen = t.shape[0]/self.tDiv
        delLen = t.shape[0]%self.tDiv
        #print(delLen)

        if( self.dDim == bcdD): #delete 3 datapoints
            phiD = np.delete(phi,[1,2,3],axis=0)
            tD = np.delete(t,[1,2,3],axis=0)
        else:
            phiD = phi
            tD = t
        
        phiArr = np.vsplit(phiD,5)
        tArr = np.vsplit(tD,5)
        #print(tArr[0])

        if div == 0:
            self.Phi = phiArr[1]
            self.T = tArr[1]
            self.PhiTest = phiArr[0]
            self.TTest = tArr[0]
            self.PhiArrS = phiArr[0]
        else:
            self.Phi = phiArr[0]
            self.T = tArr[0]
            self.PhiTest = phiArr[1]
            self.TTest = tArr[1]
            self.PhiArrS = phiArr[1]
            
        for i in range(2,5):
            if div == i:
                self.PhiTest = phiArr[i]
                self.TTest = tArr[i]
                self.PhiArrS = phiArr[i]
            else:
                self.Phi = np.concatenate((self.Phi,phiArr[i]))
                self.T = np.concatenate((self.T,tArr[i]))
                
        #print(self.T.shape,self.Phi.shape,self.PhiTest.shape,self.TTest.shape)
        
        ones = np.ones(self.Phi.shape[0])
        onesTest = np.ones(self.PhiTest.shape[0])
        a_2d = ones.reshape((self.Phi.shape[0],1))
        a_2dTest = onesTest.reshape((self.PhiTest.shape[0],1))
        #print(a_2d.shape, self.Phi.shape)

        self.Phi = np.concatenate((a_2d,self.Phi),axis=1)
        self.PhiTest = np.concatenate((a_2dTest,self.PhiTest),axis=1)
        #print(self.Phi)

        #print(self.Phi)
        #self.Phi = np.insert(self.Phi,1,999)
        #print(self.Phi)

        self.T = np.asmatrix(self.T)
        self.Phi= np.asmatrix(self.Phi)
        self.PhiTest = np.asmatrix(self.PhiTest)
        self.TTest = np.asmatrix(self.TTest)
        #print(self.PhiTest)

        self.deltaEW(self.W)


    def deltaEW(self,w):
        totDelta = np.asmatrix(np.zeros(self.Phi.shape[1])).H
        #print(totDelta)
        #print("Phi shape",self.Phi.shape[1])
        for i in range(self.Phi.shape[0]):
            #print(self.Phi[i].H)
            y = self.logit(w,self.Phi[i].H)
            locDelta = self.Phi[i].H*(y - self.T[i] )
            totDelta = totDelta+locDelta
            #print("Printing",t) 
        #print(totDelta)
        #self.predict(w,self.Phi)
        #self.predict(w-totDelta,self.Phi)
        self.W = w - totDelta

    def train(self):
        for i in range (200):
            self.deltaEW(self.W)

        #print(self.W)
        #self.predict(self.W,self.Phi,self.T,0)
        self.predict(self.W,self.PhiTest,self.TTest,0)

        #if randD == 3 then plot graph
        return self.W

    def predict(self,w,phi,t,printFlag):
        totCount = 0
        corCount = 0
        worCount = 0
        for i in range(phi.shape[0]):
            val = w.H*phi[i].H
            totCount += 1
            if val >= 0 and t[i] == 1:
                corCount += 1
            elif val < 0 and t[i] == 0:
                corCount += 1
            else:
                worCount += 1
            if printFlag == 1:
                print(" val ",val, "vs", t[i])
            #print(val)
        print("Total:",totCount,"Correct:",corCount,"Wrong:",worCount, "Correct percentage:",(corCount*100)/totCount)

    def plot(self):
        x = []
        y = []
        x1 = []
        y1 = []
        phi = self.PhiArrS
        w = np.asarray(self.W)
        print(self.W)
        print(phi.shape)
        print(phi[1][1])
        for i in range(phi.shape[0]):
            #print(phi[i][1])
            #print(phi[i][0],phi[i][1])
            if self.TTest[i] == 1:
                x.append(phi[i][0])
                y.append(phi[i][1])
            else:
                x1.append(phi[i][0])
                y1.append(phi[i][1])
        pylab.plot(x,y,'o')
        pylab.plot(x1,y1,'o')
    
        x3 = np.linspace(-10,120,100)
        y3 = (-w[0]-w[1]*x3)/w[2]
        print(x3.shape,y3.shape)

        pylab.plot(x3,y3)

        pylab.axis('equal')
        pylab.show()
        pass
    


def main():

    if len(sys.argv) != 3:
        print("Usage: ",sys.argv[0],"dataFile dimensions")
    a = linearClassification(sys.argv[1],sys.argv[2])

    for i in range(5):
        a.makePhiT(i)
        W = a.train()
        if ranD == 3:
            a.plot()
            exit()
    #we will plot the data



if __name__ == "__main__":main()
