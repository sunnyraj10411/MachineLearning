import pandas as pd
import sys
import numpy as np
import tempfile

irisD = 4
randD = 3

flagData = 0

class tensorFlowClassification:
    tDiv = 5
    e = 2.71828

    def __init__(self,dataFile,dim):
        self.dataFile = dataFile
        self.dDim = int(dim)#+1 #+1 is the added bias
        #print(self.dDim)
        if(self.dDim == irisD ): #this is the bcd delete the first row as its identifier
            #self.dataPd = pd.read_csv(self.dataFile,delimiter=';')
            self.dataPd = pd.read_csv(self.dataFile,delimiter=',',names=['0','1','2','3','4'])
            self.dataPd = self.dataPd.sample(frac=1).reset_index(drop=True)
            #print(self.dataPd)
            #exit()
        else:
            self.dataPd = pd.read_csv(self.dataFile,delimiter=',',names=['0','1','2'])
            global flagData
            flagData = 1
        #print(self.dataPd)
        #self.dataPd = pd.read_csv(self.dataFile,delimiter=',',header=None)
        self.W = np.asmatrix(np.ones(self.dDim)).H
        #self.logit(self.W,self.W) 


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
                #print(t[i])
            #print(phi)
            #print("shape",t.shape,"phi shape",phi.shape)
            #exit()
        else:        
            phi = self.dataPd[['0','1']].as_matrix()
            t = self.dataPd[['2']].as_matrix()

        dataLen = t.shape[0]

        self.blockLen = t.shape[0]/self.tDiv
        delLen = t.shape[0]%self.tDiv
        #print(delLen)

        #print(t)
        #exit()

        #exit()


        if( self.dDim == irisD): #delete 3 datapoints
            phiD = phi
            tD = t
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
        #print(self.Phi)


    def makeNetwork(self): #we are fixing the network size to 4x4 and 4x3 for iris and 2x2 2x2 for general data
        if self.dDim == irisD:
            node1Size = 4
            finSize = 3
        else:
            node1Size = 2
            finSize = 2

        
        self.node1 = np.zeros(node1Size)

        #self.layer1 = np.random.rand(node1Size+1,node1Size) #+1 is the bias
        self.layer1 = np.random.normal(0,1, [ node1Size+1,node1Size]) #+1 is the bias
        #self.layer1 = np.ones((node1Size+1,node1Size))
        self.layer1[node1Size].fill(0)
        self.layer1Delta = np.random.uniform(low = -1.0 ,high = 1.1, size=(   node1Size+1,node1Size)) #+1 is the bias

        self.node2 = np.zeros(node1Size)
        self.node2Err = np.zeros(node1Size)

        #self.layer2 = np.random.rand(node1Size+1,finSize) #+1 is the bias
        self.layer2 = np.random.normal(0,1,[node1Size+1,finSize]) #+1 is the bias
        #self.layer2 = np.ones((node1Size+1,finSize))#+1 is the bias
        self.layer2[node1Size].fill(0)
        self.layer2Delta = np.random.uniform(low=-1.0,high=1,size=(node1Size+1,finSize)) #+1 is the bias

        self.finNode = np.zeros(finSize) #one hot encoding
        self.finNodeErr = np.zeros(finSize) #one hot encoding

        #print(self.layer1.shape,self.node2.shape,self.layer2.shape,self.finNode.shape)

    def tanh(self,a):
        n = np.power(self.e,a) - np.power(self.e,-a)
        d = np.power(self.e,a) + np.power(self.e,-a)
        r = n/d
        return r
   
    def logit(self,a):
        n = 1
        d = 1 + np.power(self.e,-a)
        r = n/d
        return r

    def relu(self,a):
        return np.maximum(a,0)

    def applyGradient(self,n):
        #print(self.layer1Delta)
        #exit()
        self.layer1 = self.layer1 - n*self.layer1Delta
        self.layer2 = self.layer2 - n*self.layer2Delta
        #print(self.layer1)
        #print(layer)
        #print error

    def cleanGradient(self):
        self.layer2Delta.fill(0)
        self.layer1Delta.fill(0)

    def backwardPass(self,y):
        #error at output node
        tot = 0
        for i in range (self.finNode.shape[0]):
            tot += self.finNode[i]

        for i in range(self.finNode.shape[0]):
            if y == i:
                t = 1
                #t = tot
            else:
                t = 0
            self.finNodeErr[i] =  (self.finNode[i] - t)
           
        #print(self.T.shape[0])
        self.finNodeErr = self.finNodeErr/self.T.shape[0]
        #print(self.finNodeErr)
        
        error = 0
        for i in range (self.finNodeErr.shape[0]):
            error = abs(self.finNodeErr[i])

        #print("Err:",error)
        #update layer2 delta
        layer2DeltaTemp = np.zeros((self.node2.shape[0]+1,self.finNode.shape[0]))
        for i in range(self.finNode.shape[0]):
            for j in range(self.node2.shape[0]):
                layer2DeltaTemp[j][i] = self.finNodeErr[i]*self.node2[j]
            layer2DeltaTemp[self.node2.shape[0]][i] = self.finNodeErr[i] #updating weight for bias

        #print(self.layer2Delta.shape,layer2DeltaTemp.shape)
        self.layer2Delta += layer2DeltaTemp
        #print(self.layer2Delta,self.finNodeErr,self.node2,self.finNode,y)

        #print(self.layer2Delta)

        #calculate error at node2
        #node2ErrTemp = np.zero(self.node2.shape[0])
        for i in range(self.node2.shape[0]):
            self.node2Err[i] = 0
            for j in range(self.finNode.shape[0]):
                #self.node2Err[i] += self.layer2[i][j]*self.finNodeErr[j]  #relu
                self.node2Err[i] += self.node2[i]*(1 - self.node2[i])*self.layer2[i][j]*self.finNodeErr[j]
                if(self.node2[i] == 0):
                    self.node2Err[i] = 0
        #print(self.node2Err,self.layer2,self.finNodeErr)
        #exit()

        #calculate layer1 delta
        layer1DeltaTemp = np.zeros((self.node1.shape[0]+1,self.node2.shape[0]))
        for i in range(self.node2.shape[0]):
            for j in range(self.node1.shape[0]):
                layer1DeltaTemp[j][i] = self.node2Err[i]*self.node1[j]
            layer1DeltaTemp[self.node1.shape[0]][i] = self.node2Err[i] #updating weight for bias

        self.layer1Delta += layer1DeltaTemp




    def forwardPass(self,x):
        #print("randPoint",x)
        #add the biase term to x

        #print("layer1",self.layer1.shape[0],self.layer1.shape[1])
        self.node1 = x
        for i in range(self.node2.shape[0]):
            self.node2[i] = 0
            for j in range(x.shape[0]):
                self.node2[i] += self.layer1[j][i]*x[j]
            self.node2[i] += self.layer1[self.layer1.shape[0]-1][i]*1
            self.node2[i] = self.logit(self.node2[i])
            #self.node2[i] = self.relu(self.node2[i])

        #first layer computation done above start second layer now  
        #print("--layer 2--")
        for i in range(self.finNode.shape[0]):
            self.finNode[i] = 0
            for j in range(self.node2.shape[0]):
                self.finNode[i] += self.layer2[j][i]*self.node2[j]
            self.finNode[i] += self.layer2[self.layer2.shape[0]-1][i]*1

        #we will use the softmax function
        #print(self.node2)
        #print(self.finNode)
        self.softmaxOutput()
        #print(self.finNode)
        #exit()

        #we have the result in finNode

    def softFun(self,z):
        return np.power(self.e,z)

    def softmaxOutput(self):
        sumS = 0
        for i in range(self.finNode.shape[0]):
            sumS += self.softFun(self.finNode[i])

        for i in range(self.finNode.shape[0]):
            self.finNode[i] = self.softFun(self.finNode[i])/sumS

        #print(self.finNode)
        #exit()
    


    def Predict(self):
        maxI = 0
        maxV = -10000
        for i in range(self.finNode.shape[0]):
            if maxV < self.finNode[i]:
                maxV = self.finNode[i]
                maxI = i
    
        #print(maxV,self.finNode[0],self.finNode[1],self.finNode[2])
        return maxI

    def train(self):
        steps = 5000
        for i in range(steps):
            self.cleanGradient()
            for j in range(self.Phi.shape[0]):
                #randPoint = self.Phi[j]
                #print randPoint
                #print(j)
                self.forwardPass(self.Phi[j])
                self.backwardPass(self.T[j])
            self.applyGradient(0.01)
            #exit()
            if i%500 == 0:
                print(i)


    def predict(self,x,y):
        corr = 0
        tot = 0
        #print("____________________________")
        for i in range(x.shape[0]):
            self.forwardPass(x[i])
            yOut = self.Predict()
            #print(yOut,"vs",y[i])
            if yOut == y[i]:
                corr += 1
            tot += 1

        print("correct",corr,"tot",tot,"%age",(corr*100)/tot)
        #print(self.layer1)
        #print(self.layer2)

        #print()

    



def main():

    if len(sys.argv) != 3:
        print("Usage: ",sys.argv[0],"dataFile dimensions")
        exit()
    a = tensorFlowClassification(sys.argv[1],sys.argv[2])


    for i in range(5):
        a.makePhiT(i)
        a.makeNetwork()
        a.train()
        a.predict(a.PhiTest,a.TTest)
        #exit()
    #    W = a.train()
        #if ranD == 3:
        #    a.plot()
        #    exit()
    #we will plot the data



if __name__ == "__main__":main()
