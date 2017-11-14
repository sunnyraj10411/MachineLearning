import pandas as pd
import sys
import numpy as np
import tempfile
import scipy as sp
import matplotlib.pyplot as plt
import numbers
from matplotlib.patches import Ellipse
from itertools import cycle
from scipy import linalg


irisD = 4
randD = 3

flagData = 0

class MixtureModelClassification:
    tDiv = 5
    e = 2.71828

    def __init__(self,dataFile,dim,clus):
        self.dataFile = dataFile
        self.dDim = int(dim) #+1 #+1 is the added bias
        if(self.dDim == irisD ): #this is the bcd delete the first row as its identifier
            self.dataPd = pd.read_csv(self.dataFile,delimiter=',',names=['0','1','2','3','4'])
            self.dataPd = self.dataPd.sample(frac=1).reset_index(drop=True)
        else:
            self.dataPd = pd.read_csv(self.dataFile,delimiter=',',names=['0','1','2'])
            global flagData
            flagData = 1
        self.W = np.asmatrix(np.ones(self.dDim)).H
        self.clusters = int(clus)


    def makePhiT(self,div):
        print(self.dDim)
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
        
        elif self.dDim == 2:        
            phi = self.dataPd[['0','1']].as_matrix()
            t1 = self.dataPd[['2']].as_matrix()
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


    def randomInit(self):
        #find max min will assume to be 0

        maxI = np.zeros(self.Phi.shape[1])
        for i in range(self.Phi.shape[0]):
            for j in range(self.Phi.shape[1]):
                if( self.Phi[i][j] > maxI[j]):
                    maxI[j] = self.Phi[i][j]

        randVal = np.random.rand(self.clusters,self.Phi.shape[1])

        for i in range(self.clusters):
            for j in range(self.Phi.shape[1]):
                randVal[i][j] = randVal[i][j]*maxI[j]

        #print(maxI)
        #print(randVal)

        return randVal

    
    def findDis(self,x,y):
        tot = 0
        for i in range(x.shape[0]):
            tot = np.power(x[i]-y[i],2)

        tot = np.sqrt(tot)
        return tot


    def calRVal(self,mean):

        r = np.zeros([self.Phi.shape[0],self.clusters])
        for i in range(self.Phi.shape[0]):
            minDis = 999999
            minIndex = 100
            for j in range(self.clusters):
                disTemp = self.findDis(mean[j],self.Phi[i])
                if  disTemp < minDis:
                    minDis = disTemp
                    minIndex = j

            r[i][minIndex] = 1
        return r

    # get the ellipse
    def getElipse(self, standardDevia):
        ellip = []
        for k in range(self.clusters):
            ellip.append((self.mean[k], self.get_ellipse(standardDevia,k)))
        return ellip

    def get_ellipse(self, standardDevia, k):
        eigenvalues, eigenvectors = sp.linalg.eigh(self.covar[k])
        eigenvalues, eigenvectors = sp.linalg.eigh(self.covar[k])
        angle = np.arctan2(*eigenvectors[:, 0][::-1])
        width, height = standardDevia * np.sqrt(eigenvalues)
        return angle, width, height


    def calNewMean(self,r):
        mean = np.zeros([self.clusters,self.Phi.shape[1]])
        count = np.zeros(self.clusters)

        for i in range(self.clusters):
            for j in range(self.Phi.shape[1]):
                for k in range(r.shape[0]):
                        mean[i][j] += r[k][i]*self.Phi[k][j]
                        if r[k][i] == 1 and j == 0:
                            count[i] += 1

        #print(count)

        for i in range(self.clusters):
            for j in range(self.Phi.shape[1]):
                mean[i][j] = mean[i][j]/float(count[i])

       
        #print(mean)
        return mean
    
    def saveLabel(self,r):
        changeFlag = 0
        for i in range(r.shape[0]):
            label = 0
            for j in range(r.shape[1]):
               if r[i][j] == 1:
                   label = j

            if( self.labelStore[i][0] != label):
                changeFlag = 1
                self.labelStore[i][0] = label 

        return changeFlag


    def printLabelsK(self):
        for i in range(self.labelStore.shape[0]):
            if self.T[i] == 0:
                label = "iris setosa"
            elif self.T[i] == 1:
                label = "iris versicolor"
            else:
                label = "iris  verginica"

            print(self.labelStore[i][0]," for ",label)



    def KMeans(self):
        #initialize a random 4 dimensional point
        mean = self.randomInit()
        self.labelStore = np.zeros([self.Phi.shape[0],2])
        
        i = 0
        nochange = 1
        itrNum = 300

        while i < itrNum or (i >= itrNum and nochange == 1):
            r = self.calRVal(mean)
            #print("Gamma",i)
            mean = self.calNewMean(r)
            nochange = self.saveLabel(r)
            #print("change ",nochange)

            #nochange = 1
            i += 1
        self.kMean = mean
        self.kR = r
        print(mean)


    def initializeCovar(self):
    
        covMat = np.zeros([self.clusters,self.Phi.shape[1],self.Phi.shape[1]])
        for i in range(self.clusters):
            value = []
            for j in range(self.Phi.shape[0]):
                if self.kR[j][i] == 1:
                    value.append(self.Phi[j])

            npVal = np.array(value)
            npVal = npVal.T

            #print("npVal",npVal)
            covMat[i] = np.cov(npVal)

        print("covMat1")
        print(covMat)
        return covMat

    
    def gaussian(self,mean,covar,x):
        x_mean = x - mean;
        x_mean_T = x_mean.T;
        covarInv = np.linalg.inv(covar)
        #print(covarInv)

        expVal = np.dot(x_mean_T,covarInv)
        expVal = np.dot(expVal,x_mean)
        expVal = -1*(0.5)*expVal


        denoVal = 2*np.pi*np.linalg.det(covar)
        denoVal = np.sqrt(denoVal)

        numVal = np.exp(expVal)

        #print("n",numVal,"d",denoVal)
        return numVal/denoVal


    def getLogLikelyhood(self):
        LogLikelyhood = 0.
        tol = 0.
        for i in range(self.Phi.shape[0]):
            for j in range(self.clusters):
                tol += self.pik[j]*self.gaussian(self.mean[j],self.covar[j],self.Phi[i])
            LogLikelyhood += np.log(tol)    
        #print (LogLikelyhood)
        return LogLikelyhood

                        
    def getGamma(self):
        gammaZ = np.zeros([self.Phi.shape[0],self.clusters])

        for i in range(self.Phi.shape[0]):
            tot = 0 
            for j in range(self.clusters):
                tot += self.pik[j]*self.gaussian(self.mean[j],self.covar[j],self.Phi[i])

            for j in range(self.clusters):
                gammaZ[i][j] = self.pik[j]*self.gaussian(self.mean[j],self.covar[j],self.Phi[i])/tot

        
        #print(gammaZ)
        #exit()
        return gammaZ


    def getGammaNk(self):
        nk = np.zeros(self.clusters)

        for i in range(self.Phi.shape[0]):
            for j in range(self.clusters):
                nk[j] += self.gammaZ[i][j]

        #print(nk)
        #exit()
        return nk



    def getGammaMean(self):
        mean = np.zeros([self.clusters,self.Phi.shape[1]]) 
        for i in range(self.Phi.shape[0]):
            for k in range(self.clusters):
                mean[k] += self.gammaZ[i][k]*self.Phi[i]


        for i in range(self.clusters):
            mean[i] /= self.Nk[i]

        
        return mean

         
    def getGammaCovar(self):
        covMat = np.zeros([self.clusters,self.Phi.shape[1],self.Phi.shape[1]])

        for  i in range(self.Phi.shape[0]):
            for j in range(self.clusters):
                x_mean = self.Phi[i] - self.mean[j]
                x_mean = np.reshape(x_mean,(x_mean.shape[0],1))
                x_mean_t = np.reshape(x_mean,(1,x_mean.shape[0]))
                #print(x_mean_t.shape,x_mean.shape)

                covMat[j] += self.gammaZ[i][j]*np.matmul(x_mean,x_mean_t)
                
                #print(x_mean.shape,x_mean.T.shape)
                #print("Matmul ",np.matmul(x_mean,x_mean_t))
                #exit()

        for i in range(self.clusters):
            covMat[i] /= self.Nk[i]

        return covMat
        

    def mixtureModel(self):
        self.covar = self.initializeCovar()
        self.pik = np.ones(self.clusters)/float(3)
        self.mean = self.kMean
        self.Nk = np.zeros(self.clusters)
        iterRange = 10000
        logLikely = []

        for i in range(iterRange):

            #E step
            self.gammaZ = self.getGamma()

            #M step
            self.Nk = self.getGammaNk()
            self.mean = self.getGammaMean()
            self.covar = self.getGammaCovar()
            self.pik =  self.Nk/self.Phi.shape[0]

            #check for convergence 
            logLikely.append(self.getLogLikelyhood())
            if i >= 50:
                if logLikely[i] == logLikely[i-1]:
                    print('-----------------------------------------------')
                    print("After %d iterations, EM converges" %(i) )
                    print('-----------------------------------------------')
                    break


        print("mean2")
        print(self.mean)
        
        print("covMat2")
        print(self.covar)

        print("pik2")
        print(self.pik)
            


        

def main():
    if len(sys.argv) != 4:
        print("Usage: ",sys.argv[0],"dataFile dimensions expectedCluster")
        exit()
    a = MixtureModelClassification(sys.argv[1],sys.argv[2],sys.argv[3])
    a.makePhiT(1)
    a.KMeans()

    #a.printLabelsK()
    a.mixtureModel()

    if int(sys.argv[2]) == 2: # we will be plotting

        # set the plot params
        plt.figure(figsize=(6,6))
        plt.subplot(1, 1, 1)
        plt.xlim((1, 7))
        plt.ylim((-0., 2.7))
        colors = cycle(["black", "green", "purple"])

        for mean, (angle, width, height) in a.getElipse(3):
            ellipses = Ellipse(xy=mean, width=width, height=height,
                      angle=np.degrees(angle))
            ellipses.set_alpha(0.3)
            ellipses.set_color(next(colors))
            plt.gca().add_artist(ellipses)


        #print(a.T)

        for j in range(3):
            a0x = []
            a0y = []
            for i in range(a.Phi.shape[0]):
                if a.T[i] == j:
                    a0x.append(a.Phi[i][0])
                    a0y.append(a.Phi[i][1])
                    print(i)
            plt.scatter(a0x,a0y)

        plt.show()


if __name__ == "__main__":main()
