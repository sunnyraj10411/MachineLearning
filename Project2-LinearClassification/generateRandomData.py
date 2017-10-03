import numpy as np
import sys
import matplotlib.pyplot as plt


dataPoints = 200
numCluster = 2


def main():
    
    mean = [50,50]
    cov = [[20,0],[0,20]]
    x,y = np.random.multivariate_normal(mean,cov,200).T
    
    mean1 = [10,10]
    cov1 = [[20,0],[0,20]]
    x1,y1 = np.random.multivariate_normal(mean1,cov1,200).T

    with open("rand.dat","w") as f:
        for i in range(len(x)):
            w = str(x[i])+','+str(y[i])+','+str(1)+'\n'
            f.write(w)
            w = str(x1[i])+','+str(y1[i])+','+str(2)+'\n'
            f.write(w)


    #plt.plot(x,y,'o')
    #plt.plot(x1,y1,'o')
    #plt.axis('equal')
    #plt.show()



if __name__ == "__main__":main()
