import tensorflow as tf
import pandas as pd
import sys
import numpy as np
import tempfile

irisD = 4
randD = 3

tf.logging.set_verbosity(tf.logging.INFO)
flagData = 0

def cnn_model_fn(features,labels,mode):
    """"Model function for detectin wine"""
    global flagData
    if flagData == 0:
        input_layer = tf.reshape(features["x"],[-1,1,4])
        hidden1 = tf.layers.dense(inputs=input_layer,units=4,activation=tf.nn.relu)
        #hidden2 = tf.layers.dense(inputs=hidden1,units=4,activation=tf.nn.relu)
        logits = tf.layers.dense(inputs=hidden1,units=3)
    else:
        
        pass

    predictions = {
            "classes": tf.argmax(input=logits,axis=2),
            "probabilities": tf.nn.softmax(logits,name="softmax_tensor")
            }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode,predictions=predictions)

    if flagData == 0:
        depth = 3
    else:
        depth = 2


    onehot_labels = tf.one_hot(indices=tf.cast(labels,tf.int32),depth=depth)
    print("logits",logits,"onehot_label",onehot_labels)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels,logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
                loss = loss,
                global_step = tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode,loss=loss,train_op=train_op)


    print("labels",labels,"predictions",predictions["classes"])
    eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(
                #labels=labels,predictions=labels)
                labels=labels,predictions=predictions["classes"])
            }

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


class tensorFlowClassification:
    tDiv = 5

    def __init__(self,dataFile,dim):
        self.dataFile = dataFile
        self.dDim = int(dim)#+1 #+1 is the added bias
        #print(self.dDim)
        if(self.dDim == irisD ): #this is the bcd delete the first row as its identifier
            #self.dataPd = pd.read_csv(self.dataFile,delimiter=';')
            self.dataPd = pd.read_csv(self.dataFile,delimiter=',',names=['0','1','2','3','4'])
            self.dataPd = self.dataPd.sample(frac=1).reset_index(drop=True)
        else:
            self.dataPd = pd.read_csv(self.dataFile,delimiter=',',names=['0','1','2'])
            global flagData
            flagData = 1
        #print(self.dataPd)
        #self.dataPd = pd.read_csv(self.dataFile,delimiter=',',header=None)
        self.W = np.asmatrix(np.ones(self.dDim)).H
        #self.logit(self.W,self.W) 

    def train(self):

        #tf.device('/gpu:0')

        wineClassifier = tf.estimator.Estimator(model_fn=cnn_model_fn,model_dir='/tmp/')
        
        tensors_to_log = {"probabilities":"softmax_tensor"}
        #logging_hook = tf.train.LoggingTensorHook(tensors = tensors_to_log,every_n_iter=50)


        train_input_fn = tf.estimator.inputs.numpy_input_fn(
                x = {"x":self.Phi},
                y=self.T,
                batch_size = 10,
                num_epochs=None,
                shuffle=True)

        wineClassifier.train(input_fn=train_input_fn,steps=20000)
        #wineClassifier.train(input_fn=train_input_fn,steps=20000,hooks=[logging_hook])

        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={"x": self.PhiTest},
                y=self.TTest,
                num_epochs=1,
                shuffle=False)

        eval_results = wineClassifier.evaluate(input_fn=eval_input_fn)
        print("Sun result:",eval_results)

        predict = wineClassifier.predict(input_fn=eval_input_fn)
        #print(predict)
        #k = 0
        #for i in predict:
        #    print (i["classes"],"vs",self.TTest[k])
        #    k = k+1



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
                    t1[i] = int(3)
                print(t[i])
            #print(phi)
            #print("shape",t.shape,"phi shape",phi.shape)
            #exit()
        else:        
            phi = self.dataPd[['0','1']].as_matrix()
            t = self.dataPd[['2']].as_matrix()

        dataLen = t.shape[0]
        #print(t.shape[0])
        #divide into 5 parts

        #for i in range(t.shape[0]):
        #    if t[i] == 2:
        #        t[i] = 0
        #    elif t[i] == 4:
        #        t[i] = 1
        #    else:
        #        pass

        #print(t)
        #exit()
        #print(phi)

        self.blockLen = t.shape[0]/self.tDiv
        delLen = t.shape[0]%self.tDiv
        print(delLen)

        #exit()


        if( self.dDim == irisD): #delete 3 datapoints
            #phiD = np.delete(phi,[1,2,3],axis=0)
            #phiD = np.delete(phi,[1,2,3,4],axis=0)
            #tD = np.delete(t,[1,2,3],axis=0)
            #tD = np.delete(t,[1,2,3,4],axis=0)
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
        #print(a_2d.shape, self.Phi.shape)

        #self.Phi = np.concatenate((a_2d,self.Phi),axis=1)
        #self.PhiTest = np.concatenate((a_2dTest,self.PhiTest),axis=1)
        print(self.Phi)

        #print(self.Phi)
        #self.Phi = np.insert(self.Phi,1,999)
        #print(self.Phi)

        #sunny over here we are doing phi
        #self.T = np.asmatrix(self.T)
        #self.Phi= np.asmatrix(self.Phi)
        #self.PhiTest = np.asmatrix(self.PhiTest)
        #self.TTest = np.asmatrix(self.TTest)
        
        #print(self.PhiTest)

        #self.deltaEW(self.W)




def main():

    if len(sys.argv) != 3:
        print("Usage: ",sys.argv[0],"dataFile dimensions")
        exit()
    a = tensorFlowClassification(sys.argv[1],sys.argv[2])


    for i in range(5):
        a.makePhiT(i)
        a.train()
    #    W = a.train()
        #if ranD == 3:
        #    a.plot()
        #    exit()
    #we will plot the data



if __name__ == "__main__":main()
