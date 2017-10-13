import tensorflow as tf
import pandas as pd
import sys
import numpy as np
import tempfile

irisD = 4
randD = 3



training_data = 0
training_labels = 0
validation_data = 0
validation_labels = 0

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
            #self.dataPd = self.dataPd.sample(frac=1).reset_index(drop=True)
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
        global training_data
        global training_labels
        global validation_data
        global validation_labels
        training_data = self.Phi
        validation_data = self.PhiTest

        #print(self.T.shape[0])
        training_labels = np.zeros((120,3))
        validation_labels = np.zeros((30,3))
        for i in range(self.T.shape[0]):
            #print(self.T[i][0])
            training_labels[i][int(self.T[i][0])] = 1

        for i in range(self.TTest.shape[0]):
            validation_labels[i][int(self.TTest[i][0])] = 1
        #exit()
        #print(self.Phi)


if len(sys.argv) != 3:
    print("Usage: ",sys.argv[0],"dataFile dimensions")
    exit()
a = tensorFlowClassification(sys.argv[1],sys.argv[2])
a.makePhiT(0)




hidden_nodes = 4
num_labels = training_labels.shape[1]
num_features = training_data.shape[1]
print("labels: ",num_labels,num_features)
learning_rate = .01
reg_lambda = .01


# Weights and Bias Arrays, just like in Tensorflow
#layer1_weights_array = np.random.normal(0, 1, [num_features, hidden_nodes]) 
#layer2_weights_array = np.random.normal(0, 1, [hidden_nodes, num_labels]) 

layer1_weights_array = np.ones((num_features, hidden_nodes)) 
layer2_weights_array = np.ones((hidden_nodes, num_labels)) 

layer1_biases_array = np.zeros((1, hidden_nodes))
layer2_biases_array = np.zeros((1, num_labels))

def relu_activation(data_array):
    return np.maximum(data_array, 0)

def softmax(output_array):
    #print(output_array)
    logits_exp = np.exp(output_array)
    return logits_exp / np.sum(logits_exp, axis = 1, keepdims = True)

def cross_entropy_softmax_loss_array(softmax_probs_array, y_onehot):
    indices = np.argmax(y_onehot, axis = 1).astype(int)
    predicted_probability = softmax_probs_array[np.arange(len(softmax_probs_array)), indices]
    log_preds = np.log(predicted_probability)
    loss = -1.0 * np.sum(log_preds) / len(log_preds)
    return loss

def regularization_L2_softmax_loss(reg_lambda, weight1, weight2):
    weight1_loss = 0.5 * reg_lambda * np.sum(weight1 * weight1)
    weight2_loss = 0.5 * reg_lambda * np.sum(weight2 * weight2)
    return weight1_loss + weight2_loss

def accuracy(predictions, labels):
    preds_correct_boolean =  np.argmax(predictions, 1) == np.argmax(labels, 1)
    correct_predictions = np.sum(preds_correct_boolean)
    accuracy = 100.0 * correct_predictions / predictions.shape[0]
    return accuracy


for step in xrange(101):

    input_layer = np.dot(training_data, layer1_weights_array)
    hidden_layer = relu_activation(input_layer + layer1_biases_array)
    output_layer = np.dot(hidden_layer, layer2_weights_array) + layer2_biases_array
    output_probs = softmax(output_layer)

    #print(input_layer[0],output_probs[0])
    #exit()
    
    if step % 500 == 0:
        print 'Test accuracy: {0}%'.format(accuracy(output_probs,training_labels))
    #print(output_layer.shape)
    #exit()
    
    loss = cross_entropy_softmax_loss_array(output_probs, training_labels)
    loss += regularization_L2_softmax_loss(reg_lambda, layer1_weights_array, layer2_weights_array)

    output_error_signal = (output_probs - training_labels)  / output_probs.shape[0]
    #print(output_error_signal.shape)
    #exit()
    
    error_signal_hidden = np.dot(output_error_signal, layer2_weights_array.T) 
    error_signal_hidden[hidden_layer <= 0] = 0
   
    #print(error_signal_hidden.shape)
    #exit()

    #print(hidden_layer.T.shape,output_error_signal.shape)
    #exit()
    gradient_layer2_weights = np.dot(hidden_layer.T, output_error_signal)
    gradient_layer2_bias = np.sum(output_error_signal, axis = 0, keepdims = True)
    #print(gradient_layer2_weights.shape)
    #exit()
    
    gradient_layer1_weights = np.dot(training_data.T, error_signal_hidden)
    gradient_layer1_bias = np.sum(error_signal_hidden, axis = 0, keepdims = True)

    #gradient_layer2_weights += reg_lambda * layer2_weights_array
    #gradient_layer1_weights += reg_lambda * layer1_weights_array

    #print(error_signal_hidden[0],layer2_weights_array,layer2_biases_array,output_error_signal[0])
    #print("---")
    #print(gradient_layer1_weights)
    #print("-")
    #print(layer1_weights_array)

    layer1_weights_array -= learning_rate * gradient_layer1_weights
    layer1_biases_array -= learning_rate * gradient_layer1_bias
    layer2_weights_array -= learning_rate * gradient_layer2_weights
    layer2_biases_array -= learning_rate * gradient_layer2_bias

    #layer1_biases_array.fill(0)
    #layer2_biases_array.fill(0)

    #exit()


    if step % 500 == 0:
            print 'Loss at step {0}: {1}'.format(step, loss)

    if step == 100:
        layer1_weights_array = ([[ 0.86235516,  0.86235516,  0.86235516,  0.86235516],
                 [ 0.93176436,  0.93176436,  0.93176436,  0.93176436],
                  [ 0.90451874,  0.90451874,  0.90451874,  0.90451874],
                   [ 0.96898457,  0.96898457,  0.96898457,  0.96898457]])


        layer2_weights_array = ([[ 0.99327473 , 1.02358796,  0.98313731],
                 [ 0.99327473 , 1.02358796,  0.98313731],
                  [ 0.99327473 , 1.02358796 , 0.98313731],
                   [ 0.99327473 , 1.02358796 , 0.98313731]])


        layer1_biases_array = ([[-0.02301551, -0.02301551, -0.02301551, -0.02301551]])
        layer2_biases_array = ([[ 0.04676023,  0.01934977, -0.06611]])
        
        input_layer = np.dot(training_data, layer1_weights_array)
        hidden_layer = relu_activation(input_layer + layer1_biases_array)
        output_layer = np.dot(hidden_layer, layer2_weights_array) + layer2_biases_array
        output_probs = softmax(output_layer)

        print(hidden_layer)
        print("---")
        print(output_layer)
        print("---")
        print(output_probs)


        print(layer1_weights_array)
        print(layer1_biases_array)
        print("--")
        print(layer2_weights_array)
        print(layer2_biases_array)
        #print 'Test accuracy: {0}%'.format(accuracy(output_probs,training_labels))
        #print(output_probs)


 

        output_error_signal = (output_probs - training_labels)  / output_probs.shape[0]
    
