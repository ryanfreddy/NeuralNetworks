import numpy as np
import math
import random
#Created by Ryan Fredrickson
#back propogation network to predict numbers

def importdata(filename):#imports data into list of lists
    results = []
    with open(filename) as inputfile:
        for line in inputfile:
            results.append(line.strip().split(','))
    return results
traininga=importdata("training.txt")
testinga=importdata("testing.txt")

def dsig(x):#returns result of sigmoid function
    return((-math.e)**(-x))/((1+(math.e**(-x)))**2)

def sig(x):#returns result of derivative of sigmoid function
    return 1/(1+(math.e**(-x)))

def Mtrx(K,L,filled=0.0):#creates matrix for weights
    matrix=[]
    for k in range(K):
        matrix.append([filled]*L)
    return matrix




class Neuralnet:
    def __init__(self,inputnodes,outputnodes,hiddennodes):
        #number of nodes of each type
        self.inputnodes=inputnodes+1    #Add one for bias node
        self.hiddennodes=hiddennodes
        self.outputnodes=outputnodes
        #weights
        self.weightsin=Mtrx(self.inputnodes,self.hiddennodes)
        self.weightsout = Mtrx(self.hiddennodes, self.outputnodes)
        for row in range(self.hiddennodes):#fill output weights matrix between-1 and 1
            for col in range(self.outputnodes):
                self.weightsout[row][col] = random.uniform(-1.0, 1.0)
        for row in range(self.inputnodes):#fill input weights matrix between -1 and 1
            for col in range(self.hiddennodes):
                self.weightsin[row][col] = random.uniform(-1.0, 1.0)
        #activations
        self.actoutput = [1.0] * self.outputnodes
        self.actinput=[1.0]*self.inputnodes
        self.acthidden=[1.0]*self.hiddennodes
        #matrix for momentum
        self.momeninput = Mtrx(self.inputnodes, self.hiddennodes)
        self.momenoutput = Mtrx(self.hiddennodes, self.outputnodes)


    def Updt(self,data):
        for j in range(self.inputnodes-1):#input activation
            self.actinput[j]=data[j]
        for j in range(self.hiddennodes):#hidden layer activation
            s=0.0
            for k in range(self.inputnodes):
                s+=self.weightsin[k][j]*self.actinput[k]
            self.acthidden[j]=sig(s)
        for j in range(self.outputnodes):#output layer activation
            s=0.0
            for k in range(self.hiddennodes):
                s+=self.weightsout[k][j]*self.acthidden
            self.actoutput[j]=sig(s)


    def backprop(self,expected,learnrate,momen):
        odelta = [0.0] * self.outputnodes
        for b in range(self.outputnodes):
            e = expected[b] - self.actoutput[b]
            odelta[b] = dsig(self.actoutput[b]) * e
        # error for hidden
        hdelta = [0.0] * self.hiddennodes
        for j in range(self.hiddennodes):
            e = 0.0
            for k in range(self.outputnodes):
                e += odelta[k] * self.weightsout[j][k]
            hdelta[j] = dsig(self.acthidden[j]) * e
        # output weights update
        for j in range(self.hiddennodes):
            for k in range(self.outputnodes):
                change = odelta[k] * self.acthidden[j]
                self.weightsout[j][k] = self.weightsout[j][k] + learnrate * change + momen * self.momenoutput[j][k]
                self.momenoutput[j][k] = change
        # input weights update
        for i in range(self.inputnodes):
            for j in range(self.hiddennodes):
                change = hdelta[j] * self.actinput[i]
                self.weightsin[i][j] = self.weightsin[i][j] + learnrate * change + momen * self.momeninput[i][j]
                self.momeninput[i][j] = change
        # error calculations
        e = 0.0
        for k in range(len(expected)):
            e += 0.5 * (expected[k] - self.actoutput[k]) ** 2
        return e
    def train(self, patt, iter=100, lrate=1, mom=0.3):
        for j in range(iter):
            e = 0.0

            for d in patt:
                inputs = d[:64]
                inputs=[int(j)for j in inputs]
                targets = d[1]

                self.Updt(inputs)
                e += self.backprop(targets, lrate, mom)

            if j % 100 == 0:
                print('error %-.5f' % e)

    def testing(self, pat):
        for a in pat:

            print(a[0], '->', self.Updt(a[0]))

    def printweights(self):
        print("Input weights:")

        for j in range(self.inputnodes):
            print(self.weightsin[j])

        print()

        print("Output weights:")
        
        for j in range(self.hiddennodes):
            print(self.weightsout[j])


nnn = Neuralnet(64, 10, 64)
# train it with some patterns
nnn.train(traininga)
# test it
nnn.test(testinga)
nnn.printweights()