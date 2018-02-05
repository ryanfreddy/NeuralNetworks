import numpy as np
#Global Variables
b=1.0                                               #Bias value
t=0.0                                               #Theta value
e=True                                              #Error value
lrate=1.0                                           #Learnign rate value

def part1():                                        #Function asks for length and width and spits out whether or not it is Sentosa
    l = long(input("Length(mm): "))                 #Ask for Input
    w = long(input("Width(mm): "))
    widthw=4                                        #Set weights for linear seperator
    lengthw=-1
    biasi=-10
    t=lengthw*l+widthw*w+biasi                      #calulates theta value
    if (t>0):                                       #depending on theta value spits out yes or no yes indicating it is Sentosa
        print("No")
    else:
        print("Yes")

def part2(w):                                       #Takes in random weights and send them to the training method
    file=open('train.txt','r')
    for line in file:                               #For each line in the file split into
        a1,a2,a3,a4,a5=line.split(',')              #Split line with comma as delimiter
        if (a5 =="Iris-setosa\n"):                  #If Iris setosa then set true
            cate=1
        else:
            cate=0
        a1 = float(a1)                              #make sure values brought in are floats
        a2 = float(a2)
        a3 = float(a3)
        a4 = float(a4)
        tempa = np.array([a1,a2,a3,a4,cate])        #create temp array to store variables
        w=trainforward(w,tempa)                     #pass values from file and weights into trining method
    return w                                        #Return weight once they have been altered

def trainforward(wghts,det):                        #Forward training method that takes training data in(det) and weights and spits
    global e                                        #allowing us to modify the global variable error withich lets us now hether there is an error
    e=False
    act = (wghts[0]*det[0])+(wghts[1]*det[1])+(wghts[2]*det[2])+(wghts[3]*det[3])+(wghts[4]*b)  #calculate activation
    if act>t:                                       #If activation is greater then theta then set
        a=1
    else:
        a=0
    if (a-det[4])<0:                                #If neural network not categorizing properly give error and fix weights
        e=True
        wghts[0] += lrate * det[0]                  #Change weights as a factor of lrate and data
        wghts[1] += lrate * det[1]
        wghts[2] += lrate * det[2]
        wghts[3] += lrate * det[3]
        wghts[4] += lrate * b
    elif (a - det[4]) > 0:
        e = True
        wghts[0] -= lrate * det[0]                  #Change weights as a factor of lrate and data
        wghts[1] -= lrate * det[1]
        wghts[2] -= lrate * det[2]
        wghts[3] -= lrate * det[3]
        wghts[4] -= lrate * b
    return wghts


def testinga(wghts):                                #This method test the neural network from part2
    file = open('test.txt', 'r')
    numerrors=0
    for line in file:                               #For each line in the file split into
        a1, a2, a3, a4, a5 = line.split(',')        #Split line with comma as delimiter
        if a5 == 'Iris-setosa\n':                   #If Iris setosa then set true
            tst = 1
        else:
            tst = 0

        a1 = float(a1)                              #Make sure values brought in are floats
        a2 = float(a2)
        a3 = float(a3)
        a4 = float(a4)

        temp = np.array([a1, a2, a3, a4, tst])      #Create temp array to store variables
        testvalue = (wghts[0] * temp[0]) + (wghts[1] * temp[1]) + (wghts[2] * temp[2]) + (wghts[3] * temp[3]) + (wghts[4] * b)      #creates value to test agains theta
        if testvalue > t:                           #Tests value against theta
            g = 1
        else:
            g = 0
        if (g - temp[4]) != 0:                      #Checks to see if value is categorized correctly
            numerrors +=1
        return numerrors

def part4(w):
    file=open('train.txt','r')
    for line in file:                               #For each line in the file split into
        a1,a2,a3,a4,a5=line.split(',')              #Split line with comma as delimiter
        if (a5 =="Iris-virginica\n"):               #If Iris virginica then set true
            cate=1
        else:
            cate=0
        a1 = float(a1)                              #Make sure values brought in are floats
        a2 = float(a2)
        a3 = float(a3)
        a4 = float(a4)
        tempa = np.array([a1,a2,a3,a4,cate])        #Create temp array to store variables
        w=errcorrection(w,tempa)                    #pass values from file and weights into trining method
    return w                                        #Return trained weight

def errcorrection(wghts,det):
    act = (wghts[0]*det[0])+(wghts[1]*det[1])+(wghts[2]*det[2])+(wghts[3]*det[3])+(wghts[4]*b) #calculate activation
    if act>t:                                       #If activation is greater then theta then set
        j=1
    else:
        j=0
    wghts[0] += (det[4]-j) * lrate/2 * det[0]       #Correct errors
    wghts[1] += (det[4]-j) * lrate/2 * det[1]
    wghts[2] += (det[4]-j) * lrate/2 * det[2]
    wghts[3] += (det[4]-j) * lrate/2 * det[3]
    wghts[4] += (det[4]-j) * lrate/2 * b
    return wghts                                    #Return new weights

def testingb(wghts):
    file = open('test.txt', 'r')
    numerrors=0                                     #set num of errors to 0
    for line in file:                               #For each line in the file split into
        a1, a2, a3, a4, a5 = line.split(',')        #Split line with comma as delimiter
        if a5 == 'Iris-virginica\n':                #If Iris virginica then set true
            tst = 1
        else:
            tst = 0

        a1 = float(a1)                              #Make sure values brought in are floats
        a2 = float(a2)
        a3 = float(a3)
        a4 = float(a4)

        temp = np.array([a1, a2, a3, a4, tst])      #create temp array for test data
        testvalue = (wghts[0] * temp[0]) + (wghts[1] * temp[1]) + (wghts[2] * temp[2]) + (wghts[3] * temp[3]) + (wghts[4] * b)

        if testvalue > 0:
            g = 1
        else:
            g = 0
        if (g - temp[4]) != 0:
            numerrors +=1
        return numerrors
def tofile(wghts1,wghts2):
    ofile=open('Results.txt','w')                   #Produce file for outputting too
    ifile=open('test.txt','r')
    for line in ifile:                              #For each line in the file split into
        a1, a2, a3, a4, a5 = line.split(',')        #Split line with comma as delimiter
        if a5 == 'Iris-virginica\n':                #If Iris virginica then set true
            tst = 1
        else:
            tst = 0
    a1 = float(a1)                                  #Make sure values brought in are floats
    a2 = float(a2)
    a3 = float(a3)
    a4 = float(a4)

    temp = np.array([a1, a2, a3, a4, tst])          #create temp array for data
    testvalue = (wghts1[0] * temp[0]) + (wghts1[1] * temp[1]) + (wghts1[2] * temp[2]) + (wghts1[3] * temp[3]) + (wghts1[4] * b)     #Create value for testing
    if testvalue > 0:                               #Test to see if value is greater than theta if so print out that it is Setosa
        line1 = str(temp[0]) + ',' + str(temp[1]) + ',' + str(temp[2]) + ',' + str(temp[3]) + " Iris-setosa\n"
        ofile.write(line1)

    else:                                           # else if not setosa then see if it is one of the other two values and print out which category it is in
        testvalue2 = (wghts2[0] * temp[0]) + (wghts2[1] * temp[1]) + (wghts2[2] * temp[2]) + (wghts2[3] * temp[3]) + (wghts2[4] * b)
        if testvalue2 > 0:
            line2 = str(temp[0]) + ',' + str(temp[1]) + ',' + str(temp[2]) + ',' + str(temp[3]) + " Iris-vignica\n"
            ofile.write(line2)
        else:
            line3 = str(temp[0]) + ',' + str(temp[1]) + ',' + str(temp[2]) + ',' + str(temp[3]) + " Iris-versicolor\n"
            ofile.write(line3)
    ofile.close()                                   #close output file
    print ("Output done")

def execute():                                      #Method to test all previously created methods
    global e                                        #So you can modify global variable errors
    wghts1 = np.array([0.]*5)                       #create intial array for weights
    numerrors=30                                    #number of errors initially so program will run at least once
    part1()                                         #run part1
    while e or numerrors>0:                         #while there are errors keep modifying data
         wghts1=part2(wghts1)                       #Train part2 nueral network
         numerrors=testinga(wghts1)                 #test part2 neural network
    print('Number of errors testing part2: ', numerrors)        #Print out number of errors
    print('The weights for part2',wghts1)           #Print otu weights produced by part2
    print("Linear separator equation for part 2: "," (",wghts1[4],")0x + (",wghts1[0],")1x + (",wghts1[1],")2x + (",wghts1[2],")3x + (",wghts1[3],")4x = 0")
    #Print out equation
    j = 0
    wghts2=np.array([0.]*5)                         #Set initial values for weight array
    minerrors=30
    bestweights=np.array
    while j < (1500):
         wghts2 = part4(wghts2)                     #run part4 to trin neural network
         j += 1
         numerrors=testingb(wghts2)                 #find amount of errors
         if(minerrors>numerrors):
             minerrors=numerrors
             bestweights=wghts2
    print("Number of errors testing part4: ",minerrors)     #output amount of errors
    wghts2=bestweights
    print("The weights for part4",wghts2)           #print out weights for part4 neural network
    print("Linear separator equation for part 2: "," (",wghts2[4],")0x + (",wghts2[0],")1x + (",wghts2[1],")2x + (",wghts2[2],")3x + (",wghts2[3],")4x = 0") #output equation
    tofile(wghts1, wghts2)                          #run tofile method

execute()                                           #execute code