# Created by Ryan Fredrickson 
# This program implements a PCA to solve a cocktail party problem and out puts the results to a csv
import csv
file_name='sound.csv'
with open(file_name, 'rb') as f:  # opens PW file
    reader = csv.reader(f)
    data = list(list(rec) for rec in csv.reader(f, delimiter=',')) # reads csv into a list of lists
data = [[float(j) for j in i] for i in data]                        # convert data to floats
sumi = 0
sumj = 0
n = len(data)

for d in data:      # get sums in order to find averages
    sumi += d[0]
    sumj += d[1]
avgi = sumi/n       # finds average using
avgj = sumj/n

for d in data:      # make average 0 for both lists
    d[0] = d[0]-avgi
    d[1] = d[1]-avgj

weights = [1, 0]    # sets inital weights to 1 and 0
changeW = [0, 0]
rate=0.1            # sets learning rate

for d in data:      # adjust weights using calculation in textbook
    y = (weights[0] * d[0]) + (weights[1] * d[1])
    changeW[0] = rate*y*(d[0]-y*weights[0])
    changeW[1] = rate*y*(d[1]-y*weights[1])
    weights[0] += changeW[0]
    weights[1] += changeW[1]

result=[]           # results list

for d in data:      # calculate results and store in list
    templ=weights[0]*d[0]+weights[1]*d[1]
    result.append(templ)

with open("output.csv", "wb") as f:     # store results list in a csv
    writer = csv.writer(f)
    writer.writerow(result)

with open("readme.txt","w") as file:
    file.write("Final weights:\n")
    for item in weights:
        file.write("%s\n" % item)