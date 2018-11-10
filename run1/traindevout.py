import fileinput
import re
import json
import math
#takes in output from train.py and outputs vectors of the loss and accuracy of train and dev

f=open('CNN1output.txt', 'rU')
f2=open('trainlossacc.txt','w')

evaltf=0
trainloss=[]
trainacc=[]
devloss=[]
devacc=[]

trainlsum=0
trainasum=0
def writelist (list1, fn):
    for item in list1:
        fn.write("%s, " % item)
    fn.write('\n')

for line in f:
    if evaltf==1:
        #print(line.split()[4][:-1])
        devloss.append(line.split()[4][:-1])
        devacc.append(line.split()[6])
        evaltf=0

    if re.match('Evaluation', line):
        #print(line)
        trainloss.append(trainlsum)
        trainacc.append(trainasum/float(100)) # accuracy of 100 trainings averaged
        evaltf=1
        trainlsum=0
        trainasum=0
        
    if re.match('2018', line) and evaltf==0:
        trainlsum+=float(line.split()[4][:-1])
        trainasum+=float(line.split()[6])


#f2.write('train loss: ')
writelist(trainloss,f2)
#f2.write('train accuracy: ')
writelist(trainacc,f2)

#f2.write('dev loss: ')
writelist(devloss,f2)
#f2.write('dev accuracy: ')
writelist(devacc,f2)

f.close()
f2.close()

