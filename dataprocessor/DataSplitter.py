__author__ = 'sanbilp'

import numpy
import random

"""
Given a file (with one instance per line) and train,valid,test splits (in %, say 80,10,10)
this function will split the data and create train,test,valid files in same location.
"""
def DataSplitter(src, traincount, validcount, testcount):

    file = open(src,"r")
    totalcount=0
    for line in file:
        totalcount+=1
    file.close()

    print "total instances : "+str(totalcount)

    traincount = int(round(float(totalcount)*traincount/100))
    validcount = int(round(float(totalcount)*validcount/100))
    testcount = totalcount - validcount - traincount
    print "training instances : "+str(traincount)
    print "validation instances : "+str(validcount)
    print "testing instances : "+str(testcount)

    a = numpy.ones(traincount)
    b = numpy.ones(validcount)*2
    c = numpy.ones(testcount)*3
    index = numpy.concatenate((a,b,c))
    random.shuffle(index)

    trainfile = open(src.replace(".txt","-train.txt"),"w")
    validfile = open(src.replace(".txt","-valid.txt"),"w")
    testfile = open(src.replace(".txt","-test.txt"),"w")

    file = open(src,"r")

    count=0
    for line in file:
        if index[count]==1:
            trainfile.write(line)
        elif index[count]==2:
            validfile.write(line)
        else:
            testfile.write(line)
        count+=1

    file.close()
    trainfile.close()
    validfile.close()
    testfile.close()

