__author__ = 'sanbilp'

import numpy

from pyNN.util.NNUtil import create_folder

def GenerateSeqMat(src, folder, batch_size, dtype):

    create_folder(folder)
    ipfile = open(folder+"ip.txt","w")

    file = open(src,"r")
    slength = len(file.readline().strip().split())
    file.close()

    mat = numpy.zeros((batch_size,slength),dtype=dtype)
    maskmat = numpy.zeros((batch_size,slength),dtype=dtype)
    count = 0
    file = open(src,"r")
    for line in file:
        line = line.strip().split()
        linelen = len(line)
        for i in range(0,linelen):
            mat[count%batch_size][i] = int(line[i])
            maskmat[count%batch_size][i] = 1
        count+=1
        if(count%batch_size == 0):
            name = folder+str(count/batch_size)
            numpy.save(name,mat)
            numpy.save(name+"_mask",maskmat)
            ipfile.write(name+","+str(batch_size)+"\n")
            mat = numpy.zeros((batch_size,linelen), dtype=dtype)
            maskmat = numpy.zeros((batch_size,linelen),dtype=dtype)

    if count%batch_size!=0:
        name = folder.str((count/batch_size)+1)
        numpy.save(name, mat)
        numpy.save(name+"_mask", maskmat)
        ipfile.write(name+","+str(count%batch_size)+"\n")
    file.close()
    ipfile.close()


def GenerateSeqMatTVT(folder, batch_size, dtype):

    create_folder(folder+"mat_pic/")
    GenerateSeqMat(folder+"strain-seq.txt",folder+"mat_pic/train/",batch_size,dtype)
    GenerateSeqMat(folder+"svalid-seq.txt",folder+"mat_pic/valid/",batch_size,dtype)
    GenerateSeqMat(folder+"stest-seq.txt",folder+"mat_pic/test/",batch_size,dtype)


GenerateSeqMatTVT("C:/Users/IBM_ADMIN/Data/watson_data/input/RecAE/wiki/",100,"int32")
