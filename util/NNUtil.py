__author__ = 'Sarath'

import os
import numpy
from os import listdir
from os.path import isfile, join
import sys

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from scipy import sparse

def create_folder(folder):

	if not os.path.exists(folder):
		os.makedirs(folder)

def denseTheanoloader(file, x, bit):
    mat = denseloader(file,bit)
    x.set_value(mat, borrow=True)

def sparseTheanoloader(file, x, bit, row, col):
    mat = sparseloader(file, bit, row, col)
    x.set_value(mat, borrow=True)

def denseloader(file,bit):

	#print "loading ...", file
	matrix = numpy.load(file+".npy")
	matrix = numpy.array(matrix,dtype=bit)
	return matrix

def sparseloader(file,bit,row,col):

	print "loading ...", file
	x = numpy.load(file+"d.npy")
	y = numpy.load(file+"i.npy")
	z = numpy.load(file+"p.npy")
	matrix = sparse.csr_matrix((x,y,z),shape=(row,col), dtype = bit)
	matrix = matrix.todense()
	return matrix


def prepare_dataset(src_folder, sct_folder, batch_size):

	mat_pic = sct_folder+"mat_pic/"

	create_folder(mat_pic)
	create_folder(mat_pic+"train/")
	create_folder(mat_pic+"test/")
	create_folder(mat_pic+"valid/")

	'''    Get the number of features from the data       '''
	trainfiles = [ src_folder+"train/"+f for f in listdir(src_folder+"train/") if isfile(join(src_folder+"train/",f)) ]
	trainfiles = [ x for x in trainfiles if '_label' not in x]
	file = open(trainfiles[0],"r")
	fts = len(file.readline().strip().split())
	file.close()

	'''     Get the number of labels from the data       '''
	labelfile = trainfiles[0].split(".")[0]+'_label.txt'
	if(isfile(labelfile)):
		file = open(labelfile,"r")
		labels = len(file.readline().strip().split())
		file.close()
	else:
		labels = None

	prepare_data(src_folder+"train/", sct_folder+"mat_pic/train/", batch_size, fts, labels)
	prepare_data(src_folder+"valid/", sct_folder+"mat_pic/valid/", batch_size, fts, labels)
	prepare_data(src_folder+"test/", sct_folder+"mat_pic/test/", batch_size, fts, labels)



def prepare_data(src_folder, tgt_folder, batch_size, fts, labels):


	fcount = 0

	files = [ src_folder+f for f in listdir(src_folder) if isfile(join(src_folder,f)) ]
	files = [ x for x in files if '_label' not in x]

	if len(files)==0:
		return

	'''     Preparing all data files          '''
	ipfile = open(tgt_folder+"ip.txt","w")

	for filename in files:

		file = open(filename,"r")
		matrix = numpy.zeros((1000,fts))
		count = 0
		for line in file:
			line = line.strip().split()
			for i in range(0,fts):
				matrix[count][i] = float(line[i])
			count+=1
			if(count==1000):
				numpy.save(tgt_folder+str(fcount),matrix)
				ipfile.write(tgt_folder+str(fcount)+","+str(1000/batch_size)+"\n")
				matrix = numpy.zeros((1000,fts))
				count = 0
				fcount += 1
		file.close()

	if(count!=0):
		numpy.save(tgt_folder+str(fcount),matrix)
		ipfile.write(tgt_folder+str(fcount)+","+str(count/batch_size)+"\n")

	ipfile.close()


	'''      Preparing all label files (if any)        '''
	if(labels != None):

		fcount = 0
		for filename in files:
			file = open(filename.split(".")[0]+'_label.txt',"r")
			matrix = numpy.zeros((1000,labels))
			count = 0

			for line in file:
				line = line.strip().split()
				for i in range(0,labels):
					matrix[count][i] = float(line[i])
				count+=1
				if(count==1000):
					numpy.save(tgt_folder+str(fcount)+"_label",matrix)
					mat = numpy.zeros((1000,labels))
					count = 0
					fcount+=1
			file.close()
		if(count!=0):
			numpy.save(tgt_folder+str(fcount)+"_label",matrix)


def activation(x, function):

	if (function == "sigmoid"):
		return T.nnet.sigmoid(x)
	elif(function == "tanh"):
		return T.tanh(x)
	elif(function == "identity"):
		return x
	elif(function == "softmax"):
		return T.nnet.softmax(x)
	elif(function == "softplus"):
		return T.nnet.softplus(x)
	elif(function == "relu"):
		return T.switch(x<0, 0, x)


def loss(pred, tgt, function):

	if (function == "squarrederror"):
		return T.sum(T.sqr(tgt-pred)/2,axis = 1)
	elif (function == "crossentrophy"):
		return -T.sum(tgt * T.log(pred) + (1 - tgt) * T.log(1 - pred), axis=1)



def bptt_batch(l, bs):
    '''
    l :: list of word idxs
    return a list of minibatches of indexes
    which size is equal to bs
    border cases are treated as follow:
    eg: [0,1,2,3] and bs = 3
    will output:
    [[0],[0,1],[0,1,2],[1,2,3]]
    '''
    out  = [l[:i] for i in xrange(1, min(bs,len(l)+1) )]
    out += [l[i-bs:i] for i in xrange(bs,len(l)+1) ]
    assert len(l) == len(out)
    return out

def contextwin(l, win, wtype="mid"):
    '''
    win :: int corresponding to the size of the window
    given a list of indexes composing a sentence
    it will return a list of list of indexes corresponding
    to context windows surrounding each word in the sentence

    The context window will be padded with 0 in the beginning and 1 in the end.
    '''

    if(wtype == "mid"):

    	#print "hi"
	    assert (win % 2) == 1
	    assert win >=1
	    l = list(l)

	    lpadded = win/2 * [0] + l + win/2 * [1]
	    out = [ lpadded[i:i+win] for i in range(len(l)) ]

	    assert len(out) == len(l)
	    return out


def log_sum_exp(x, axis=1):
    max_x = T.max(x, axis)
    return max_x + T.log(T.sum(T.exp(x - T.shape_padright(max_x, 1)), axis))
