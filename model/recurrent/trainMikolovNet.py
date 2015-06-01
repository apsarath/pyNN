import sys
from os import listdir
from os.path import isfile, join

import os
import time
from numpy import *
import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from scipy import sparse

sys.path.append('../../')

from pyNN import *

from MikolovNet import *


def trainMikolovNet(src_folder, sct_folder, tgt_folder, batch_size = 20, wsize=1, esize=10, ne=10, n_hidden=5, n_out=10, n_context=10, embedding=None,  W_in=None, W_out=None, W_r = None, B=None, V=None, P=None, bhid=None, h0=None, c0=None, bout=None, W_prime = None,  learning_rate = 0.1, training_epochs = 40, hidden_activation = "sigmoid", output_activation = "softmax",loss = "crossentrophy",use_valid=False, optimization = "sgd", bptt=10, alpha = 0.95):




		

	rng = numpy.random.RandomState(123)
	theano_rng = RandomStreams(rng.randint(2 ** 30))

	
	recnet = MikolovNet( numpy_rng = rng, l_rate=learning_rate, theano_rng=theano_rng, wsize=wsize, esize=esize, ne=ne, n_hidden=n_hidden, n_out=n_out, embedding=embedding,  W_in=W_in, W_out=W_out, W_r = W_r, bhid=bhid, h0=h0, bout=bout, c0=c0, B=B, V=V, P=P, hidden_activation = hidden_activation, output_activation = output_activation, loss = loss,optimization = optimization, n_context = n_context)

	start_time = time.clock()

	    

	diff = 0
	flag = 1

	detfile = open(tgt_folder+"details.txt","w")
	detfile.close()

	oldtc = float("inf")



	for epoch in xrange(training_epochs):

	    print "in epoch ", epoch

	    #print da.get_lr_rate()

	    
	    c = []
	    
	    ipfile = open(sct_folder+"trn.iseq","r")
	    opfile = open(sct_folder+"trn.oseq", "r")

	    count = 0
	    for line in ipfile:
	    	if(count%100 == 0):
	    		print str(count) + '\r' 
	    	count+=1
	    	iseq = line.strip().split()
	    	oseq = opfile.readline().strip().split()
	    	cwords = contextwin(iseq, wsize)
	    	words  = map(lambda x: numpy.asarray(x).astype('int32'),bptt_batch(cwords, bptt))

	    	for word_batch , label_last_word in zip(words, oseq):
	    		#print word_batch, label_last_word
	    		nll =  recnet.train(word_batch, int(label_last_word))
	    		c.append(nll)
	    		recnet.normalize()

	    print 		
	            
	    if(flag==1):
	        flag = 0
	        diff = numpy.mean(c)
	        di = diff
	    else:
	        di = numpy.mean(c) - diff
	        diff = numpy.mean(c)
	        print 'Difference between 2 epochs is ', di
	    print 'Training epoch %d, cost ' % epoch, diff

	    ipfile.close()
	    opfile.close()

	    detfile = open(tgt_folder+"details.txt","a")
	    detfile.write("train "+str(diff)+"\n")
	    detfile.close()
	    # save the parameters for every 2 epochs
	    if((epoch+1)%2==0):
	        recnet.save_matrices(tgt_folder, str(epoch))
	        

	    if(use_valid==True):

		    print "validating"

		    tc = []

		    ipfile = open(sct_folder+"dev.iseq","r")
		    opfile = open(sct_folder+"dev.oseq", "r")

		    for line in ipfile:
		    	iseq = line.strip().split()
		    	oseq = opfile.readline().strip().split()
		    	cwords = contextwin(iseq, wsize)
		    	words  = map(lambda x: numpy.asarray(x).astype('int32'),bptt_batch(cwords, bptt))

		    	for word_batch , label_last_word in zip(words, oseq):
		    		nll =  recnet.valid(word_batch, int(label_last_word))
		    		tc.append(nll)
	                
		    
		    cur_tc = numpy.mean(tc)

		    detfile = open(tgt_folder+"details.txt","a")
		    detfile.write("valid "+str(cur_tc)+"\n")
		    detfile.close()
	    	

		    print cur_tc
		    if(cur_tc < oldtc ):
		    	oldtc = cur_tc
		    else:
		    	oldtc = cur_tc
		    	m = recnet.get_lr_rate() * 0.5
		    	recnet.set_lr_rate(m)
		    	print "updated lrate" 



	end_time = time.clock()

	training_time = (end_time - start_time)

	print ' code ran for %.2fm' % (training_time / 60.)
	recnet.save_matrices(tgt_folder,"final")




"""



rng = numpy.random.RandomState(123)
theano_rng = RandomStreams(rng.randint(2 ** 30))
da = RecurrentNet( numpy_rng = rng, l_rate=0.1, theano_rng=theano_rng, wsize=1, esize=10, ne=10, n_hidden=5, n_out=10, embedding=None,  W_in=None, W_out=None, W_r = None, bhid=None, h0=None, bout=None, W_prime = None, hidden_activation = "sigmoid", output_activation = "softmax", loss = "crossentrophy",optimization = "sgd")

print da.train([[1],[2],[3],[4]],3)
da.normalize()
print da.classify([[1],[2],[3],[4]])
"""