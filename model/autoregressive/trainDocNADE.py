__author__ = 'Sarath'

import sys
from os import listdir
from os.path import isfile, join
from random import shuffle

import os
import time
from numpy import *
import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from scipy import sparse

from pyNN.model.autoregressive.DocNADE import *

def trainDocNADE(src_folder, sct_folder, tgt_folder, batch_size = 20, n_hid = 40, learning_rate = 0.1, training_epochs = 40, gen_data = True, tied = False, use_valid= False, optimization= "sgd", vocab_size=100, bvis=None):

    n_vis = vocab_size

    x = T.imatrix('x')

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    da = DocNADE()
    da.init(numpy_rng=rng, theano_rng=theano_rng, l_rate = learning_rate, input=x, n_visible=n_vis, n_hidden=n_hid, bvis=bvis, tied = tied, optimization = optimization, op_folder=tgt_folder)
    #da.load(tgt_folder,x)
    start_time = time.clock()
    cost, updates = da.get_nll_updates()
    train_da = theano.function(inputs = [x], outputs= [cost],updates=updates)

    vcost = da.predict_nll()
    test_da = theano.function( inputs = [x], outputs = [vcost] )

    diff = 0
    flag = 1
    detfile = open(tgt_folder+"details.txt","w")
    detfile.close()
    oldtc = float("inf")

    for epoch in xrange(training_epochs):
        print "in epoch ", epoch
        c = []
        ipfile = open(src_folder+"train/seq.txt","r")
        for line in ipfile:
            words = line.strip().split("\t")
            words = words[1].split()
            nw = list()
            for x in words:
                nw.append(int(x))
            shuffle(nw)
            nw = numpy.asarray([nw],dtype=int32)
            c.append(train_da(nw))

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

        detfile = open(tgt_folder+"details.txt","a")
        detfile.write(str(diff)+"\n")
        detfile.close()
        # save the parameters for every 5 epochs
        da.save_matrices()


        if(use_valid==True):

            print "validating"
            tc = []
            ipfile = open(src_folder+"valid/seq.txt","r")

            for line in ipfile:
                #print "hi"
                words = line.strip().split("\t")
                words = words[1].split()
                nwa = list()
                for x in words:
                    nwa.append(int(x))
                #shuffle(nwa)
                nwa = numpy.asarray([nwa],dtype=int32)
                #print test_da(nwa)
                tc.append(test_da(nwa)[0]/len(nwa[0]))


            cur_tc = numpy.mean(tc)

            print cur_tc
            if(cur_tc < oldtc ):
                oldtc = cur_tc
            else:
                oldtc = cur_tc
                m = da.get_lr_rate() * 0.5
                da.set_lr_rate(m)
                print "updated lrate"




    end_time = time.clock()

    training_time = (end_time - start_time)

    print ' code ran for %.2fm' % (training_time / 60.)
    da.save_matrices()

    print "testing"
    tc = []
    ipfile = open(src_folder+"test/seq.txt","r")

    for line in ipfile:
        #print "hi"
        words = line.strip().split("\t")
        words = words[1].split()
        nwa = list()
        for x in words:
            nwa.append(int(x))
        #shuffle(nwa)
        nwa = numpy.asarray([nwa],dtype=int32)
        tc.append(test_da(nwa)[0]/len(nwa[0]))
    print numpy.mean(tc)

