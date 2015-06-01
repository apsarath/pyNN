# Author : Sarath Chandar


import os
import sys
import time
from numpy import *
import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from scipy import sparse
from collections import OrderedDict


sys.path.append('../../')

from pyNN import *

from optimization import get_optimizer

class MikolovNet(object):

    def __init__(self, numpy_rng, l_rate=None, theano_rng=None, wsize=400, esize=10, ne=10, n_hidden=200, n_context=100, n_out=10, embedding=None,  W_in=None, W_out=None, W_r = None, B = None, P = None, V = None, bhid=None, h0=None, c0=None, bout=None, W_prime = None, hidden_activation = "sigmoid", output_activation = "identity", loss = "squarrederror",optimization = "sgd", alpha=0.95):

        
        # Set the number of visible units and hidden units in the network
        self.wsize = wsize
        self.esize = esize
        self.ne = ne   #vocab of src L
        n_visible = self.wsize * self.esize
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.n_context = n_context
        self.n_out = n_out    #vocab of tgt L
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.loss = loss
        self.alpha = alpha


        self.optimizer = get_optimizer(optimization, l_rate)

        # Random seed
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))


        if not embedding:

            embedding = numpy.asarray(numpy_rng.uniform(low=-1, high=1,size=(ne, esize)), dtype=theano.config.floatX)
            embedding = theano.shared(value=embedding, name="embedding", borrow=True)

        else:
            print "loading embedding"
            embedding = numpy.load(embedding+".npy")
            embedding = theano.shared(value=embedding, name = "embedding", borrow = True)

        self.embedding = embedding

        self.optimizer.register_variable("embedding",ne,esize)

        if not W_in:
            print "randomly initializing W"
            initial_W_in = numpy.asarray(numpy_rng.uniform(low=-1 * numpy.sqrt(6. / (n_hidden + n_visible)),high=1 * numpy.sqrt(6. / (n_hidden + n_visible)),size=(n_visible, n_hidden)), dtype=theano.config.floatX)
            if self.hidden_activation == "sigmoid":
                initial_W_in = initial_W_in * 4
            W_in = theano.shared(value=initial_W_in, name='W_in', borrow=True)

        else:
            print "loading W_in matrix"
            initial_W_in = numpy.load(W_in+".npy")
            W_in = theano.shared(value=initial_W_in, name='W_in', borrow=True)
      
        self.W_in = W_in

        self.optimizer.register_variable("W_in",n_visible,n_hidden)


        if not W_r:
            print "randomly initializing W_r"
            initial_W_r = numpy.asarray(numpy_rng.uniform(low=-1 * numpy.sqrt(6. / (n_hidden + n_hidden)),high=1 * numpy.sqrt(6. / (n_hidden + n_hidden)),size=(n_hidden, n_hidden)), dtype=theano.config.floatX)
            if self.hidden_activation == "sigmoid":
                initial_W_r = initial_W_r * 4
            W_r = theano.shared(value=initial_W_r, name='W_r', borrow=True)

        else:
            print "loading W_r matrix"
            initial_W_r = numpy.load(W_r+".npy")
            W_r = theano.shared(value=initial_W_r, name='W_r', borrow=True)
      
        self.W_r = W_r

        self.optimizer.register_variable("W_r",n_hidden,n_hidden)


        if not B:
            print "randomly initializing B"
            initial_B = numpy.asarray(numpy_rng.uniform(low=-1 * numpy.sqrt(6. / (n_visible + n_context)),high=1 * numpy.sqrt(6. / (n_visible + n_context)),size=(n_visible, n_context)), dtype=theano.config.floatX)
            B = theano.shared(value=initial_B, name = 'B', borrow = True)

        else:
            print "loading B matrix"
            initial_B = numpy.load(B+".npy")
            B = theano.shared(value=initial_B, name = 'B', borrow = True)

        self.B = B
        self.optimizer.register_variable("B",n_visible,n_context)


        if not P:
            print "randomly initializing P"
            initial_P = numpy.asarray(numpy_rng.uniform(low=-1 * numpy.sqrt(6. / (n_context + n_hidden)),high=1 * numpy.sqrt(6. / (n_context + n_hidden)),size=(n_context, n_hidden)), dtype=theano.config.floatX)
            if self.hidden_activation == "sigmoid":
                initial_P = initial_P * 4
            P = theano.shared(value=initial_P, name='P', borrow=True)

        else:
            print "loading P matrix"
            initial_P = numpy.load(P+".npy")
            P = theano.shared(value=initial_P, name='P', borrow=True)
      
        self.P = P

        self.optimizer.register_variable("P",n_context,n_hidden)



        if not V:
            print "randomly initializing V"
            initial_V = numpy.asarray(numpy_rng.uniform(low=-1 * numpy.sqrt(6. / (n_context + n_out)),high=1 * numpy.sqrt(6. / (n_context + n_out)),size=(n_context, n_out)), dtype=theano.config.floatX)
            if self.hidden_activation == "sigmoid":
                initial_V = initial_V * 4
            V = theano.shared(value=initial_V, name='V', borrow=True)

        else:
            print "loading V matrix"
            initial_V = numpy.load(V+".npy")
            V = theano.shared(value=initial_V, name='V', borrow=True)
      
        self.V = V

        self.optimizer.register_variable("V",n_context,n_out)



        if not W_out:
            print "randomly initializing W_out"
            initial_W_out = numpy.asarray(numpy_rng.uniform(low=-1 * numpy.sqrt(6. / (n_hidden + n_out)),high=1 * numpy.sqrt(6. / (n_hidden + n_out)),size=(n_hidden, n_out)), dtype=theano.config.floatX)
            W_out = theano.shared(value=initial_W_out, name='W_out', borrow=True)

        else:
            print "loading W_out matrix"
            initial_W_out = numpy.load(W_out+".npy")
            W_out = theano.shared(value=initial_W_out, name='W_out', borrow=True)
      
        self.W_out = W_out

        self.optimizer.register_variable("W_out",n_hidden,n_out)


        if not bhid:
            print "randomly initializing hidden bias"
            bhid = theano.shared(value=numpy.zeros(n_hidden,dtype=theano.config.floatX),name='bhid',borrow=True)

        else:
            print "loading hidden bias"
            initial_bhid = numpy.load(bhid+".npy")
            bhid = theano.shared(value=initial_bhid, name='bhid', borrow=True)

        self.b_h = bhid

        self.optimizer.register_variable("b_h",1,n_hidden)



        if not h0:
            print "randomly initializing h0"
            h0 = theano.shared(value=numpy.zeros(n_hidden,dtype=theano.config.floatX),name='h0',borrow=True)

        else:
            print "loading h0"
            initial_h0 = numpy.load(h0+".npy")
            h0 = theano.shared(value=initial_h0, name='h0', borrow=True)

        self.h0 = h0

        self.optimizer.register_variable("h0",1,n_hidden)



        if not c0:
            print "randomly initializing c0"
            c0 = theano.shared(value=numpy.zeros(n_context,dtype=theano.config.floatX),name='c0',borrow=True)

        else:
            print "loading c0"
            initial_c0 = numpy.load(c0+".npy")
            c0 = theano.shared(value=initial_c0, name='c0', borrow=True)

        self.c0 = c0




        if not bout:
            print "randomly initializing op bias"
            bout = theano.shared(value=numpy.zeros(n_out,dtype=theano.config.floatX),borrow=True)

        else:
            print "loading op bias"
            initial_bout = numpy.load(bout+".npy")
            bout = theano.shared(value=initial_bout, name='bout', borrow=True)
     
        
        self.b_out = bout
        
        self.optimizer.register_variable("b_out",1,n_out)




        self.theano_rng = theano_rng
        
        self.idxs = T.imatrix()

        self.x = self.embedding[self.idxs].reshape((self.idxs.shape[0],self.n_visible))

        self.y = T.iscalar('y') 
            

        self.params = [self.embedding, self.W_in, self.W_r, self.W_out, self.b_h, self.b_out, self.h0, self.B, self.P, self.V]
        self.param_names = ["embedding", "W_in", "W_r", "W_out", "b_h", "b_out", "h0", "B", "P", "V"]
       
        def recurrence(x_t, h_tm1, c_tm1):

            c_t =  (1 - self.alpha) * T.dot(x_t, self.B) + self.alpha * c_tm1
            h_t =  T.dot(c_t, self.P)  +  T.dot(x_t, self.W_in) + T.dot(h_tm1, self.W_r) + self.b_h
            h_t = activation(h_t, self.hidden_activation)
            s_t = T.dot(c_t, self.V) + T.dot(h_t, self.W_out) + self.b_out
            s_t = activation(s_t, self.output_activation)
            return [h_t, c_t, s_t]

        [_,_,s],_ = theano.scan(fn=recurrence, sequences=self.x,outputs_info=[self.h0,self.c0, None], n_steps=self.x.shape[0])

        p_y_given_x_lastword = s[-1,0,:]
        p_y_given_x_sentence = s[:,0,:]
        y_pred = T.argmax(p_y_given_x_sentence, axis=1)
        #y_pred = p_y_given_x_sentence

        nll = -T.mean(T.log(p_y_given_x_lastword)[self.y])


        gradients = T.grad( nll, self.params )
        updates = []

        for p,g,n in zip(self.params, gradients, self.param_names):
            gr, upd = self.optimizer.get_grad_update(n,g)
            updates.append((p,p+gr))
            updates.extend(upd)

        # theano functions
        self.classify = theano.function(inputs=[self.idxs], outputs=y_pred)

        self.train = theano.function( inputs  = [self.idxs, self.y],
                                      outputs = [nll],
                                      updates = updates )

        self.valid = theano.function( inputs  = [self.idxs, self.y],
                                      outputs = [nll])

        self.normalize = theano.function( inputs = [],
                         updates = {self.embedding:\
                         self.embedding/T.sqrt((self.embedding**2).sum(axis=1)).dimshuffle(0,'x')})


  


    def get_lr_rate(self):
        return self.optimizer.get_l_rate()

    def set_lr_rate(self,new_lr):
        self.optimizer.set_l_rate(new_lr)



    # This method saves W, bvis and bhid matrices. `n` is the string attached to the file name. 
    def save_matrices(self,folder,n):

        for p,nm in zip(self.params, self.param_names):
            numpy.save(folder+nm+n, p.get_value(borrow=True))
