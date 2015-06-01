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

class JordanNet(object):

    def __init__(self, numpy_rng, l_rate=None, theano_rng=None, wsize=400, esize=10, ne=10, n_hidden=200, n_out=10, embedding=None,  W_in=None, W_out=None, W_r = None, bhid=None, h0=None, bout=None, W_prime = None, hidden_activation = "sigmoid", output_activation = "identity", loss = "squarrederror",optimization = "sgd"):

        
        # Set the number of visible units and hidden units in the network
        self.wsize = wsize
        self.esize = esize
        self.ne = ne   #vocab of src L
        n_visible = self.wsize * self.esize
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.n_out = n_out    #vocab of tgt L
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.loss = loss


        self.optimizer = get_optimizer(optimization, l_rate)
        self.Initializer = Initializer(numpy_rng)

        # Random seed
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        self.embedding = self.Initializer.gaussian("embedding", embedding, ne, esize, 0, 1, 0.1)
        self.optimizer.register_variable("embedding",ne,esize)
      
        self.W_in = self.Initializer.gaussian("W_in", W_in, n_visible, n_hidden, 0, 1, 0.1)
        self.optimizer.register_variable("W_in",n_visible,n_hidden)
    
        self.W_r = self.Initializer.spectral("W_r", W_r, n_out, n_hidden, 1.1)
        self.optimizer.register_variable("W_r",n_out,n_hidden)
     
        self.W_out = self.Initializer.gaussian("W_out", W_out, n_hidden, n_out, 0, 1, 0.1)
        self.optimizer.register_variable("W_out",n_hidden,n_out)

        self.b_h = self.Initializer.zero_vector("b_h", bhid, n_hidden)
        self.optimizer.register_variable("b_h",1,n_hidden)


        self.b_out = self.Initializer.zero_vector("b_out", bout, n_out)        
        self.optimizer.register_variable("b_out",1,n_out)




        self.theano_rng = theano_rng
        
        self.idxs = T.imatrix()

        self.x = self.embedding[self.idxs].reshape((self.idxs.shape[0],self.n_visible))

        self.y = T.iscalar('y') 
            

        self.params = [self.embedding, self.W_in, self.W_r, self.W_out, self.b_h, self.b_out]
        self.param_names = ["embedding", "W_in", "W_r", "W_out", "b_h", "b_out"]
       
        def recurrence(x_t, s_tm1):
            h_t = T.dot(x_t, self.W_in) + T.dot(s_tm1, self.W_r) + self.b_h
            h_t = activation(h_t, self.hidden_activation)
            s_t = T.dot(h_t, self.W_out) + self.b_out
            s_t = activation(s_t, self.output_activation)
            return s_t[0],s_t

        b0 = T.zeros_like(self.W_out[0],dtype=theano.config.floatX)

        [_,s],_ = theano.scan(fn=recurrence, sequences=self.x,outputs_info=[b0,None], n_steps=self.x.shape[0])

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
