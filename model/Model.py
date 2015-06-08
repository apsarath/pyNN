__author__ = 'sanbilp'

import pickle

from pyNN.optimization.optimization import *
from pyNN.util.Initializer import *

class Model(object):
    ''' Base class for all Neural Network Models
    '''

    def init(self, numpy_rng, theano_rng, optimization, l_rate, op_folder):

        print "in model"

        self.numpy_rng = numpy_rng
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        self.theano_rng = theano_rng

        self.optimization = optimization
        self.l_rate = l_rate
        self.op_folder = op_folder

        self.optimizer = get_optimizer(self.optimization, self.l_rate)
        self.Initializer = Initializer(self.numpy_rng)

        self.params = list()
        self.param_names = list()

        self.hparams = {}
        self.hparams["numpy_rng"] = self.numpy_rng
        self.hparams["theano_rng"] = self.theano_rng
        self.hparams["optimization"] = self.optimization
        self.hparams["l_rate"] = self.l_rate


    def add_params(self, param_name, param, d1, d2):

        self.params.append(param)
        self.param_names.append(param_name)
        self.optimizer.register_variable(param_name, d1, d2)

    def get_lr_rate(self):
        return self.optimizer.get_l_rate()

    def set_lr_rate(self,new_lr):
        self.optimizer.set_l_rate(new_lr)

    def save_matrices(self):

        for p,nm in zip(self.params, self.param_names):
            numpy.save(self.op_folder+nm, p.get_value(borrow=True))

    def save_params(self):

        pickle.dump(self.hparams,open(self.op_folder+"params.pck","wb"),-1)

