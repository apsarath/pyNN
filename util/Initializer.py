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




class Initializer(object):

	def __init__(self, numpy_rng):

		self.numpy_rng = numpy_rng


	def load(self, param_name, param):

		print "loading "+ param_name +" matrix"
		initial_param = numpy.load(param+".npy")
		final_param = theano.shared(value=initial_param, name=param_name, borrow=True)
		return final_param


	def fan_based(self, param_name, param, d1, d2):
		if not param:

			print "fan-based initialization of "+param_name
			initial_param = numpy.asarray(self.numpy_rng.uniform(low=-1 * numpy.sqrt(6. / (d1 + d2)),high=1 * numpy.sqrt(6. / (d1 + d2)),size=(d1, d2)), dtype=theano.config.floatX)
			final_param = theano.shared(value = initial_param, name = param_name, borrow = True)
			return final_param
		else:
			return self.load(param_name, param)

	def fan_based_sigmoid(self, param_name, param, d1, d2):
		if not param:

			print "fan-based-sigmoid initialization of "+param_name
			initial_param = numpy.asarray(self.numpy_rng.uniform(low=-1 * numpy.sqrt(6. / (d1 + d2)),high=1 * numpy.sqrt(6. / (d1 + d2)),size=(d1, d2)), dtype=theano.config.floatX)
			initial_param = initial_param * 4
			final_param = theano.shared(value = initial_param, name = param_name, borrow = True)
			return final_param
		else:
			return self.load(param_name, param)

	def random_vector(self, param_name, param, d1):
		if not param:

			print "random initialization of "+param_name
			initial_param = numpy.asarray(self.numpy_rng.uniform(low=-1 * numpy.sqrt(6. / (d1 + 1)),high=1 * numpy.sqrt(6. / (d1 + 1)),size=d1), dtype=theano.config.floatX)
			final_param = theano.shared(value = initial_param, name = param_name, borrow = True)
			return final_param
		else:
			return self.load(param_name, param)


	def zero_vector(self, param_name, param, d1):
		if not param:
			print "zeros initialization of "+param_name
			param = theano.shared(value = numpy.zeros(d1,dtype=theano.config.floatX), borrow=True)
			return param
		else:
			return self.load(param_name, param)

	def one_vector(self, param_name, param, d1, scale= 1):
		if not param:
			print "ones initialization of "+param_name
			param = theano.shared(value = numpy.ones(d1,dtype=theano.config.floatX)*scale, borrow=True)
			return param
		else:
			return self.load(param_name, param)


	def zero_matrix(self, param_name, param, d1, d2):
		if not param:
			print "zeros initialization of "+param_name
			param = theano.shared(value = numpy.zeros((d1,d2), dtype = theano.config.floatX), borrow = True)
			return param
		else:
			return self.load(param_name, param)

	def identity_matrix(self, param_name, param, dim):
		if not param:
			print "identity matrix initialization of "+param_name
			param = theano.shared(value = numpy.identity(dim, dtype = theano.config.floatX), borrow = True)
			return param
		else:
			return self.load(param_name, param)

	def gaussian(self, param_name, param, d1, d2, mean, sd, center):
		if not param:
			print "normal initialization of "+param_name
			initial_param = numpy.asarray(self.numpy_rng.normal(loc=mean, scale = sd, size=(d1,d2)),dtype=theano.config.floatX)
			initial_param = center + initial_param
			final_param = theano.shared(value = initial_param, name = param_name, borrow = True)
			return final_param

		else:
			return self.load(param_name, param)

	def spectral(self, param_name, param, d1, d2, scale):
		if not param:
			print "spectral initialization of "+param_name
			values = numpy.zeros((d1,d2), dtype = theano.config.floatX)
			for dx in xrange(d1):
				new_vals = self.numpy_rng.uniform(low = -scale, high = scale, size=(d2,))
				vals_norm = numpy.sqrt((new_vals**2).sum())
				new_vals = scale * new_vals/vals_norm
				values[dx] = new_vals
			_,v,_ = numpy.linalg.svd(values)
			values = scale * values/v[0]
			final_param = theano.shared(value = values, name = param_name, borrow = True)
			return final_param

		else:
			return self.load(param_name, param)
