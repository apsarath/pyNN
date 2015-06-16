__author__ = 'sanbilp'

from pyNN import *

'''
Vanilla Elman Recurrent Network.

When sepemb = True, there will be a separate embedding matrix and separate W_in matrix. When sepemb = False, W_in acts as the embedding matrix.
learnh0 = True means the model learn h0 (initial hidden state). Else it is set to zero vector.
'''
class ElmanNet(Model):

    def init(self, numpy_rng, theano_rng=None, l_rate=None, optimization = "sgd", vsize=400, esize=10, n_hidden=200, n_out=10,
             hidden_activation = "sigmoid", output_activation="softmax", learnh0=True, sepemb=False, data=None, mask=None,
             embedding=None, W_in=None, W_r = None, b_h=None, h0=None, W_out=None, b_out=None, op_folder=None):


        Model.init(self, numpy_rng, theano_rng, optimization, l_rate, op_folder)

        self.vsize = vsize
        self.hparams["vsize"] = self.vsize
        self.esize = esize
        self.hparams["esize"] = self.esize
        self.n_hidden = n_hidden
        self.hparams["n_hidden"] = self.n_hidden
        self.hidden_activation = hidden_activation
        self.hparams["hidden_activation"] = self.hidden_activation
        self.output_activation = output_activation
        self.hparams["output_activation"] = self.output_activation
        self.n_out = n_out
        self.hparams["n_out"] = self.n_out
        self.learnh0 = learnh0
        self.hparams["learnh0"] = self.learnh0
        self.sepemb = sepemb
        self.hparams["sepemb"] = self.sepemb

        self.embedding = self.Initializer.gaussian("embedding", embedding, self.vsize, self.esize, 0, 0.1, 0)
        self.add_params("embedding", self.embedding, self.vsize, self.esize)

        if(self.sepemb==True):
            self.W_in = self.Initializer.gaussian("W_in", W_in, self.esize, self.n_hidden, 0, 0.1, 0)
            self.add_params("W_in", self.W_in, self.esize, self.n_hidden)

        self.W_r = self.Initializer.spectral("W_r", W_r, self.n_hidden, self.n_hidden, 1.1)
        self.add_params("W_r", self.W_r, self.n_hidden, self.n_hidden)

        self.b_h = self.Initializer.zero_vector("b_h", b_h, self.n_hidden)
        self.add_params("b_h", self.b_h, 1, self.n_hidden)

        if self.learnh0==True:
            self.h0 = self.Initializer.zero_vector("h0", h0, self.n_hidden)
            self.add_params("h0", self.h0, 1, self.n_hidden)

        self.W_out = self.Initializer.gaussian("W_out", W_out, self.n_hidden, self.n_out, 0, 0.1, 0)
        self.add_params("W_out", self.W_out, self.n_hidden, self.n_out)

        self.b_out = self.Initializer.zero_vector("b_out", b_out, self.n_out)
        self.add_params("b_out", self.b_out, 1, self.n_out)

        if data==None:
            self.data = T.imatrix()
        else:
            self.data = data
        if mask==None:
            self.mask = T.imatrix()
        else:
            self.mask = mask
        self.x = self.embedding[self.data.T]

        h_0 = T.alloc(numpy.asarray(0.0,dtype=theano.config.floatX),self.data.shape[0],self.n_hidden)
        if self.learnh0==True:
            h_0 = h_0 + self.h0

        l_0 = T.alloc(numpy.asarray(0.0,dtype=theano.config.floatX),self.data.shape[0])

        if self.sepemb==True:
            def recurrence(x_t, y_t, m_t, h_tm1, l_tm1, W_in, W_r, b_h, W_out, b_out):
                h_t = T.dot(x_t, W_in) + T.dot(h_tm1, W_r) + b_h
                h_t = activation(h_t, self.hidden_activation)
                o_t = T.dot(h_t, W_out) + b_out
                o_t = activation(o_t, self.output_activation)
                self.re = T.arange(self.data.shape[0])
                l_t = l_tm1 + -T.log(o_t[self.re, y_t]+1e-8)*m_t
                return [h_t, l_t]

            [_,l],_ = theano.scan(fn=recurrence, sequences=[self.x[0:self.x.shape[0]-1],self.data.T[1:self.data.T.shape[0]],self.mask.T[1:self.data.T.shape[0]]],
                                  non_sequences=[self.W_in, self.W_r, self.b_h, self.W_out, self.b_out],
                                  outputs_info=[h_0, l_0], n_steps=self.x.shape[0]-1, strict=True)

        else:
            def recurrence(x_t, y_t, m_t, h_tm1, l_tm1, W_r, b_h, W_out, b_out):
                h_t = x_t + T.dot(h_tm1, W_r) + b_h
                h_t = activation(h_t, self.hidden_activation)
                o_t = T.dot(h_t, W_out) + b_out
                o_t = activation(o_t, self.output_activation)
                self.re = T.arange(self.data.shape[0])
                l_t = l_tm1 + -T.log(o_t[self.re, y_t]+1e-8)*m_t
                return [h_t, l_t]


            [_,l],_ = theano.scan(fn=recurrence, sequences=[self.x[0:self.x.shape[0]-1],self.data.T[1:self.data.T.shape[0]],self.mask.T[1:self.data.T.shape[0]]],
                                  non_sequences=[self.W_r, self.b_h, self.W_out, self.b_out],
                                  outputs_info=[h_0, l_0], n_steps=self.x.shape[0]-1, strict=True)

        nll = T.sum(l[-1])

        gradients = T.grad(nll, self.params )
        updates = []
        for p,g,n in zip(self.params, gradients, self.param_names):
            gr, upd = self.optimizer.get_grad_update(n,g)
            updates.append((p,p+gr))
            updates.extend(upd)

        self.train = theano.function( inputs  = [self.data, self.mask],
                                      outputs = [nll],
                                      updates = updates )

        self.valid = theano.function( inputs  = [self.data, self.mask],
                                      outputs = [l[-1]])


#test case
rng = numpy.random.RandomState(123)
theano_rng = RandomStreams(rng.randint(2 ** 30))

x = theano.shared(numpy.asarray(numpy.zeros((5,4)), dtype="int32"), borrow=True)


net = ElmanNet()
net.init(rng,theano_rng,0.01,"sgd",100,10,5,100,"sigmoid","softmax",op_folder="../../../../watson_data/output/test/",sepemb=True)
a = numpy.arange(20).reshape((5,4))
b = numpy.ones((5,4),dtype="int32")
#b[4]=0
#b[4,3] = 1
#b[4] = 0
print net.train(a,b)
print net.valid(a,b)
