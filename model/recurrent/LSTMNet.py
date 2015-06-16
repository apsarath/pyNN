__author__ = 'sanbilp'


from pyNN import *

'''
LSTM Recurrent Network.

When sepemb = True, there will be a separate embedding matrix and separate W_in matrix. When sepemb = False, W_in acts as the embedding matrix.
learnh0 = True means the model learn h0 (initial hidden state). Else it is set to zero vector.
'''
class LSTMNet(Model):

    def init(self, numpy_rng, theano_rng=None, l_rate=None, optimization = "sgd", vsize=400, esize=10, n_hidden=200, n_out=10,
             hidden_activation = "sigmoid", output_activation="softmax", learnh0=True, sepemb=False, data=None, mask=None,
             embedding=None, W_x=None, W_h = None, b=None, h0=None, W_out=None, b_out=None, op_folder=None):


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

        if W_x == None:
            W_xi = W_xj = W_xf = W_xo = None
            self.W_xi = self.Initializer.gaussian("W_xi", W_xi, self.esize, self.n_hidden, 0, 0.1, 0, getnp=True)
            self.W_xj = self.Initializer.gaussian("W_xj", W_xj, self.esize, self.n_hidden, 0, 0.1, 0, getnp=True)
            self.W_xf = self.Initializer.gaussian("W_xf", W_xf, self.esize, self.n_hidden, 0, 0.1, 0, getnp=True)
            self.W_xo = self.Initializer.gaussian("W_xo", W_xo, self.esize, self.n_hidden, 0, 0.1, 0, getnp=True)
            self.W_x = numpy.concatenate((self.W_xi, self.W_xj, self.W_xf, self.W_xo), axis=1)
        else:
            self.W_x = self.Initializer.load("W_x", W_x, getnp=True)

        self.W_x = theano.shared(value=self.W_x, name="W_x", borrow=True)
        self.add_params("W_x", self.W_x, self.esize, 4*self.n_hidden)

        if W_h == None:
            W_hi = W_hj = W_hf = W_ho = None
            self.W_hi = self.Initializer.spectral("W_hi", W_hi, self.n_hidden, self.n_hidden, 1.1, getnp=True)
            self.W_hj = self.Initializer.spectral("W_hj", W_hj, self.n_hidden, self.n_hidden, 1.1, getnp=True)
            self.W_hf = self.Initializer.spectral("W_hf", W_hf, self.n_hidden, self.n_hidden, 1.1, getnp=True)
            self.W_ho = self.Initializer.spectral("W_ho", W_ho, self.n_hidden, self.n_hidden, 1.1, getnp=True)
            self.W_h = numpy.concatenate((self.W_hi, self.W_hj, self.W_hf, self.W_ho), axis=1)
        else:
            self.W_h = self.Initializer.load("W_h", W_h, getnp=True)

        self.W_h = theano.shared(value=self.W_h, name="W_h", borrow=True)
        self.add_params("W_h", self.W_h, self.n_hidden, 4*self.n_hidden)

        if b == None:
            b_i = b_j = b_f = b_o = None
            self.b_i = self.Initializer.zero_vector("b_i", b_i, self.n_hidden, getnp=True)
            self.b_j = self.Initializer.zero_vector("b_j", b_j, self.n_hidden, getnp=True)
            self.b_f = self.Initializer.one_vector("b_f", b_f, self.n_hidden, getnp=True)
            self.b_o = self.Initializer.zero_vector("b_o", b_o, self.n_hidden, getnp=True)
            self.b = numpy.concatenate((self.b_i, self.b_j, self.b_f, self.b_o), axis=1)
        else:
            self.b = self.Initializer.load("b", b, getnp=True)

        self.b = theano.shared(value=self.b, name="b", borrow=True)
        self.add_params("b", self.b, 1, 4*self.n_hidden)

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
        c_0 = T.alloc(numpy.asarray(0.0,dtype=theano.config.floatX),self.data.shape[0],self.n_hidden)
        if self.learnh0==True:
            h_0 = h_0 + self.h0

        l_0 = T.alloc(numpy.asarray(0.0,dtype=theano.config.floatX),self.data.shape[0])


        def recurrence(x_t, y_t, m_t, h_tm1, c_tm1, l_tm1, W_x, W_h, b, W_out, b_out):
            # This is similar to Jozefowicz et. al., ICML 2015
            def _slice(_x, n, dim):
                if _x.ndim == 3:
                    return _x[:, :, n*dim:(n+1)*dim]
                return _x[:, n*dim:(n+1)*dim]

            preact = T.dot(x_t, W_x) + T.dot(h_tm1, W_h) + b

            i = T.tanh(_slice(preact, 0, self.n_hidden))
            j = T.nnet.sigmoid(_slice(preact, 1, self.n_hidden))
            f = T.nnet.sigmoid(_slice(preact, 2, self.n_hidden))
            o = T.nnet.sigmoid(_slice(preact, 3, self.n_hidden))
            c_t = c_tm1 * f + i * j
            h_t = T.tanh(c_t) * o

            o_t = T.dot(h_t, W_out) + b_out
            o_t = activation(o_t, self.output_activation)
            self.re = T.arange(self.data.shape[0])
            l_t = l_tm1 + -T.log(o_t[self.re, y_t]+1e-8)*m_t
            return [h_t, c_t, l_t]

        [_,_,l],_ = theano.scan(fn=recurrence, sequences=[self.x[0:self.x.shape[0]-1],self.data.T[1:self.data.T.shape[0]], self.mask.T[1:self.data.T.shape[0]]],
                              non_sequences=[self.W_x, self.W_h, self.b, self.W_out, self.b_out],
                              outputs_info=[h_0, c_0, l_0], n_steps=self.x.shape[0]-1, strict=True)


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
net = LSTMNet()
net.init(rng,theano_rng,0.01,"sgd",100,10,5,100,"sigmoid","softmax",op_folder="../../../../watson_data/output/test/")
a = numpy.arange(20).reshape((5,4))
b = numpy.ones((5,4),dtype="int32")
#b[4]=0
#b[4,3] = 1
#b[4] = 0
print net.train(a,b)
print net.valid(a,b)
