__author__ = 'sanbilp'


from pyNN import *

'''
GRU Recurrent Network.

When sepemb = True, there will be a separate embedding matrix and separate W_in matrix. When sepemb = False, W_in acts as the embedding matrix.
learnh0 = True means the model learn h0 (initial hidden state). Else it is set to zero vector.
'''
class GRUNet(Model):

    def init(self, numpy_rng, theano_rng=None, l_rate=None, optimization = "sgd", vsize=400, esize=10, n_hidden=200, n_out=10,
             hidden_activation = "sigmoid", output_activation="softmax", learnh0=True, sepemb=False, data=None, mask=None,
             embedding=None, W_x=None, W_h = None, b=None, W_xh=None, W_hh=None, b_h=None, h0=None, W_out=None, b_out=None, op_folder=None):


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
            W_xr = W_xz = None
            self.W_xr = self.Initializer.gaussian("W_xr", W_xr, self.esize, self.n_hidden, 0, 0.1, 0, getnp=True)
            self.W_xz = self.Initializer.gaussian("W_xz", W_xz, self.esize, self.n_hidden, 0, 0.1, 0, getnp=True)
            self.W_x = numpy.concatenate((self.W_xr, self.W_xz), axis=1)
        else:
            self.W_x = self.Initializer.load("W_x", W_x, getnp=True)

        self.W_x = theano.shared(value=self.W_x, name="W_x", borrow=True)
        self.add_params("W_x", self.W_x, self.esize, 2*self.n_hidden)

        self.W_xh = self.Initializer.gaussian("W_xh", W_xh, self.esize, self.n_hidden, 0, 0.1, 0)
        self.add_params("W_xh", self.W_xh, self.esize, self.n_hidden)

        if W_h == None:
            W_hr = W_hz = None
            self.W_hr = self.Initializer.spectral("W_hr", W_hr, self.n_hidden, self.n_hidden, 1.1, getnp=True)
            self.W_hz = self.Initializer.spectral("W_hz", W_hz, self.n_hidden, self.n_hidden, 1.1, getnp=True)
            self.W_h = numpy.concatenate((self.W_hr, self.W_hz), axis=1)
        else:
            self.W_h = self.Initializer.load("W_h", W_h, getnp=True)

        self.W_h = theano.shared(value=self.W_h, name="W_h", borrow=True)
        self.add_params("W_h", self.W_h, self.n_hidden, 2*self.n_hidden)

        self.W_hh = self.Initializer.spectral("W_hh", W_hh, self.n_hidden, self.n_hidden, 1.1)
        self.add_params("W_hh", self.W_hh, self.n_hidden, self.n_hidden)


        if b == None:
            b_r = b_z = None
            self.b_r = self.Initializer.zero_vector("b_r", b_r, self.n_hidden, getnp=True)
            self.b_z = self.Initializer.zero_vector("b_z", b_z, self.n_hidden, getnp=True)
            self.b = numpy.concatenate((self.b_r, self.b_z), axis=1)
        else:
            self.b = self.Initializer.load("b", b, getnp=True)

        self.b = theano.shared(value=self.b, name="b", borrow=True)
        self.add_params("b", self.b, 1, 2*self.n_hidden)

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


        def recurrence(x_t, y_t, m_t, h_tm1, l_tm1, W_x, W_h, b, W_out, b_out, W_xh, W_hh, b_h):
            # This is similar to GRU defn in Jozefowicz et. al., ICML 2015
            def _slice(_x, n, dim):
                if _x.ndim == 3:
                    return _x[:, :, n*dim:(n+1)*dim]
                return _x[:, n*dim:(n+1)*dim]

            preact = T.dot(x_t, W_x) + T.dot(h_tm1, W_h) + b

            r = T.nnet.sigmoid(_slice(preact, 0, self.n_hidden))
            z = T.nnet.sigmoid(_slice(preact, 1, self.n_hidden))
            hp = T.tanh(T.dot(x_t, W_xh) + T.dot(r*h_tm1, W_hh) + b_h )
            h_t = z*h_tm1 + (1-z)*hp

            o_t = T.dot(h_t, W_out) + b_out
            o_t = activation(o_t, self.output_activation)
            self.re = T.arange(self.data.shape[0])
            l_t = l_tm1 + -T.log(o_t[self.re, y_t]+1e-8)*m_t
            return [h_t, l_t]

        [_,l],_ = theano.scan(fn=recurrence, sequences=[self.x[0:self.x.shape[0]-1],self.data.T[1:self.data.T.shape[0]], self.mask.T[1:self.data.T.shape[0]]],
                              non_sequences=[self.W_x, self.W_h, self.b, self.W_out, self.b_out, self.W_xh, self.W_hh, self.b_h],
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


"""#test case
rng = numpy.random.RandomState(123)
theano_rng = RandomStreams(rng.randint(2 ** 30))
net = GRUNet()
net.init(rng,theano_rng,0.01,"sgd",100,10,5,100,"sigmoid","softmax",op_folder="../../../../watson_data/output/test/")
a = numpy.arange(20).reshape((5,4))
b = numpy.ones((5,4),dtype="int32")
#b[4]=0
#b[4,3] = 1
#b[4] = 0
print net.train(a,b)
print net.valid(a,b)
"""