# Author : Sarath Chandar

from pyNN import *
import time
from pyNN.optimization.optimization import *
from pyNN.util.Initializer import *
import pickle

class LSTM(object):

    def __init__(self, numpy_rng, l_rate=None, theano_rng=None, wsize=400, esize=10, ne=10, n_hidden=200, n_out=10, embedding=None, W_z=None, U_z=None, W_r=None, U_r=None, W=None, U=None,  W_out=None, bhid=None, b_z=None, b_r=None, h0=None, bout=None,  hidden_activation = "sigmoid", output_activation = "identity", loss = "squarrederror",optimization = "sgd"):

        self.numpy_rng = numpy_rng
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        self.theano_rng = theano_rng

        self.ne = ne
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        self.optimizer = get_optimizer(optimization, l_rate)
        self.Initializer = Initializer(numpy_rng)

        # Random seed
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        self.embedding = self.Initializer.gaussian("embedding", embedding, self.ne, self.n_visible, 0, 1, 0.1)
        self.optimizer.register_variable("embedding",self.ne,self.n_visible)
      
        self.W_ix = self.Initializer.gaussian("W_ix", W_ix, self.n_visible, self.n_hidden, 0, 1, 0.1)
        self.optimizer.register_variable("W_ix",self.n_visible, self.n_hidden)
        self.W_fx = self.Initializer.gaussian("W_fx", W_fx, self.n_visible, self.n_hidden, 0, 1, 0.1)
        self.optimizer.register_variable("W_fx",self.n_visible, self.n_hidden)
        self.W_ox = self.Initializer.gaussian("W_ox", W_ox, self.n_visible, self.n_hidden, 0, 1, 0.1)
        self.optimizer.register_variable("W_ox",self.n_visible, self.n_hidden)
        self.W_gx = self.Initializer.gaussian("W_gx", W_gx, self.n_visible, self.n_hidden, 0, 1, 0.1)
        self.optimizer.register_variable("W_gx",self.n_visible, self.n_hidden)

        self.W_im = self.Initializer.spectral("W_im", W_im, self.n_hidden, self.n_hidden, 1.1)
        self.optimizer.register_variable("W_im",self.n_hidden,self.n_hidden)
        self.W_fm = self.Initializer.spectral("W_fm", W_fm, self.n_hidden, self.n_hidden, 1.1)
        self.optimizer.register_variable("W_fm",self.n_hidden,self.n_hidden)
        self.W_om = self.Initializer.spectral("W_om", W_om, self.n_hidden, self.n_hidden, 1.1)
        self.optimizer.register_variable("W_om",self.n_hidden,self.n_hidden)
        self.W_gm = self.Initializer.spectral("W_gm", W_gm, self.n_hidden, self.n_hidden, 1.1)
        self.optimizer.register_variable("W_gm",self.n_hidden,self.n_hidden)

        self.b_i = self.Initializer.zero_vector("b_i", b_i, self.n_hidden)
        self.optimizer.register_variable("b_i",1,self.n_hidden)
        self.b_f = self.Initializer.zero_vector("b_f", b_f, self.n_hidden)
        self.optimizer.register_variable("b_f",1,self.n_hidden)
        self.b_o = self.Initializer.zero_vector("b_o", b_o, self.n_hidden)
        self.optimizer.register_variable("b_o",1,self.n_hidden)
        self.b_g = self.Initializer.zero_vector("b_g", b_g, self.n_hidden)
        self.optimizer.register_variable("b_g",1,self.n_hidden)
        self.m_zero = self.Initializer.zero_vector("m_zero", m_zero, self.n_hidden)
        self.optimizer.register_variable("m_zero",1,self.n_hidden)

        self.idxs = T.imatrix()
        self.x = self.embedding[self.idxs].reshape((self.idxs.shape[0],self.n_visible))
        self.irep = T.dvector()

        self.params = [self.embedding, self.W_ix, self.W_fx, self.W_ox, self.W_gx,
                       self.W_im, self.W_fm, self.W_om, self.W_gm,
                       self.b_i, self.b_f, self.b_o, self.b_g, self.m_zero]
        self.param_names = ["embedding", W_ix, W_fx, W_ox, W_gx,
                            W_im, W_fm, W_om, W_gm,
                            b_i, b_f, b_o, b_g, m_zero]
       
        def recurrence(x_t, y_t, m_tm1, c_tm1, logp):

            i_t = T.nnet.sigmoid(T.dot(x_t, self.W_ix) + T.dot(m_tm1, self.W_im) + self.b_i)
            f_t = T.nnet.sigmoid(T.dot(x_t, self.W_fx) + T.dot(m_tm1, self.W_fm) + self.b_f)
            o_t = T.nnet.sigmoid(T.dot(x_t, self.W_ox) + T.dot(m_tm1, self.W_om) + self.b_o)
            g_t = T.tanh(T.dot(x_t, self.W_gx) + T.dot(m_tm1, self.W_gm) + self.b_g)
            c_t = f_t * c_tm1 + i_t * g_t
            m_t = o_t * c_t
            p_t = T.nnet.softmax(m_t)
            logp += -T.log(p_t[y_t])
            return [m_t, c_t, logp]

        i_t = T.nnet.sigmoid(T.dot(self.irep, self.W_ix) + T.dot(self.m_zero, self.W_im) + self.b_i)
        o_t = T.nnet.sigmoid(T.dot(self.irep, self.W_ox) + T.dot(self.m_zero, self.W_om) + self.b_o)
        g_t = T.tanh(T.dot(self.irep, self.W_gx) + T.dot(self.m_zero, self.W_gm) + self.b_g)
        c_t = i_t * g_t
        m_t = o_t * c_t

        [_,_,logp],_ = theano.scan(fn=recurrence, sequences=[self.x,self.y],outputs_info=[m_t, c_t, None], n_steps=self.x.shape[0])

        loss = logp
        gradients = T.grad( loss, self.params )
        updates = []

        for p,g,n in zip(self.params, gradients, self.param_names):
            gr, upd = self.optimizer.get_grad_update(n,g)
            updates.append((p,p+gr))
            updates.extend(upd)

        self.train = theano.function( inputs  = [self.idxs, self.irep],
                                      outputs = [loss],
                                      updates = updates )

    def get_lr_rate(self):
        return self.optimizer.get_l_rate()

    def set_lr_rate(self,new_lr):
        self.optimizer.set_l_rate(new_lr)


    # This method saves W, bvis and bhid matrices. `n` is the string attached to the file name.
    def save_matrices(self,folder,n):

        for p,nm in zip(self.params, self.param_names):
            numpy.save(folder+nm+n, p.get_value(borrow=True))
