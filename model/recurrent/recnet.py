# Author : Sarath Chandar

from pyNN import *

class RecurrentNet(object):

    def __init__(self, numpy_rng, l_rate=None, theano_rng=None,   ne=10, n_hidden=200, n_out=10, embedding=None,  W_out=None, W_r = None, bhid=None, h0=None, bout=None,  hidden_activation = "sigmoid", output_activation = "identity",optimization = "sgd"):

        
        # Set the number of visible units and hidden units in the network
        self.ne = ne   #vocab of src L
        self.n_hidden = n_hidden
        self.n_out = n_out    #vocab of tgt L
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        self.optimizer = get_optimizer(optimization, l_rate)
        self.Initializer = Initializer(numpy_rng)

        # Random seed
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        self.embedding = self.Initializer.gaussian("embedding", embedding, ne, n_hidden, 0, 1, 0.1)
        self.optimizer.register_variable("embedding",ne,n_hidden)

        self.W_r = self.Initializer.spectral("W_r", W_r, n_hidden, n_hidden, 1.1)
        self.optimizer.register_variable("W_r",n_hidden,n_hidden)
     
        self.W_out = self.Initializer.gaussian("W_out", W_out, n_hidden, n_out, 0, 1, 0.1)
        self.optimizer.register_variable("W_out",n_hidden,n_out)

        self.b_h = self.Initializer.zero_vector("b_h", bhid, n_hidden)
        self.optimizer.register_variable("b_h",1,n_hidden)

        self.h0 = self.Initializer.zero_vector("h0", h0, n_hidden)
        self.optimizer.register_variable("h0",1,n_hidden)

        self.b_out = self.Initializer.zero_vector("b_out", bout, n_out)        
        self.optimizer.register_variable("b_out",1,n_out)


        self.theano_rng = theano_rng
        
        self.idxs = T.imatrix()
        self.x = self.embedding[self.idxs].reshape((self.idxs.shape[1],self.n_hidden))


        self.params = [self.embedding, self.W_r, self.W_out, self.b_h, self.b_out, self.h0]
        self.param_names = ["embedding", "W_r", "W_out", "b_h", "b_out", "h0"]
       
        def recurrence(x_t, h_tm1):
            h_t = x_t + T.dot(h_tm1, self.W_r) + self.b_h
            h_t = activation(h_t, self.hidden_activation)
            s_t = T.dot(h_t, self.W_out) + self.b_out
            s_t = activation(s_t, self.output_activation)
            return [h_t, s_t]

        [h,s],_ = theano.scan(fn=recurrence, sequences=self.x,outputs_info=[self.h0, None], n_steps=self.x.shape[0])

        p_y_given_x_sentence = s[:,0,:]
        self.re = T.arange(self.x.shape[0])
        nll = T.sum(-T.log(p_y_given_x_sentence[self.re,self.idxs[0]]))

        gradients = T.grad( nll, self.params )
        updates = []
        for p,g,n in zip(self.params, gradients, self.param_names):
            gr, upd = self.optimizer.get_grad_update(n,g)
            updates.append((p,p+gr))
            updates.extend(upd)

        # theano functions
        self.train = theano.function( inputs  = [self.idxs],
                                      outputs = [nll],
                                      updates = updates )

        self.valid = theano.function( inputs  = [self.idxs],
                                      outputs = [nll])



    def get_lr_rate(self):
        return self.optimizer.get_l_rate()

    def set_lr_rate(self,new_lr):
        self.optimizer.set_l_rate(new_lr)


    # This method saves W, bvis and bhid matrices. `n` is the string attached to the file name. 
    def save_matrices(self,folder):

        for p,nm in zip(self.params, self.param_names):
            numpy.save(folder+nm, p.get_value(borrow=True))
