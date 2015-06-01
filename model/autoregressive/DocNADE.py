__author__ = 'Sarath'

# Author : Sarath Chandar


from pyNN import *
import time
from pyNN.optimization.optimization import *
from pyNN.util.Initializer import *
import pickle


class DocNADE(object):

    def init(self, numpy_rng, theano_rng=None, l_rate=None, optimization = "sgd", tied = False, n_visible=400, n_hidden=200,
             W=None, bhid=None, bvis=None, W_prime = None, input=None, op_folder=None):


        self.numpy_rng = numpy_rng
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        self.theano_rng = theano_rng

        # Set the number of visible units and hidden units in the network
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.tied = tied

        self.optimization = optimization
        self.l_rate = l_rate
        self.op_folder = op_folder

        self.optimizer = get_optimizer(optimization, l_rate)
        self.Initializer = Initializer(numpy_rng)

        self.W = self.Initializer.fan_based_sigmoid("W", W, n_visible, n_hidden)
        self.optimizer.register_variable("W",n_visible,n_hidden)

        if tied==False:
            self.W_prime = self.Initializer.fan_based_sigmoid("W_prime", W_prime, n_visible, n_hidden)
            self.optimizer.register_variable("W_prime",n_visible,n_hidden)
        else:
            self.W_prime = self.W

        self.b = self.Initializer.zero_vector("b", bhid, n_hidden)
        self.optimizer.register_variable("b",1,n_hidden)

        self.b_prime = self.Initializer.zero_vector("b_prime", bvis, n_visible)
        self.optimizer.register_variable("b_prime",1,n_visible)

        if input == None:
            self.x = T.imatrix(name='input')
        else:
            self.x = input

        if(tied==True):
            self.params = [self.W,  self.b, self.b_prime]
            self.param_names = ["W", "b", "b_prime"]
        else:
            self.params = [self.W, self.b, self.b_prime, self.W_prime]
            self.param_names = ["W", "b", "b_prime", "W_prime"]

        self.save_params()

    def get_nll(self):

        def cum(W,X,p_prev,a_prev,w_prev,b,V,bp):
            a = a_prev + w_prev
            h = T.nnet.sigmoid(a + b)
            o = T.nnet.softmax(T.dot(h,V.T) + bp)
            p = p_prev - T.log(o[0][X])
            return p,a,W
        doc_W = self.W[self.x].reshape((self.x.shape[1],self.W.shape[1]))
        p0 = T.zeros_like(self.x[0][0],dtype=theano.config.floatX)
        a0 = T.zeros_like(self.W[0],dtype=theano.config.floatX)
        w0 = T.zeros_like(self.W[0],dtype=theano.config.floatX)
        ([res1,_,_],_) = theano.scan(fn=cum,outputs_info=[p0,a0,w0],sequences=[doc_W,self.x[0]],non_sequences=[self.b,self.W_prime,self.b_prime])
        return res1[-1]

    def get_nll_cum(self):

        input_times_W = self.W[self.x][0][:,None,:]
        acc_input_times_W = T.cumsum(input_times_W, axis=0)
        acc_input_times_W = T.set_subtensor(acc_input_times_W[1:], acc_input_times_W[:-1])
        acc_input_times_W = T.set_subtensor(acc_input_times_W[0, :], 0.0)
        acc_input_times_W += self.b[None, None, :]
        h = T.nnet.sigmoid(acc_input_times_W).reshape((self.x.shape[1],self.W.shape[1]))
        pre_output = T.dot(h, self.W_prime.T) + self.b_prime
        output = T.nnet.softmax(pre_output)
        self.re = T.arange(self.x.shape[1])
        nll = T.sum(-T.log(output[self.re,self.x[0]]))
        return nll

    def get_nll_updates(self):

        nll = self.get_nll_cum()
        mean_nll = nll.mean()
        gradients = T.grad(mean_nll, self.params)
        updates = []
        for p,g,n in zip(self.params, gradients, self.param_names):
            gr, upd = self.optimizer.get_grad_update(n,g)
            updates.append((p,p+gr))
            updates.extend(upd)
        return (mean_nll, updates)


    def predict_nll(self):
        nll = self.get_nll_cum()
        return nll


    def get_lr_rate(self):
        return self.optimizer.get_l_rate()

    def set_lr_rate(self,new_lr):
        self.optimizer.set_l_rate(new_lr)


    def save_params(self):

        params = {}
        params["optimization"] = self.optimization
        params["l_rate"] = self.l_rate
        params["n_visible"] = self.n_visible
        params["n_hidden"] = self.n_hidden
        params["tied"] = self.tied
        params["numpy_rng"] = self.numpy_rng
        params["theano_rng"] = self.theano_rng

        pickle.dump(params,open(self.op_folder+"params.pck","wb"),-1)


    def load(self, folder, input=None):

        plist = pickle.load(open(folder+"params.pck","rb"))

        self.init(plist["numpy_rng"], theano_rng=plist["theano_rng"], l_rate=plist["l_rate"], optimization=plist["optimization"],
                  tied=plist["tied"], n_visible=plist["n_visible"], n_hidden=plist["n_hidden"],
                  W=folder+"W", bhid=folder+"bhid", bvis=folder+"bvis", W_prime=folder+"W_prime",
                  input=input, op_folder=folder)


    def save_matrices(self):
        for p,nm in zip(self.params, self.param_names):
            numpy.save(self.op_folder+nm, p.get_value(borrow=True))

