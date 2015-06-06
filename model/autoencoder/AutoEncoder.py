# Author : Sarath Chandar

from pyNN import *
import time
from pyNN.optimization.optimization import *
from pyNN.util.Initializer import *
import pickle

# hi
class Autoencoder(object):

    def init(self, numpy_rng, theano_rng=None, l_rate=None, optimization = "sgd", tied = False, n_visible=400, n_hidden=200,W=None, bhid=None, bvis=None, W_prime = None, input=None, hidden_activation = "sigmoid", output_activation = "identity", loss_fn = "squarrederror", op_folder=None):


        self.numpy_rng = numpy_rng
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        self.theano_rng = theano_rng

        self.optimization = optimization
        self.l_rate = l_rate

        self.optimizer = get_optimizer(self.optimization, self.l_rate)
        self.Initializer = Initializer(self.numpy_rng)

        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.tied = tied
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.loss_fn = loss_fn
        self.op_folder = op_folder


        if self.hidden_activation == "sigmoid":
            self.W = self.Initializer.fan_based_sigmoid("W", W, n_visible, n_hidden)
            self.optimizer.register_variable("W",n_visible,n_hidden)
        else:
            self.W = self.Initializer.fan_based("W", W, n_visible, n_hidden)
            self.optimizer.register_variable("W",n_visible,n_hidden)

        if not tied:
            self.W_prime = self.Initializer.fan_based_sigmoid("W_prime", W_prime, n_hidden, n_visible)
            self.optimizer.register_variable("W_prime",n_hidden, n_visible)
        else:
            self.W_prime = self.W.T

        self.b = self.Initializer.zero_vector("b", bhid, n_hidden)
        self.optimizer.register_variable("b",1,n_hidden)

        self.b_prime = self.Initializer.zero_vector("b_prime", bvis, n_visible)
        self.optimizer.register_variable("b_prime",1,n_visible)

        if input == None:
            self.x = T.dmatrix(name='input')
        else:
            self.x = input

        if(tied==True):
            self.params = [self.W,  self.b, self.b_prime]
            self.param_names = ["W", "b", "b_prime"]
        else:
            self.params = [self.W, self.b, self.b_prime, self.W_prime]
            self.param_names = ["W", "b", "b_prime", "W_prime"]

        self.save_params()


    def get_cost(self):

        y = T.dot(self.x, self.W) + self.b
        y = activation(y, self.hidden_activation)
        z = T.dot(y, self.W_prime) + self.b_prime
        z = activation(z, self.output_activation)
        L = loss(z, self.x, self.loss_fn)
        return L        

    def get_cost_updates(self):

        cost = self.get_cost()
        cost = T.mean(cost)
        updates = []
        gradients = T.grad(cost, self.params)
        for p,g,n in zip(self.params, gradients, self.param_names):
            gr, upd = self.optimizer.get_grad_update(n,g)
            updates.append((p,p+gr))
            updates.extend(upd)
        return (cost, updates)        

    def get_lr_rate(self):
        return self.optimizer.get_l_rate()

    def set_lr_rate(self,new_lr):
        self.optimizer.set_l_rate(new_lr)

    def save_matrices(self):

        for p,nm in zip(self.params, self.param_names):
            numpy.save(self.op_folder+nm, p.get_value(borrow=True))

    def save_params(self):

        params = {}
        params["optimization"] = self.optimization
        params["l_rate"] = self.l_rate
        params["n_visible"] = self.n_visible
        params["n_hidden"] = self.n_hidden
        params["hidden_activation"] = self.hidden_activation
        params["output_activation"] = self.output_activation
        params["loss_fn"] = self.loss_fn
        params["tied"] = self.tied
        params["numpy_rng"] = self.numpy_rng
        params["theano_rng"] = self.theano_rng

        pickle.dump(params,open(self.op_folder+"params.pck","wb"),-1)


    def load(self, folder, input=None):

        plist = pickle.load(open(folder+"params.pck","rb"))

        self.init(plist["numpy_rng"], theano_rng=plist["theano_rng"], l_rate=plist["l_rate"], optimization=plist["optimization"],
                  tied=plist["tied"], n_visible=plist["n_visible"],n_hidden=plist["n_hidden"], W=folder+"W", bhid=folder+"b", W_prime=folder+"W_prime",
                  bvis=folder+"b_prime", input=input, hidden_activation=plist["hidden_activation"], output_activation=plist["output_activation"],
                  loss_fn = plist["loss_fn"], op_folder=folder)


def trainAutoencoder(src_folder, sct_folder, tgt_folder, batch_size = 20, training_epochs = 40, use_valid=False, l_rate=0.01, optimization = "sgd", tied = False, n_visible=400, n_hidden=200, W=None, bhid=None, bvis=None, W_prime = None, hidden_activation = "sigmoid", output_activation = "identity", loss_fn = "squarrederror"):


    index = T.lscalar()
    x = T.matrix('x')

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    da = Autoencoder()
    da.init(numpy_rng=rng, theano_rng=theano_rng, l_rate=l_rate, optimization=optimization, tied=tied, n_visible=n_visible, n_hidden = n_hidden, W=W, bhid=bhid, bvis=bvis, W_prime=W_prime, input=x, hidden_activation=hidden_activation, output_activation=output_activation, loss_fn=loss_fn, op_folder=tgt_folder)

    start_time = time.clock()
    train_set_x = theano.shared(numpy.asarray(numpy.zeros((50000,n_visible)), dtype=theano.config.floatX), borrow=True)

    cost, updates = da.get_cost_updates()
    train_da = theano.function([index], cost,updates=updates,givens=[(x, train_set_x[index * batch_size:(index + 1) * batch_size])])

    vcost = da.get_cost()
    test_da = theano.function([index], vcost, givens=[(x, train_set_x[index * batch_size:(index + 1) * batch_size])])


    diff = 0
    flag = 1
    detfile = open(tgt_folder+"details.txt","w")
    detfile.close()
    oldtc = float("inf")
    for epoch in xrange(training_epochs):
        print "in epoch ", epoch
        c = []
        ipfile = open(sct_folder+"mat_pic/train/ip.txt","r")

        for line in ipfile:
            next = line.strip().split(",")
            denseTheanoloader(next[0],train_set_x,"float32")
            for batch_index in range(0,int(next[1])):
                c.append(train_da(batch_index))

        if(flag==1):
            flag = 0
            diff = numpy.mean(c)
            di = diff
        else:
            di = numpy.mean(c) - diff
            diff = numpy.mean(c)
            print 'Difference between 2 epochs is ', di
        print 'Training epoch %d, cost ' % epoch, diff

        ipfile.close()

        detfile = open(tgt_folder+"details.txt","a")
        detfile.write(str(diff)+"\n")
        detfile.close()
        if((epoch+1)%5==0):
            da.save_matrices()


        if(use_valid==True):
            print "validating"
            tc = []
            ipfile = open(sct_folder+"mat_pic/valid/ip.txt","r")
            for line in ipfile:
                next = line.strip().split(",")
                denseTheanoloader(next[0],train_set_x,"float32")
                for batch_index in range(0,int(next[1])):
                    tc.extend(test_da(batch_index))
            cur_tc = numpy.mean(tc)
            print cur_tc
            if(cur_tc < oldtc ):
                oldtc = cur_tc
            else:
                oldtc = cur_tc
                m = da.get_lr_rate() * 0.5
                da.set_lr_rate(m)
                print "updated lrate"
    end_time = time.clock()
    training_time = (end_time - start_time)

    print ' code ran for %.2fm' % (training_time / 60.)
    da.save_matrices()




