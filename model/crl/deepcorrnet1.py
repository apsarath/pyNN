__author__ = 'Sarath'

from pyNN import *
import time
from pyNN.optimization.optimization import *
from pyNN.util.Initializer import *
import pickle

class DeepCorrNet1(object):

    def init(self, numpy_rng, theano_rng=None, l_rate=0.01, optimization="sgd", tied=False, n_visible_left=None, n_visible_right=None, n_hidden=None, n_hidden2=None, lamda=5, W_left=None, W_right=None, b_left=None, b_right=None, W_left_prime=None, W_right_prime=None, b_prime_left=None, b_prime_right=None, W_left2=None, W_right2=None, b2=None, W_left_prime2=None, W_right_prime2=None, b_prime_left2=None, b_prime_right2=None, input_left=None, input_right=None, hidden_activation="sigmoid", output_activation="sigmoid", loss_fn = "squarrederror", op_folder=None):

        self.numpy_rng = numpy_rng
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        self.theano_rng = theano_rng

        self.optimization = optimization
        self.l_rate = l_rate

        self.optimizer = get_optimizer(self.optimization, self.l_rate)
        self.Initializer = Initializer(self.numpy_rng)

        self.n_visible_left = n_visible_left
        self.n_visible_right = n_visible_right
        self.n_hidden = n_hidden
        self.n_hidden2 = n_hidden2
        self.lamda = lamda
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.loss_fn = loss_fn
        self.tied = tied
        self.op_folder = op_folder

        self.W_left = self.Initializer.fan_based_sigmoid("W_left", W_left, n_visible_left, n_hidden)
        self.optimizer.register_variable("W_left",n_visible_left,n_hidden)

        self.W_right = self.Initializer.fan_based_sigmoid("W_right", W_right, n_visible_right, n_hidden)
        self.optimizer.register_variable("W_right",n_visible_right,n_hidden)

        self.W_left2 = self.Initializer.fan_based_sigmoid("W_left2", W_left2, n_hidden, n_hidden2)
        self.optimizer.register_variable("W_left2",n_hidden, n_hidden2)

        self.W_right2 = self.Initializer.fan_based_sigmoid("W_right2", W_right2, n_hidden, n_hidden2)
        self.optimizer.register_variable("W_right2", n_hidden, n_hidden2)


        if not tied:
            self.W_left_prime = self.Initializer.fan_based_sigmoid("W_left_prime", W_left_prime, n_hidden, n_visible_left)
            self.optimizer.register_variable("W_left_prime",n_hidden, n_visible_left)
            self.W_right_prime = self.Initializer.fan_based_sigmoid("W_right_prime", W_right_prime, n_hidden, n_visible_right)
            self.optimizer.register_variable("W_right_prime",n_hidden, n_visible_right)

            self.W_left_prime2 = self.Initializer.fan_based_sigmoid("W_left_prime2", W_left_prime2, n_hidden2, n_hidden)
            self.optimizer.register_variable("W_left_prime2",n_hidden2, n_hidden)
            self.W_right_prime2 = self.Initializer.fan_based_sigmoid("W_right_prime2", W_right_prime2, n_hidden2, n_hidden)
            self.optimizer.register_variable("W_right_prime2",n_hidden2, n_hidden)

        else:
            self.W_left_prime = self.W_left.T
            self.W_right_prime = self.W_right.T

            self.W_left_prime2 = self.W_left2.T
            self.W_right_prime2 = self.W_right2.T

        self.b_left = self.Initializer.zero_vector("b_left", b_left, n_hidden)
        self.optimizer.register_variable("b_left",1,n_hidden)
        self.b_right = self.Initializer.zero_vector("b_right", b_right, n_hidden)
        self.optimizer.register_variable("b_right",1,n_hidden)

        self.b_prime_left = self.Initializer.zero_vector("b_prime_left", b_prime_left, n_visible_left)
        self.optimizer.register_variable("b_prime_left",1,n_visible_left)
        self.b_prime_right = self.Initializer.zero_vector("b_prime_right", b_prime_right, n_visible_right)
        self.optimizer.register_variable("b_prime_right",1,n_visible_right)

        self.b2 = self.Initializer.zero_vector("b2", b2, n_hidden2)
        self.optimizer.register_variable("b2",1,n_hidden2)

        self.b_prime_left2 = self.Initializer.zero_vector("b_prime_left2", b_prime_left2, n_hidden)
        self.optimizer.register_variable("b_prime_left2",1,n_hidden)
        self.b_prime_right2 = self.Initializer.zero_vector("b_prime_right2", b_prime_right2, n_hidden)
        self.optimizer.register_variable("b_prime_right2",1,n_hidden)


        if input_left is None:
            self.x_left = T.matrix(name='x_left')
        else:
            self.x_left = input_left

        if input_right is None:
            self.x_right = T.matrix(name='x_right')
        else:
            self.x_right = input_right


        if tied:
            self.params = [self.W_left, self.W_right,  self.b_left, self.b_right, self.b_prime_left, self.b_prime_right, self.W_left2, self.W_right2,  self.b2, self.b_prime_left2, self.b_prime_right2]
            self.param_names = ["W_left", "W_right", "b_left", "b_right", "b_prime_left", "b_prime_right", "W_left2", "W_right2", "b2", "b_prime_left2", "b_prime_right2"]
        else:
            self.params = [self.W_left, self.W_right,  self.b_left, self.b_right, self.b_prime_left, self.b_prime_right, self.W_left_prime, self.W_right_prime, self.W_left2, self.W_right2,  self.b2, self.b_prime_left2, self.b_prime_right2, self.W_left_prime2, self.W_right_prime2]
            self.param_names = ["W_left", "W_right", "b_left", "b_right", "b_prime_left", "b_prime_right", "W_left_prime", "W_right_prime", "W_left2", "W_right2", "b2", "b_prime_left2", "b_prime_right2", "W_left_prime2", "W_right_prime2"]


        self.proj_from_left = theano.function([self.x_left],self.project_from_left())
        self.proj_from_right = theano.function([self.x_right],self.project_from_right())
        self.recon_from_left = theano.function([self.x_left],self.reconstruct_from_left())
        self.recon_from_right = theano.function([self.x_right],self.reconstruct_from_right())

        self.save_params()


    def train_common(self,mtype="1111"):

        y1_pre = T.dot(self.x_left, self.W_left) + self.b_left
        y1 = activation(y1_pre, self.hidden_activation)
        yy1_pre = T.dot(y1, self.W_left2) + self.b2
        yy1 = activation(yy1_pre, self.hidden_activation)
        z1_left_pre = T.dot(yy1, self.W_left_prime2) + self.b_prime_left2
        z1_right_pre = T.dot(yy1,self.W_right_prime2) + self.b_prime_right2
        z1_left = activation(z1_left_pre, self.output_activation)
        z1_right = activation(z1_right_pre, self.output_activation)
        zz1_left_pre = T.dot(z1_left, self.W_left_prime) + self.b_prime_left
        zz1_right_pre = T.dot(z1_right,self.W_right_prime) + self.b_prime_right
        zz1_left = activation(zz1_left_pre, self.output_activation)
        zz1_right = activation(zz1_right_pre, self.output_activation)

        L1 = loss(zz1_left, self.x_left, self.loss_fn) + loss(zz1_right, self.x_right, self.loss_fn)

        y2_pre = T.dot(self.x_right, self.W_right) + self.b_right
        y2 = activation(y2_pre, self.hidden_activation)
        yy2_pre = T.dot(y2, self.W_right2) + self.b2
        yy2 = activation(yy2_pre, self.hidden_activation)
        z2_left_pre = T.dot(yy2, self.W_left_prime2) + self.b_prime_left2
        z2_right_pre = T.dot(yy2,self.W_right_prime2) + self.b_prime_right2
        z2_left = activation(z2_left_pre, self.output_activation)
        z2_right = activation(z2_right_pre, self.output_activation)
        zz2_left_pre = T.dot(z2_left, self.W_left_prime) + self.b_prime_left
        zz2_right_pre = T.dot(z2_right,self.W_right_prime) + self.b_prime_right
        zz2_left = activation(zz2_left_pre, self.output_activation)
        zz2_right = activation(zz2_right_pre, self.output_activation)

        L2 = loss(zz2_left, self.x_left, self.loss_fn) + loss(zz2_right, self.x_right, self.loss_fn)

        y3left_pre = T.dot(self.x_left, self.W_left) + self.b_left
        y3right_pre = T.dot(self.x_right, self.W_right) + self.b_right
        y3left = activation(y3left_pre, self.hidden_activation)
        y3right = activation(y3right_pre, self.hidden_activation)
        y3_pre = T.dot(y3left, self.W_left2) + T.dot(y3right, self.W_right2) + self.b2
        y3 = activation(y3_pre, self.hidden_activation)

        z3_left_pre = T.dot(y3, self.W_left_prime2) + self.b_prime_left2
        z3_right_pre = T.dot(y3,self.W_right_prime2) + self.b_prime_right2
        z3_left = activation(z3_left_pre, self.output_activation)
        z3_right = activation(z3_right_pre, self.output_activation)
        zz3_left_pre = T.dot(z3_left, self.W_left_prime) + self.b_prime_left
        zz3_right_pre = T.dot(z3_right,self.W_right_prime) + self.b_prime_right
        zz3_left = activation(zz3_left_pre, self.output_activation)
        zz3_right = activation(zz3_right_pre, self.output_activation)

        L3 = loss(zz3_left, self.x_left, self.loss_fn) + loss(zz3_right, self.x_right, self.loss_fn)

        y1_mean = T.mean(yy1, axis=0)
        y1_centered = yy1 - y1_mean
        y2_mean = T.mean(yy2, axis=0)
        y2_centered = yy2 - y2_mean
        corr_nr = T.sum(y1_centered * y2_centered, axis=0)
        corr_dr1 = T.sqrt(T.sum(y1_centered * y1_centered, axis=0)+1e-8)
        corr_dr2 = T.sqrt(T.sum(y2_centered * y2_centered, axis=0)+1e-8)
        corr_dr = corr_dr1 * corr_dr2
        corr = corr_nr/corr_dr
        L4 = T.sum(corr) * self.lamda

        if mtype=="1111":
            print "1111"
            L = L1 + L2 + L3 - L4
        elif mtype=="1110":
            print "1110"
            L = L1 + L2 + L3
        elif mtype=="1101":
            print "1101"
            L = L1 + L2 - L4
        elif mtype == "0011":
            print "0011"
            L = L3 - L4
        elif mtype == "1100":
            print "1100"
            L = L1 + L2
        elif mtype == "0010":
            print "0010"
            L = L3

        cost = T.mean(L)

        gradients = T.grad(cost, self.params)
        updates = []
        for p,g,n in zip(self.params, gradients, self.param_names):
            gr, upd = self.optimizer.get_grad_update(n,g)
            updates.append((p,p+gr))
            updates.extend(upd)

        return cost, updates


    def project_from_left(self):

        y1_pre = T.dot(self.x_left, self.W_left) + self.b_left
        y1 = activation(y1_pre, self.hidden_activation)
        yy1_pre = T.dot(y1, self.W_left2) + self.b2
        yy1 = activation(yy1_pre, self.hidden_activation)
        return yy1

    def project_from_right(self):

        y2_pre = T.dot(self.x_right, self.W_right) + self.b_right
        y2 = activation(y2_pre, self.hidden_activation)
        yy2_pre = T.dot(y2, self.W_right2) + self.b2
        yy2 = activation(yy2_pre, self.hidden_activation)
        return yy2

    def reconstruct_from_left(self):

        y1_pre = T.dot(self.x_left, self.W_left) + self.b_left
        y1 = activation(y1_pre, self.hidden_activation)
        yy1_pre = T.dot(y1, self.W_left2) + self.b2
        yy1 = activation(yy1_pre, self.hidden_activation)
        z1_left_pre = T.dot(yy1, self.W_left_prime2) + self.b_prime_left2
        z1_right_pre = T.dot(yy1,self.W_right_prime2) + self.b_prime_right2
        z1_left = activation(z1_left_pre, self.output_activation)
        z1_right = activation(z1_right_pre, self.output_activation)
        zz1_left_pre = T.dot(z1_left, self.W_left_prime) + self.b_prime_left
        zz1_right_pre = T.dot(z1_right,self.W_right_prime) + self.b_prime_right
        zz1_left = activation(zz1_left_pre, self.output_activation)
        zz1_right = activation(zz1_right_pre, self.output_activation)
        return zz1_left, zz1_right

    def reconstruct_from_right(self):

        y2_pre = T.dot(self.x_right, self.W_right) + self.b_right
        y2 = activation(y2_pre, self.hidden_activation)
        yy2_pre = T.dot(y2, self.W_right2) + self.b2
        yy2 = activation(yy2_pre, self.hidden_activation)
        z2_left_pre = T.dot(yy2, self.W_left_prime2) + self.b_prime_left2
        z2_right_pre = T.dot(yy2,self.W_right_prime2) + self.b_prime_right2
        z2_left = activation(z2_left_pre, self.output_activation)
        z2_right = activation(z2_right_pre, self.output_activation)
        zz2_left_pre = T.dot(z2_left, self.W_left_prime) + self.b_prime_left
        zz2_right_pre = T.dot(z2_right,self.W_right_prime) + self.b_prime_right
        zz2_left = activation(zz2_left_pre, self.output_activation)
        zz2_right = activation(zz2_right_pre, self.output_activation)
        return zz2_left, zz2_right

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
        params["n_visible_left"] = self.n_visible_left
        params["n_visible_right"] = self.n_visible_right
        params["n_hidden"] = self.n_hidden
        params["n_hidden2"] = self.n_hidden2
        params["lamda"] = self.lamda
        params["hidden_activation"] = self.hidden_activation
        params["output_activation"] = self.output_activation
        params["loss_fn"] = self.loss_fn
        params["tied"] = self.tied
        params["numpy_rng"] = self.numpy_rng
        params["theano_rng"] = self.theano_rng

        pickle.dump(params,open(self.op_folder+"params.pck","wb"),-1)


    def load(self, folder, input_left=None, input_right=None):

        plist = pickle.load(open(folder+"params.pck","rb"))
        print plist["n_hidden"]
        print type(plist["n_hidden"])

        self.init(plist["numpy_rng"], theano_rng=plist["theano_rng"], l_rate=plist["l_rate"], optimization=plist["optimization"],
                  tied=plist["tied"], n_visible_left=plist["n_visible_left"], n_visible_right=plist["n_visible_right"], n_hidden=plist["n_hidden"], n_hidden2=plist["n_hidden2"],
                  lamda=plist["lamda"], W_left=folder+"W_left", W_right=folder+"W_right", b_left=folder+"b_left", b_right=folder+"b_right", W_left_prime=folder+"W_left_prime",
                  W_right_prime=folder+"W_right_prime", b_prime_left=folder+"b_prime_left", b_prime_right=folder+"b_prime_right",
                  W_left2=folder+"W_left2", W_right2=folder+"W_right2", b2=folder+"b2",  W_left_prime2=folder+"W_left_prime2",
                  W_right_prime2=folder+"W_right_prime2", b_prime_left2=folder+"b_prime_left2", b_prime_right2=folder+"b_prime_right2",
                  input_left=input_left, input_right=input_right, hidden_activation=plist["hidden_activation"], output_activation=plist["output_activation"],
                  loss_fn = plist["loss_fn"], op_folder=folder)




def trainCorrNet(src_folder, sct_folder, tgt_folder, batch_size = 20, training_epochs=40, use_valid=False, l_rate=0.01, optimization="sgd", tied=False, n_visible_left=None, n_visible_right=None, n_hidden=None, n_hidden2=None, lamda=5, W_left=None, W_right=None, b_left=None, b_right=None, W_left_prime=None, W_right_prime=None, b_prime_left=None, b_prime_right=None, W_left2=None, W_right2=None, b2=None, W_left_prime2=None, W_right_prime2=None, b_prime_left2=None, b_prime_right2=None, hidden_activation="sigmoid", output_activation="sigmoid", loss_fn = "squarrederror"):

    index = T.lscalar()
    x_left = T.matrix('x_left')
    x_right = T.matrix('x_right')

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    model = DeepCorrNet1()
    model.init(numpy_rng=rng, theano_rng=theano_rng, l_rate=l_rate, optimization=optimization, tied=tied, n_visible_left=n_visible_left, n_visible_right=n_visible_right, n_hidden=n_hidden,n_hidden2=n_hidden2, lamda=lamda, W_left=W_left, W_right=W_right, b_left=b_left, b_right=b_right, W_left_prime=W_left_prime, W_right_prime=W_right_prime, b_prime_left=b_prime_left, b_prime_right=b_prime_right, W_left2=W_left2, W_right2=W_right2, b2=b2, W_left_prime2=W_left_prime2, W_right_prime2=W_right_prime2, b_prime_left2=b_prime_left2, b_prime_right2=b_prime_right2, input_left=x_left, input_right=x_right, hidden_activation=hidden_activation, output_activation=output_activation, loss_fn =loss_fn, op_folder=tgt_folder)
    #model.load(tgt_folder,x_left,x_right)
    start_time = time.clock()
    train_set_x_left = theano.shared(numpy.asarray(numpy.zeros((1000,n_visible_left)), dtype=theano.config.floatX), borrow=True)
    train_set_x_right = theano.shared(numpy.asarray(numpy.zeros((1000,n_visible_right)), dtype=theano.config.floatX), borrow=True)

    common_cost, common_updates = model.train_common("1111")
    mtrain_common = theano.function([index], common_cost,updates=common_updates,givens=[(x_left, train_set_x_left[index * batch_size:(index + 1) * batch_size]),(x_right, train_set_x_right[index * batch_size:(index + 1) * batch_size])])

    """left_cost, left_updates = model.train_left()
    mtrain_left = theano.function([index], left_cost,updates=left_updates,givens=[(x_left, train_set_x_left[index * batch_size:(index + 1) * batch_size])])

    right_cost, right_updates = model.train_right()
    mtrain_right = theano.function([index], right_cost,updates=right_updates,givens=[(x_right, train_set_x_right[index * batch_size:(index + 1) * batch_size])])"""


    diff = 0
    flag = 1
    detfile = open(tgt_folder+"details.txt","w")
    detfile.close()
    oldtc = float("inf")

    for epoch in xrange(training_epochs):

        print "in epoch ", epoch
        c = []
        ipfile = open(src_folder+"train/ip.txt","r")
        for line in ipfile:
            next = line.strip().split(",")
            if(next[0]=="xy"):
                if(next[1]=="dense"):
                    denseTheanoloader(next[2]+"_left",train_set_x_left,"float32")
                    denseTheanoloader(next[2]+"_right",train_set_x_right, "float32")
                else:
                    sparseTheanoloader(next[2]+"_left",train_set_x_left,"float32",1000,n_visible_left)
                    sparseTheanoloader(next[2]+"_right",train_set_x_right, "float32", 1000, n_visible_right)
                for batch_index in range(0,int(next[3])/batch_size):
                    c.append(mtrain_common(batch_index))


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
        detfile.write("train\t"+str(diff)+"\n")
        detfile.close()
        # save the parameters for every 5 epochs
        if((epoch+1)%5==0):
            model.save_matrices()

    end_time = time.clock()
    training_time = (end_time - start_time)
    print ' code ran for %.2fm' % (training_time / 60.)
    model.save_matrices()
