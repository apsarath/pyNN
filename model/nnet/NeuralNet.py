__author__ = 'sanbilp'

from pyNN import *

class NeuralNet(Model):

    def init(self, numpy_rng, theano_rng=None, l_rate=None, optimization = "sgd", n_input=400, n_hidden=200, n_output=10, W=None, bhid=None, bvis=None, W_prime = None, input=None, output=None, hidden_activation = "sigmoid", output_activation = "identity", loss_fn = "squarrederror", op_folder=None):

        Model.init(self, numpy_rng, theano_rng, optimization, l_rate, op_folder)

        self.n_input = n_input
        self.hparams["n_input"] = self.n_input

        self.n_hidden = n_hidden
        self.hparams["n_hidden"] = self.n_hidden

        self.n_output = n_output
        self.hparams["n_output"] = self.n_output

        self.hidden_activation = hidden_activation
        self.hparams["hidden_activation"] = self.hidden_activation

        self.output_activation = output_activation
        self.hparams["output_activation"] = self.output_activation

        self.loss_fn = loss_fn
        self.hparams["loss_fn"] = self.loss_fn

        if self.hidden_activation == "sigmoid":
            self.W = self.Initializer.fan_based_sigmoid("W", W, n_input, n_hidden)
            self.add_params("W", self.W, n_input, n_hidden)
        else:
            self.W = self.Initializer.fan_based("W", W, n_input, n_hidden)
            self.add_params("W", self.W, n_input, n_hidden)

        self.W_prime = self.Initializer.fan_based_sigmoid("W_prime", W_prime, n_hidden, n_output)
        self.add_params("W_prime", self.W_prime, n_hidden, n_output)

        self.b = self.Initializer.zero_vector("b", bhid, n_hidden)
        self.add_params("b", self.b, 1, n_hidden)

        self.b_prime = self.Initializer.zero_vector("b_prime", bvis, n_output)
        self.add_params("b_prime", self.b_prime, 1, n_output)

        if input == None:
            self.x = T.dmatrix(name='input')
        else:
            self.x = input

        if output == None:
            self.y = T.dmatrix(name='output')
        else:
            self.y = output

        self.save_params()

        print self.get_lr_rate()

        y = T.dot(self.x, self.W) + self.b
        y = activation(y, self.hidden_activation)
        z = T.dot(y, self.W_prime) + self.b_prime
        z = activation(z, self.output_activation)
        L = loss(z, self.y, self.loss_fn)

        cost = T.mean(L)
        updates = []
        gradients = T.grad(cost, self.params)
        for p,g,n in zip(self.params, gradients, self.param_names):
            gr, upd = self.optimizer.get_grad_update(n,g)
            updates.append((p,p+gr))
            updates.extend(upd)

        self.train = theano.function( inputs = [self.x, self.y],
                                      outputs = [cost],
                                      updates=updates)

        self.test = theano.function( inputs = [self.x],
                                     outputs = [z])


    def load(self, folder, input=None):

        plist = pickle.load(open(folder+"params.pck","rb"))

        self.init(plist["numpy_rng"], theano_rng=plist["theano_rng"], l_rate=plist["l_rate"], optimization=plist["optimization"],
                  tied=plist["tied"], n_input=plist["n_input"],n_hidden=plist["n_hidden"], W=folder+"W", bhid=folder+"b", W_prime=folder+"W_prime",
                  bvis=folder+"b_prime", input=input, hidden_activation=plist["hidden_activation"], output_activation=plist["output_activation"],
                  loss_fn = plist["loss_fn"], op_folder=folder)


#test case
rng = numpy.random.RandomState(123)
theano_rng = RandomStreams(rng.randint(2 ** 30))

model = NeuralNet()
model.init(rng, theano_rng, 0.01, "sgd", 10, 5, 2, op_folder="../../../../watson_data/output/test/")

a = numpy.arange(50).reshape((5,10))
a = a+0.0

print model.test(a)