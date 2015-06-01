import time
from pyNN.model.recurrent.recnet import *


def trainrnet(src_folder, sct_folder, tgt_folder,  ne=10, n_hidden=5, n_out=10, embedding=None,  W_out=None, W_r = None, bhid=None, h0=None, bout=None,  learning_rate = 0.1, training_epochs = 40, hidden_activation = "sigmoid", output_activation = "softmax",use_valid=False, optimization = "sgd"):


	rng = numpy.random.RandomState(123)
	theano_rng = RandomStreams(rng.randint(2 ** 30))
	recnet = RecurrentNet( numpy_rng = rng, l_rate=learning_rate, theano_rng=theano_rng, ne=ne, n_hidden=n_hidden, n_out=n_out, embedding=embedding,  W_out=W_out, W_r = W_r, bhid=bhid, h0=h0, bout=bout, hidden_activation = hidden_activation, output_activation = output_activation, optimization = optimization)
	start_time = time.clock()

	diff = 0
	flag = 1
	detfile = open(tgt_folder+"details.txt","w")
	detfile.close()
	oldtc = float("inf")

	for epoch in xrange(training_epochs):

		print "in epoch ", epoch
		c = []
		ipfile = open(sct_folder+"itrain.txt","r")
		count = 0
		for line in ipfile:
			#print "hi"
			if(count%100 == 0):
				print str(count) + '\r'
			count+=1
			iseq = line.strip().split()
			iseq = numpy.asarray([iseq],dtype="int32")
			nll = recnet.train(iseq)
			c.append(nll)

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
		detfile.write("train "+str(diff)+"\n")
		detfile.close()
		# save the parameters for every 2 epochs
		if((epoch+1)%1==0):
			recnet.save_matrices(tgt_folder)


		if(use_valid==True):

			print "validating"

			tc = []

			ipfile = open(sct_folder+"ivalid.txt","r")
			for line in ipfile:
				iseq = line.strip().split()
				iseq = numpy.asarray([iseq],dtype="int32")
				nll = recnet.valid(iseq)
				tc.append(nll)

			cur_tc = numpy.mean(tc)

			detfile = open(tgt_folder+"details.txt","a")
			detfile.write("valid "+str(cur_tc)+"\n")
			detfile.close()

			print cur_tc
			if(cur_tc < oldtc ):
				oldtc = cur_tc
			else:
				oldtc = cur_tc
				m = recnet.get_lr_rate() * 0.5
				recnet.set_lr_rate(m)
				print "updated lrate"



	end_time = time.clock()

	training_time = (end_time - start_time)

	print ' code ran for %.2fm' % (training_time / 60.)
	recnet.save_matrices(tgt_folder)




rng = numpy.random.RandomState(123)
theano_rng = RandomStreams(rng.randint(2 ** 30))
da = RecurrentNet( numpy_rng = rng, l_rate=0.1, theano_rng=theano_rng, ne=10, n_hidden=10, n_out=10, embedding=None, W_out=None, W_r = None, bhid=None, h0=None, bout=None, hidden_activation = "sigmoid", output_activation = "softmax",optimization = "sgd")

print da.train(numpy.asarray([[1,2,3,4]],dtype='int32'))
print da.valid([[1,2,3,4]])
print da.train(numpy.asarray([[1,2,3,4]],dtype='int32'))
print da.valid([[1,2,3,4]])
print da.train(numpy.asarray([[1,2,3,4]],dtype='int32'))
print da.valid([[1,2,3,4]])
print da.train(numpy.asarray([[1,2,3,4]],dtype='int32'))
print da.valid([[1,2,3,4]])
