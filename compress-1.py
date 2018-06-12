from __future__ import division
import io
import os
import sys
import nltk
import numpy as np
import theano
import theano.tensor as T
from theano.gradient import grad_clip
import time
import operator
import csv
import itertools
import array
from datetime import datetime


class GRUTheano:

	def __init__(self, word_dim, hidden_dim=100, bptt_truncate=-1):
		# Assign instance variables
		self.word_dim = word_dim
		self.hidden_dim = hidden_dim
		self.bptt_truncate = bptt_truncate
		# Initialize the network parameters
		E = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim-3, word_dim))
		Ey_encode = np.zeros(3)
		Ey_decode = np.identity(3)

		U = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (8, hidden_dim, hidden_dim))
		W = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (8, hidden_dim, hidden_dim))
		b = np.zeros((8, hidden_dim))

		U_decode = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim),
									 (4, hidden_dim, hidden_dim))
		W_decode = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim),
									 (4, hidden_dim, hidden_dim))
		b_decode = np.zeros((4, hidden_dim))
		UA = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (4, hidden_dim, hidden_dim * 2))

		V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (3, hidden_dim))
		c = np.zeros(3)
		# Theano: Created shared variables
		self.E = theano.shared(name='E', value=E.astype(theano.config.floatX))
		self.Ey_encode = theano.shared(name='Ey_encode', value=Ey_encode.astype(theano.config.floatX))
		self.Ey_decode = theano.shared(name='Ey_decode', value=Ey_decode.astype(theano.config.floatX))
		self.U = theano.shared(name='U', value=U.astype(theano.config.floatX))
		self.UA = theano.shared(name='UA', value=UA.astype(theano.config.floatX))
		self.W = theano.shared(name='W', value=W.astype(theano.config.floatX))
		self.V = theano.shared(name='V', value=V.astype(theano.config.floatX))
		self.b = theano.shared(name='b', value=b.astype(theano.config.floatX))
		self.c = theano.shared(name='c', value=c.astype(theano.config.floatX))
		self.U_decode = theano.shared(name='U_decode', value=U_decode.astype(theano.config.floatX))
		self.W_decode = theano.shared(name='W_decode', value=W_decode.astype(theano.config.floatX))
		self.b_decode = theano.shared(name='b_decode', value=b_decode.astype(theano.config.floatX))
		# SGD / rmsprop: Initialize parameters
		# self.mE = theano.shared(name='mE', value=np.zeros(E.shape).astype(theano.config.floatX))
		# self.mEy_encode = theano.shared(name='mEy_encode', value=np.zeros(Ey_encode.shape).astype(theano.config.floatX))
		# self.mEy_decode = theano.shared(name='mEy_decode', value=np.zeros(Ey_decode.shape).astype(theano.config.floatX))
		# self.mU = theano.shared(name='mU', value=np.zeros(U.shape).astype(theano.config.floatX))
		# self.mUA = theano.shared(name='mUA', value=np.zeros(UA.shape).astype(theano.config.floatX))
		# self.mV = theano.shared(name='mV', value=np.zeros(V.shape).astype(theano.config.floatX))
		# self.mW = theano.shared(name='mW', value=np.zeros(W.shape).astype(theano.config.floatX))
		# self.mb = theano.shared(name='mb', value=np.zeros(b.shape).astype(theano.config.floatX))
		# self.mc = theano.shared(name='mc', value=np.zeros(c.shape).astype(theano.config.floatX))
		# self.mU_decode = theano.shared(name='mU_decode', value=np.zeros(U_decode.shape).astype(theano.config.floatX))
		# self.mW_decode = theano.shared(name='mW_decode', value=np.zeros(W_decode.shape).astype(theano.config.floatX))
		# self.mb_decode = theano.shared(name='mb_decode', value=np.zeros(b_decode.shape).astype(theano.config.floatX))
		# We store the Theano graph here
		self.theano = {}
		self.__theano_build__()

	def __theano_build__(self):
		E, Ey_encode, Ey_decode, V, U, UA, W, b, c, U_decode, W_decode, b_decode = self.E, self.Ey_encode, self.Ey_decode, self.V, self.U, self.UA, self.W, self.b, self.c, self.U_decode, self.W_decode, self.b_decode

		x = T.ivector('x')
		y = T.ivector('y')

		def forward_prop_step_encode_backward(x_t, s_t1_prev_b, c_t1_prev_b):
			# This is how we calculated the hidden state in a simple RNN. No longer!
			# s_t = T.tanh(U[:,x_t] + W.dot(s_t1_prev))

			# Word embedding layer
			x_e_b = E[:,x_t]
			xy_e_b = theano.tensor.concatenate([x_e_b,Ey_encode], axis=0)

			#Encode   #LSTM Layer 1
			# i=z, f=r, add o,
			i_t1_b = T.nnet.hard_sigmoid(U[0].dot(xy_e_b) + W[0].dot(s_t1_prev_b) + b[0])
			f_t1_b = T.nnet.hard_sigmoid(U[1].dot(xy_e_b) + W[1].dot(s_t1_prev_b) + b[1])
			o_t1_b = T.nnet.hard_sigmoid(U[2].dot(xy_e_b) + W[2].dot(s_t1_prev_b) + b[2])
			g_t1_b = T.tanh(U[3].dot(xy_e_b) + W[3].dot(s_t1_prev_b) + b[3])
			c_t1_b = c_t1_prev_b*f_t1_b + g_t1_b*i_t1_b
			s_t1_b = T.tanh(c_t1_b)*o_t1_b

			return [s_t1_b, c_t1_b]

		def forward_prop_step_encode(x_t, s_t1_prev, c_t1_prev):
			# This is how we calculated the hidden state in a simple RNN. No longer!
			# s_t = T.tanh(U[:,x_t] + W.dot(s_t1_prev))

			# Word embedding layer
			x_e = E[:,x_t]
			xy_e = theano.tensor.concatenate([x_e,Ey_encode], axis=0)

			#Encode   #LSTM Layer 1
			# i=z, f=r, add o,
			i_t1 = T.nnet.hard_sigmoid(U[4].dot(xy_e) + W[4].dot(s_t1_prev) + b[4])
			f_t1 = T.nnet.hard_sigmoid(U[5].dot(xy_e) + W[5].dot(s_t1_prev) + b[5])
			o_t1 = T.nnet.hard_sigmoid(U[6].dot(xy_e) + W[6].dot(s_t1_prev) + b[6])
			g_t1 = T.tanh(U[7].dot(xy_e) + W[7].dot(s_t1_prev) + b[7])
			c_t1 = c_t1_prev * f_t1 + g_t1 * i_t1
			s_t1 = T.tanh(c_t1) * o_t1

			return [s_t1, c_t1]

		def forward_prop_step_decode(x_t, y_t, r_t1, s_t1_prev_d, c_t1_prev_d):
			# This is how we calculated the hidden state in a simple RNN. No longer!
			# s_t = T.tanh(U[:,x_t] + W.dot(s_t1_prev))

			# Word embedding layer
			x_e = E[:, x_t]
			y_e = Ey_decode[:,y_t]
			xy_e_d = theano.tensor.concatenate([x_e, y_e], axis=0)

			#Add s_xt to s_t_pre
			#s_t1_prev_d = s_t1_prev_d*(s_t1_matrix)

			# Decode   #LSTM Layer 1
			i_t1_d = T.nnet.hard_sigmoid(U_decode[0].dot(xy_e_d) + W_decode[0].dot(s_t1_prev_d) + UA[0].dot(r_t1) + b_decode[0])
			f_t1_d = T.nnet.hard_sigmoid(U_decode[1].dot(xy_e_d) + W_decode[1].dot(s_t1_prev_d) + UA[1].dot(r_t1) + b_decode[1])
			o_t1_d = T.nnet.hard_sigmoid(U_decode[2].dot(xy_e_d) + W_decode[2].dot(s_t1_prev_d) + UA[2].dot(r_t1) + b_decode[2])
			g_t1_d = T.tanh(U_decode[3].dot(xy_e_d) + W_decode[3].dot(s_t1_prev_d)+ UA[3].dot(r_t1) + b_decode[3])
			c_t1_d = c_t1_prev_d * f_t1_d + g_t1_d * i_t1_d
			s_t1_d = T.tanh(c_t1_d) * o_t1_d

			o = T.nnet.softmax(V.dot(s_t1_d) + c)[0]

			return [o, s_t1_d, c_t1_d]

		def forward_prop_step_decode_test(x_t, r_t1, o_t_pre_test,s_t1_prev_d_test, c_t1_prev_d_test):
			# This is how we calculated the hidden state in a simple RNN. No longer!
			# s_t = T.tanh(U[:,x_t] + W.dot(s_t1_prev))

			# Word embedding layer
			x_e = E[:, x_t]
			#y_e = Ey[:, y_t]
			xy_e_d_test = theano.tensor.concatenate([x_e, o_t_pre_test], axis=0)

			#Add s_xt to s_t_pre
			#s_t1_prev_d_test = s_t1_prev_d_test*(s_t1_matrix)

			# Decode   #LSTM Layer 1
			i_t1_d_test = T.nnet.hard_sigmoid(U_decode[0].dot(xy_e_d_test) + W_decode[0].dot(s_t1_prev_d_test) + UA[0].dot(r_t1) + b_decode[0])
			f_t1_d_test = T.nnet.hard_sigmoid(U_decode[1].dot(xy_e_d_test) + W_decode[1].dot(s_t1_prev_d_test) + UA[1].dot(r_t1)+ b_decode[1])
			o_t1_d_test = T.nnet.hard_sigmoid(U_decode[2].dot(xy_e_d_test) + W_decode[2].dot(s_t1_prev_d_test) + UA[2].dot(r_t1)+ b_decode[2])
			g_t1_d_test = T.tanh(U_decode[3].dot(xy_e_d_test) + W_decode[3].dot(s_t1_prev_d_test) + UA[3].dot(r_t1) + b_decode[3])
			c_t1_d_test = c_t1_prev_d_test * f_t1_d_test + g_t1_d_test * i_t1_d_test
			s_t1_d_test = T.tanh(c_t1_d_test) * o_t1_d_test

			o_test = T.nnet.softmax(V.dot(s_t1_d_test) + c)[0]

			return [o_test, s_t1_d_test, c_t1_d_test]

		[s_t1_b, c_t1_b], updates = theano.scan(
			forward_prop_step_encode_backward,
			sequences=x[::-1], #reverse y
			truncate_gradient=self.bptt_truncate,
			outputs_info=[dict(initial=T.zeros(self.hidden_dim)),
						  dict(initial=T.zeros(self.hidden_dim))])
		s_t1_b = s_t1_b[::-1]
		c_t1_b = c_t1_b[::-1]

		[s_t1, c_t1], updates = theano.scan(
			forward_prop_step_encode,
			sequences=x,
			truncate_gradient=self.bptt_truncate,
			outputs_info=[dict(initial=T.zeros(self.hidden_dim)),
						  dict(initial=T.zeros(self.hidden_dim))])

		s_t1_encode = theano.tensor.concatenate([s_t1, s_t1_b], axis=1)

		[o_test, s_t1_d_test, c_t1_d_test], updates = theano.scan(
			forward_prop_step_decode_test,
			sequences=[x, s_t1_encode],
			truncate_gradient=self.bptt_truncate,
			outputs_info=[dict(initial=T.zeros(3)),
						  dict(initial=s_t1_b[0]),
						  dict(initial=c_t1_b[0])]
			)


		prediction = T.argmax(o_test, axis=1)
		self.predict_class = theano.function([x], prediction)

def load_model_parameters_theano(path, modelClass=GRUTheano):
	npzfile = np.load(path)
	E, Ey_encode, Ey_decode, U, UA, W, V, b, c, U_decode, W_decode, b_decode = npzfile["E"], npzfile["Ey_encode"], npzfile["Ey_decode"], npzfile["U"],  npzfile["UA"], npzfile["W"], npzfile["V"], npzfile["b"], npzfile["c"], npzfile["U_decode"], npzfile["W_decode"], npzfile["b_decode"]
	hidden_dim, word_dim = E.shape[0] + 3, E.shape[1]
	print "Building model from %s with hidden_dim=%d word_dim=%d" % (path, hidden_dim, word_dim)
	sys.stdout.flush()
	model = modelClass(word_dim, hidden_dim=hidden_dim)
	model.E.set_value(E)
	model.Ey_encode.set_value(Ey_encode)
	model.Ey_decode.set_value(Ey_decode)
	model.U.set_value(U)
	model.UA.set_value(UA)
	model.W.set_value(W)
	model.V.set_value(V)
	model.b.set_value(b)
	model.c.set_value(c)
	model.U_decode.set_value(U_decode)
	model.W_decode.set_value(W_decode)
	model.b_decode.set_value(b_decode)
	return model

def lowersent(sent):
	return sent.lower()

def read3reftxt(path_to_3reftxt):
	'''
	input: path_to_3reftxt
	output: dictionary 
		{cluster_id: [original, compressed_1, compressed_2, compressed_3]}
	'''
	result = {}
	with io.open(path_to_3reftxt, 'r', encoding = 'utf8') as f:
		for line in f:
			line1 = line.split('ID=')[1].split('</')[0]
			cluster_id, sent = line1.split('>')
			sent = sent.strip()
			result.setdefault(cluster_id, [])
			result[cluster_id].append(lowersent(sent))

	return result

def word2index(list_original_sentence, vocabulary_size):
	SENTENCE_START_TOKEN = 'eos'
	UNKNOWN_TOKEN = 'unk'
	unknown_token = UNKNOWN_TOKEN
	sentence_start_token = SENTENCE_START_TOKEN
	# Tokenize the sentences into words
	tokenized_sentences = [lowersent(sent).split() for sent in list_original_sentence]
	# Count the word frequencies
	word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
	#print "Found %d unique words words." % len(word_freq.items())
	# Get the most common words and build index_to_word and word_to_index vectors
	vocab = word_freq.most_common(vocabulary_size - 2)
	index_to_word = [x[0] for x in vocab]
	index_to_word.append(unknown_token)
	index_to_word.append(sentence_start_token)
	#print(len(index_to_word))
	word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])
	return (word_to_index, index_to_word)

def word2vec(sent, word2index_dict):
	'''
	Word2vec of a sentence
	:param sent: input sentence
	:param word2vec_dict: dict of word2vec
	:param maxlen: max len of sentence in dataset
	:return: vector of sentence (list vector of words)
	'''
	SENTENCE_START_TOKEN = 'eos'
	UNKNOWN_TOKEN = 'unk'
	sent = "%s %s" % (SENTENCE_START_TOKEN, lowersent(sent))
	words_in_sent = sent.split()
	i = len(words_in_sent)
	array_sent=[0]*i
	for j in range(i):
		if words_in_sent[j] not in word2index_dict.keys():
			words_in_sent[j] = UNKNOWN_TOKEN
		array_sent[j] = (word2index_dict[words_in_sent[j]])
	return array_sent

def compressSentence(input, output, model, data):
	t1 = time.time()
	model_theano = load_model_parameters_theano(model)
	t2 =time.time()
	print 'loading model time: ', (t2-t1)/60.0
	#load word2vec
	dict_id_ori_com123 = read3reftxt(data)
	t3 =time.time()
	print 'loading data time: ', (t3-t2)/60.0
	original_arr =[i[0] for i in dict_id_ori_com123.values()]
	word_to_index_dict, index_to_word_dict = word2index(original_arr, 3500)
	t4 =time.time()
	print 'loading w2v time: ', (t4-t3)/60.0
	#compress cau input
	try:
		with io.open(input, 'r', encoding = 'utf8') as fin, io.open(output, 'w', encoding='utf8') as fout:
			for sent in fin:
				sent1 = lowersent(sent.strip())
				ori_sent_vec = word2vec(sent1, word_to_index_dict)
				comp_sent_vec = model_theano.predict_class(ori_sent_vec)
				comp_sent = ''
				print 'save compressed sent'
				for i, word in enumerate(sent.split()):
					if comp_sent_vec[i+1] == 1:
						comp_sent += word.replace('_', ' ') + ' '
				comp_sent = comp_sent.strip()
				fout.write(comp_sent)
				fout.write(u'\n')

		t5 =time.time()
		print 'processing time: ', (t5-t1)/60.0
	except Exception as e:
		print e

if __name__ == '__main__':
	if not len(sys.argv) == 5:
		print 'Useage: python compress-1.py <input> <output> <model> <dataset>'
	else:
		compressSentence(str(sys.argv[1]), str(sys.argv[2]), str(sys.argv[3]), str(sys.argv[4]))
