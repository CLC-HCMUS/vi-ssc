from __future__ import division
import sys
import os
import time
import numpy as np
from datetime import datetime
from tattention_initial_stb import *
import math
import io
import itertools
import nltk.data
import sys


SENTENCE_START_TOKEN = 'eos'
UNKNOWN_TOKEN = 'unk'

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
    sent = "%s %s" % (SENTENCE_START_TOKEN, lowersent(sent))
    words_in_sent = sent.split()
    i = len(words_in_sent)
    array_sent=[0]*i
    for j in range(i):
        if words_in_sent[j] not in word2index_dict.keys():
            words_in_sent[j] = UNKNOWN_TOKEN
        array_sent[j] = (word2index_dict[words_in_sent[j]])
    return array_sent

def label_compress(sent, comp):
    '''
    Label compressed of sentence
    :param sent: original sentence
    :param comp: compressed sentence
    :return: list of label (0 or 1) of each word of sentence
    '''
    sent = "%s %s" % (SENTENCE_START_TOKEN, lowersent(sent))
    words_in_sent = sent.split()
    comp = "%s %s %s" % (SENTENCE_START_TOKEN, lowersent(comp), '<endofstring>')
    words_in_comp = comp.split()
    i = len(words_in_sent)
    l= [1]*i
    for k in range(i):
    	if not words_in_sent[k] == words_in_comp[k]:
    		words_in_comp.insert(k, 'xxx')
    for j in range(i):
        if words_in_comp[j] == 'xxx':
            l[j] = 0
    l[0]=2
    return (l,i)

def load_data_from_file(path_to_3reftxt, test_split, vocabulary_size):
    '''
    Load data for training and testing from json file
    :param path_to_json: path to json file
    :param word2vec_dict: dictionary of word2vec
    :return: X_train, y_train, X_test, y_test
    '''
    X=[]
    y=[]
    len_sent_array=[]
    dict_id_ori_com123 = read3reftxt(path_to_3reftxt)
    print 'Data %d sentences'%(len(dict_id_ori_com123)*3)
    count=0
    original_sentence_array=[]
    compression_sentence_array=[]
    original_arr, compr1_arr, compr2_arr, compr3_arr=[i[0] for i in dict_id_ori_com123.values()], [i[1] for i in dict_id_ori_com123.values()], [i[2] for i in dict_id_ori_com123.values()], [i[3] for i in dict_id_ori_com123.values()]
    word_to_index_dict, index_to_word_dict = word2index(original_arr, vocabulary_size)
    for k in range(len(original_arr)):
        arr_sent_vect = word2vec(original_arr[k], word_to_index_dict)
        X.append(arr_sent_vect)
        X.append(arr_sent_vect)
        X.append(arr_sent_vect)
        y_l,l = label_compress(original_arr[k], compr1_arr[k])
        y.append(y_l)
        y_l,l = label_compress(original_arr[k], compr2_arr[k])
        y.append(y_l)
        y_l,l = label_compress(original_arr[k], compr3_arr[k])
        y.append(y_l)
        len_sent_array.append(l)
        count+=1
        if count%100==0:
            sys.stdout.write('.')
        #get text array:
        original_sentence_array.append(original_arr[k])
        original_sentence_array.append(original_arr[k])
        original_sentence_array.append(original_arr[k])
        compression_sentence_array.append(compr1_arr[k])
        compression_sentence_array.append(compr2_arr[k])
        compression_sentence_array.append(compr3_arr[k])
    return ((X[int(len(X)*test_split):],y[int(len(y)*test_split):], len_sent_array[int(len(len_sent_array)*test_split):]), (X[:int(len(X)*test_split)], y[:int(len(y)*test_split)], len_sent_array[:int(len(len_sent_array)*test_split)]), (original_sentence_array[int(len(X)*test_split):], compression_sentence_array[int(len(X)*test_split):]), (original_sentence_array[:int(len(X)*test_split)], compression_sentence_array[:int(len(X)*test_split)]))

def testing(model, X_test):
    predict_y_test=[]
    for i in range(len(X_test)):
        predict_test=model.predict_class(X_test[i])
        predict_y_test.append(predict_test)
    return predict_y_test

def early_stop_flag(m, X_v, y_v, f1_p, original_sentence_test_text, compression_sentence_test_text):
  predict_v = testing(m, X_v)
  f1 = compute_f1(y_v, predict_v, original_sentence_test_text, compression_sentence_test_text)[3]
  if f1>f1_p:
    return (True,f1)
  else:
    return (False,f1_p)

def compute_f1(y_test, predict_test, original_sentence_test_text, compression_sentence_test_text):
    f1=[0,0,0,0,0,[]]
    num_error=0
    num_f1_0 = 0
    for i in range(len(y_test)):
        num_True = 0
        num_predict = 0
        num_test = 0
        for y in range(1,len(y_test[i])):
            if y_test[i][y] == predict_test[i][y] and y_test[i][y] == 1:
                num_True+=1
            if y_test[i][y] ==1:
                num_test+=1
            if predict_test[i][y] ==1:
                num_predict+=1
        f1[0]+=num_True
        f1[1]+=num_predict
        f1[2]+=num_test
        if num_True ==0:
            f1_sent = 0
            num_f1_0+=1
        else:
            precision = num_True/num_predict
            recall = num_True/num_test
            f1_sent= 2*(precision*recall)/(precision+recall)
        if math.isnan(f1_sent):
            f1_sent=0
            num_error+=1
        f1[3]+=f1_sent

        f1[4]+=len(y_test[i])
        f1[5].append((str(i), original_sentence_test_text[i], compression_sentence_test_text[i], str(y_test[i]), str(predict_test[i]), str(f1_sent)))
    f1[3] = f1[3] / (len(y_test) - num_error)
    if (len(y_test)-num_error) ==0:
        print ('----- All predicts are zero -----')
    else:
        print ('Total: %d - Error sentences: %d = %d'%(len(y_test), num_error, len(y_test)-num_error))
        print ('No. sent F1 is 0: %d'%num_f1_0)
        print (f1[3])
        return f1

def main(dataset):
	LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "0.001"))
	VOCABULARY_SIZE = int(os.environ.get("VOCABULARY_SIZE", "3500"))
	EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "50"))
	HIDDEN_DIM = int(os.environ.get("HIDDEN_DIM", "100"))
	NEPOCH = int(os.environ.get("NEPOCH", "20"))
	MODEL_OUTPUT_FILE = os.environ.get("MODEL_OUTPUT_FILE")
	INPUT_DATA_FILE = os.environ.get("INPUT_DATA_FILE", dataset)
	PRINT_EVERY = int(os.environ.get("PRINT_EVERY", "500"))

	title = 'tattention-Vietnamese-initial-stb-1layer-earlystop-GRU'
	ts = datetime.now().strftime("%Y-%m-%d-%H-%M")
	if not MODEL_OUTPUT_FILE:
	  MODEL_OUTPUT_FILE = "%s-%s-%s-%s-%s.dat" % (title, ts, VOCABULARY_SIZE, EMBEDDING_DIM, HIDDEN_DIM)

	# Load data
	#x_train, y_train, word_to_index, index_to_word = load_data(INPUT_DATA_FILE, VOCABULARY_SIZE)
	test_split = 0.1
	(X_train, y_train, len_sent_train), (X_test, y_test, len_sent_test), (original_sentence_train_text, compression_sentence_train_text), (original_sentence_test_text, compression_sentence_test_text) = load_data_from_file(INPUT_DATA_FILE, test_split, VOCABULARY_SIZE)

	# Build model
	print '\nBuild model'
	model = GRUTheano(VOCABULARY_SIZE, hidden_dim=HIDDEN_DIM, bptt_truncate=-1)
	model_best = GRUTheano(VOCABULARY_SIZE, hidden_dim=HIDDEN_DIM, bptt_truncate=-1)
	#model = load_model_parameters_theano('GRU-2016-08-05-13-48-2000-50-100.dat.npz')

	#Print SGD step time
	print title
	t1 = time.time()
	model.sgd_step(X_train[10], y_train[10], LEARNING_RATE)
	c = model.ce_error(X_train[10], y_train[10])
	print ('loss x[10]: %f'%c)
	t2 = time.time()
	print "SGD Step time: %f milliseconds" % ((t2 - t1) * 1000.)
	sys.stdout.flush()

	# We do this every few examples to understand what's going on
	def sgd_callback(model, num_examples_seen):
	  dt = datetime.now().isoformat()
	  loss = model.calculate_loss(X_train[:PRINT_EVERY], y_train[:PRINT_EVERY])
	  print("\n%s (%d)" % (dt, num_examples_seen))
	  print("--------------------------------------------------")
	  print("Loss: %f" % loss)
	  # save_model_parameters_theano(model, MODEL_OUTPUT_FILE)
	  print("\n")
	  sys.stdout.flush()

	t3 = time.time()
	f1_prev=0
	no_epoch_es=0
	for epoch in range(NEPOCH):
	  print('Epoch %d: ' % epoch)
	  train_with_sgd(model, X_train, y_train, learning_rate=LEARNING_RATE, nepoch=1, decay=0.9, callback_every=PRINT_EVERY, callback=sgd_callback)
	  es_flag, f1_prev = early_stop_flag(model, X_test, y_test, f1_prev,original_sentence_test_text, compression_sentence_test_text)
	  if es_flag == False:
	    no_epoch_es += 1
	    if no_epoch_es > 4:
	      break
	  else:
	    model_best = model
	    save_model_parameters_theano(model_best, MODEL_OUTPUT_FILE)
	    no_epoch_es = 0

	t4 = time.time()
	print "SGD Train time: %f" % ((t4 - t3))
	sys.stdout.flush()
	#
	print 'Testing...'
	predict_test = testing(model_best, X_test)
	np.save("%s.predict" % (MODEL_OUTPUT_FILE), predict_test)
	print 'Compute f1:...'
	f1 = compute_f1(y_test, predict_test, original_sentence_test_text, compression_sentence_test_text)
	with io.open('./Output-' + title, 'w', encoding='utf8') as f:
		for if1 in f1[5]:
			f.write('\n'.join([i for i in if1]))
			f.write(u'\n')

if __name__ == '__main__':
    if not len(sys.argv) == 2:
        print 'Useage: python sentencecompessionVietnamese.py <dataset>'
    else:
	   main(str(sys.argv[1]))

