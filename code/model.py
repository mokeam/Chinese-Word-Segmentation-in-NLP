import os
import collections
import tensorflow as tf
import numpy as np
import pickle
from typing import Tuple, List, Dict
from tensorflow.keras.callbacks import TensorBoard,ModelCheckpoint
from collections import Counter
from tensorflow.keras.layers import Dense, Input,Masking,LSTM, Embedding,Reshape, Dropout, Activation,TimeDistributed,Bidirectional
from tensorflow.keras.models import Model,Sequential,load_model
from tensorflow.keras import initializers, regularizers, constraints, optimizers, layers
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical 
from tensorflow.keras import backend as K

unigram_path = '../resources/as_cityu_msr_pku_unigram.utf8'
X_test_path = 'pku_input.utf8'
unigram_vocab = dict()
unigram_word_to_id = dict()
X_test_uni = []

def vocabulary(unigram_path):
	"""
	This is the function to build the vocabulary of the dataset.
	
	:param unigram_path: The path to the file that contains the unigrams
	:return: None
	"""
	with open(unigram_path, 'r', encoding='utf8') as f:
	  original_lines = f.readlines()
	  for line in original_lines:
	  	words = line.split()
	  	for word in words:
	  		if word not in unigram_vocab:
	  			unigram_vocab[word] = 1
	  		else:
	  			unigram_vocab[word] += 1

def word2index():
	"""
	Converts each character to its index in the vocabulary
	
	:return: None
	"""
	vocabulary(unigram_path)
	unigram_word_to_id["<PAD>"] = 0 #zero is not casual!
	unigram_word_to_id["<UNK>"] = 1 #OOV are mapped as <UNK>
	unigram_word_to_id.update({k:v+len(unigram_word_to_id) for k, v in unigram_vocab.items()})

def tokenize_dataset(X_test_path):
	word2index()
	with open(X_test_path, 'r', encoding='utf8') as f:
	  original_lines = f.readlines()
	  original_lines = [line.replace("\u3000","")for line in original_lines]
	  for line in original_lines:
	  	words = line.split()
	  	for word in words:
	  		char = []
	  		for c in word:
	  			try:
	  				char.append(unigram_word_to_id[c])
	  			except KeyError:
	  				char.append(unigram_word_to_id["<UNK>"])
	  	X_test_uni.append(char)

def precision(y_true, y_pred):
	"""Precision metric.
	Only computes a batch-wise average of precision.
	Computes the precision, a metric for multi-label classification of
	how many selected items are relevant.
	"""
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
	precision = true_positives / (predicted_positives + K.epsilon())
	return precision

def convert_integer_to_label(string):
	"""
	Converts the encoded integer labels from integers to BIES format
	:param string: integer to be converted to BIES format
	:return: Array of labels in BIES format+
	"""
	tags = []
	for word in string:
		if word == '3':
			tags.append('S')  # 'S', a Single character
		if word == '0':
				tags.append('B')  # 'B', Begin of a word
		if word == '1':
			tags.append('I')  # 'I', Middle of a word
		if word == '2':
			tags.append('E')  # 'E',  End of a word
	return tags

def getlabel(array):
	"""
	Get the BIES format of an array
	:param array: The encoded integer labels to be converted to BIES format
	"""
	result = []
	for i in array:
		string = ""
		for digit in i:
			string += str(digit)
		result.append(convert_integer_to_label(string))
	return result

def predict_model(input_path,output_path,model_path):
	tokenize_dataset(input_path)
	model = load_model(model_path,custom_objects={"precision": precision})

	y_pred = [None]*len(X_test_uni)
	for i in range(len(X_test_uni)):
		this_pred = model.predict(X_test_uni[i])    
		y_pred[i] = this_pred

	Y = [None]*len(y_pred)
	for i in range(len(y_pred)):
	  Y[i] = y_pred[i].argmax(axis=-1)

	A = getlabel(Y)

	with open(output_path, 'w') as f:
		for item in A:
			line = ("".join("%s"%a for a in item))
			f.write("%s\n" %line)
		print("BIES Predictions Saved at "+output_path)