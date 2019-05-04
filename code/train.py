import os
from typing import Tuple, List, Dict
import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import TensorBoard,ModelCheckpoint,EarlyStopping
from collections import Counter
from tensorflow.keras.layers import Dense, Input,Masking,LSTM, Embedding,Reshape, Dropout, Activation,TimeDistributed,Bidirectional,concatenate, GlobalMaxPool1D
from tensorflow.keras.models import Model,Sequential,load_model
from tensorflow.keras import initializers, regularizers, constraints, optimizers, layers
import pickle
from tensorflow.keras.optimizers import SGD
from  tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical 
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import collections
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K

unigram_path = '../resources/as_cityu_msr_pku_unigram.utf8'
X_train_path = '../resources/as_cityu_msr_pku_input.utf8'
Y_train_path = '../resources/as_cityu_msr_pku_label.utf8'

unigram_vocab = dict()
unigram_word_to_id = dict()
X_train_uni = []
Y_train = []


def vocabulary(unigram_path=unigram_path ):
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
	vocabulary()
	unigram_word_to_id["<PAD>"] = 0 #zero is not casual!
	unigram_word_to_id["<UNK>"] = 1 #OOV are mapped as <UNK>
	unigram_word_to_id.update({k:v+len(unigram_word_to_id) for k, v in unigram_vocab.items()})

def tokenize_dataset(X_train_path=X_train_path):
	"""
	Converts each character to its index in the vocabulary

	:param X_train_path: path to the trainig set with no spaces 
	:return: encoded X  training set
	"""
	word2index()
	with open(X_train_path, 'r', encoding='utf8') as f:
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
	  	X_train_uni.append(char)
	return X_train_uni

def convert_labels_to_integer(string):
	"""
	Converts the labels from BIES format to integer
	:param string: integer to be converted to BIES format
	:return: Array of labels in Integer
	"""
	tags = []
	for words in string.split():
		for word in words:
			if word == 'S':
				tags.append(3)  # 'S', a Single character
			if word == 'B':
					tags.append(0)  # 'B', Begin of a word
			if word == 'I':
				tags.append(1)  # 'I', Middle of a word
			if word == 'E':
				tags.append(2)  # 'E',  End of a word
	return tags


def encode_y(Y_train_path=Y_train_path):
	"""
	Encodes the labels
	:param Y_train_path: Path to labels in BIES format
	:return: Array of one hot encoded training labels
	"""
	#Training Labels
	with open(Y_train_path, 'r', encoding='utf8') as f:
			label_original_lines = f.readlines()
	Y_tra = [convert_labels_to_integer(label) for label in label_original_lines]

	#One Hot Encoding of Training Labels
	for y in Y_tra:
		Y_train.append(to_categorical(y,num_classes=4))
	return Y_train

def pad_data(X_train_path,Y_train_path):
	"""
	Pad training set sequences
	:param X_train_path: Path to X encoded
	:param Y_train_path: Path to Y encoded
	:return: padded training sets
	"""
	max_len = (sum(len(line) for line in X_train_uni) / len(X_train_uni))
	MAX_LEN = round(max_len)+1

	train_x_uni_padded = pad_sequences(X_train_uni,padding='post', maxlen=MAX_LEN)
	train_y_padded = pad_sequences(Y_train,padding='post', maxlen=MAX_LEN)

	return train_x_uni_padded,train_y_padded


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

def bilstm_model():
	"""
	Bilstm model
	:return: model
	"""
	LEN = 2000000
	visible = Input(shape=(None,))
	em = Embedding(LEN,64,input_length=None,mask_zero=True)(visible)
	hidden = Bidirectional(LSTM(256,return_sequences=True,dropout=0.6,recurrent_dropout=0.4),merge_mode='sum')(em)
	output = TimeDistributed(Dense(4,activation='softmax'))(hidden)
	model = Model(inputs=visible, outputs=output)
	model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.035, momentum=0.95), metrics=['accuracy',precision])
	return model


if __name__ == '__main__':
	word2index()
	X_train_uni = tokenize_dataset()
	Y_train = encode_y()
	train_x_uni_padded,train_y_padded = pad_data(X_train_uni,Y_train)
	model = bilstm_model()
	filepath = "weights.{epoch:02d}-{val_loss:.2f}.hdf5"
	mc = ModelCheckpoint(filepath, monitor='val_precision', verbose=1, save_best_only=True, mode='max')
	print("Training")
	history = model.fit(train_x_uni_padded,train_y_padded,batch_size=256, epochs=20, verbose=1,shuffle=True,validation_split=0.2,callbacks=[mc])

	# Plot training & validation precision values
	plt.plot(history.history['precision'])
	plt.plot(history.history['val_precision'])
	plt.title('Model Precision')
	plt.ylabel('Precision')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.show()

	# Plot training & validation accuracy values
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('Model accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.show()

	# Plot training & validation loss values
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('Model loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.show()