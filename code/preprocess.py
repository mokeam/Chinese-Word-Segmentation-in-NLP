#!/usr/local/bin/python
# -*- coding: utf8 -*-
import re
import numpy as np
import six
import sys
import getopt
import locale
from argparse import ArgumentParser

def parse_args():
	parser = ArgumentParser()
	parser.add_argument("original_file_path", help="The path of the original file")
	parser.add_argument("input_file_path", help="The path of the input file with no spaces")
	parser.add_argument("label_file_path", help="The path of the label file in BIES format")
	return parser.parse_args()

def generate_ngram(string,n):
	"""
	This is the function to generate n-grams.
	
	:param string: The string to be splitted.
	:param n: The number of gram
	:return: A string of ngrams
	:usage: ("ABCD",1) -> "A B C D" or ("ABCD",2) -> "AB BC CD"
	"""

	ans = ''
	for i in range(len(string) - n + 1):
		ans += string[i:i+n]
		ans += '  '
	return ans

def get_ngrams(input_file_path):
	"""
	This is the function that writes ngrams to file.
	
	:param input_file_path: The path to file that contains strings to be converted to ngrams
	:return: unigram file
	"""
	
	corpora = open(input_file_path, 'r', encoding='utf8')
	unigram_input = open('unigram.utf8', 'w', encoding='utf8')
	all_lines = corpora.readlines()
	all_lines = [line.replace(' ', '')[0:-1] for line in all_lines]
	for line in all_lines:
		unigram_input.write(generate_ngram(line,1))
		if line != all_lines[-1]:
			unigram_input.write('\n')
		else:
			pass
	corpora.close()
	unigram_input.close()
	print("Unigram Generated!")

def convert_to_bies(string):
	"""
	This is the function to encode labels in BIES format.
	
	:param string: The labels to be encoded
	:return: Encoded labels in BIES format
	:usage: ("共同 创造 美好 的 新 世纪 ——") -> "BEBEBESSBEBE"
	"""

	features = []
	for word in string.split():
		for c in word:
			feature = ""
			len_word = len(word)
			if len_word == 1:
				feature += "S"
			else:
				feature += "B"
				for i_ in range(len_word - 2):
					feature += "I"
				feature += "E"
		features.append(feature)
	results = ''.join(str(e) for e in features)
	return results

def preprocess(original_file,input_file,label_file):
	"""
	This is the function that read the original training file.
 
	:param string: The labels to be encoded
	:return: Input file with no spaces and label file in BIES format
	"""

	with open(original_file, 'r', encoding='utf8') as f:
		original_lines = f.readlines()
		original_lines = list(filter(lambda x: x.strip(),original_lines))
	# remove spacess
	lines = [re.sub(r'\s(?=[^A-z0-9])','',line) for line in original_lines]
	lines = [line.replace(" ","")for line in lines]
	lines = [line.replace("\u3000","")for line in lines]
	
	# finally, write lines in the file
	with open(input_file, 'w') as f:
		f.writelines(lines)

	# finally, write labels in the file
	label_lines = [convert_to_bies(label) for label in original_lines]
	with open(label_file, 'w') as f:
		for item in label_lines:
			f.write("%s" % item)
			if item != label_lines[-1]:
				f.write("\n")
	print("Input and Label files generated!")

if __name__ == '__main__':
	args = parse_args()
	preprocess(args.original_file_path, args.input_file_path, args.label_file_path)
	get_ngrams(args.input_file_path)