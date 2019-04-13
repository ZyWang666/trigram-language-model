#!/bin/python

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import collections
from math import log
import sys

# Python 3 backwards compatibility tricks
if sys.version_info.major > 2:

	def xrange(*args, **kwargs):
		return iter(range(*args, **kwargs))

	def unicode(*args, **kwargs):
		return str(*args, **kwargs)

class LangModel:
	def fit_corpus(self, corpus):
		"""Learn the language model for the whole corpus.

		The corpus consists of a list of sentences."""
		for s in corpus:
			self.fit_sentence(s)
		self.norm()

	def perplexity(self, corpus):
		"""Computes the perplexity of the corpus by the model.

		Assumes the model uses an EOS symbol at the end of each sentence.
		"""
		return pow(2.0, self.entropy(corpus))

	def entropy(self, corpus):
		num_words = 0.0
		sum_logprob = 0.0
		for s in corpus:
			num_words += len(s) + 1 # for EOS
			sum_logprob += self.logprob_sentence(s)
		return -(1.0/num_words)*(sum_logprob)

	def logprob_sentence(self, sentence):
		p = 0.0
		for i in xrange(len(sentence)):
			p += self.cond_logprob(sentence[i], sentence[:i])
		p += self.cond_logprob('END_OF_SENTENCE', sentence)
		return p

	# required, update the model when a sentence is observed
	def fit_sentence(self, sentence): pass
	# optional, if there are any post-training steps (such as normalizing probabilities)
	def norm(self): pass
	# required, return the log2 of the conditional prob of word, given previous words
	def cond_logprob(self, word, previous): pass
	# required, the list of words the language model suports (including EOS)
	def vocab(self): pass

class Unigram(LangModel):
	def __init__(self, backoff = 0.000001):
		self.model = dict()
		self.lbackoff = log(backoff, 2)

	def inc_word(self, w):
		if w in self.model:
			self.model[w] += 1.0
		else:
			self.model[w] = 1.0

	def fit_sentence(self, sentence):
		for w in sentence:
			self.inc_word(w)
		self.inc_word('END_OF_SENTENCE')

	def norm(self):
		"""Normalize and convert to log2-probs."""
		tot = 0.0
		for word in self.model:
			tot += self.model[word]
		ltot = log(tot, 2)
		for word in self.model:
			self.model[word] = log(self.model[word], 2) - ltot

	def cond_logprob(self, word, previous):
		if word in self.model:
			return self.model[word]
		else:
			return self.lbackoff

	def vocab(self):
		return self.model.keys()

class Trigram(LangModel):
	def __init__(self):
		self.one_word = dict()
		self.two_words = dict()
		self.three_words = dict()
		self.rare_words = dict()

	def fit_corpus(self, corpus):
		#check for rare words
		words_count = dict()
		for s in corpus:
			for w in s:
				if w in words_count:
					words_count[w] += 1
				else:
					words_count[w] = 1
		print("finish words_count: ", len(words_count))
		for w in words_count:
			if words_count[w] == 1:
				self.rare_words[w] = 1
		print("finish rare words: ", len(self.rare_words))

		"""Learn the language model for the whole corpus.
		The corpus consists of a list of sentences."""
		for s in corpus:
			self.fit_sentence(s)
		self.norm()

	def fit_sentence(self, sentence):
		f_prev = '*'
		s_prev = '*'
		comb = ''
		for w in sentence:
			if w in self.rare_words:
				w = 'UNK'
			self.words_count(1,w)
			comb = f_prev + ' ' + s_prev
			self.words_count(2,comb)
			comb = comb + ' ' + w
			self.words_count(3,comb)
			f_prev = s_prev
			s_prev = w

		w = 'END_OF_SENTENCE'
		self.words_count(1,w)
		comb = f_prev + ' ' + s_prev
		self.words_count(2,comb)
		comb = comb + ' ' + w
		self.words_count(3,comb)
		
	def vocab(self):
		return self.one_word
	
	def cond_logprob(self, word, previous):
		f_prev = '*'
		s_prev = '*'
		if word not in self.one_word:
			word = 'UNK'
		if len(previous) == 1:
			if previous[0] not in self.one_word:
				previous[0] = 'UNK'
			s_prev = previous[0]
		elif len(previous) >= 2:
			if previous[len(previous)-2] not in self.one_word:
				previous[len(previous)-2] = 'UNK'
			if previous[len(previous)-1] not in self.one_word:
				previous[len(previous)-1] = 'UNK'

			f_prev = previous[len(previous)-2]
			s_prev = previous[len(previous)-1]
		numerator = 1.0
		denumerator = len(self.one_word) + 0.0
		comb = f_prev + ' ' + s_prev
		#print("2: ", comb)
		if comb in self.two_words:
			denumerator += self.two_words[comb]
		comb = comb + ' ' + word
		#print("3: ", comb)
		if comb in self.three_words:
			numerator += self.three_words[comb]
		return log(numerator, 2) - log(denumerator, 2)
	

	def words_count(self, num, comb):
		if num == 1:
			if comb in self.one_word:
				self.one_word[comb] += 1
			else:
				self.one_word[comb] = 1
		elif num == 2:
			if comb in self.two_words:
				self.two_words[comb] += 1
			else:
				self.two_words[comb] = 1
		elif num == 3:
			if comb in self.three_words:
				self.three_words[comb] += 1
			else:
				self.three_words[comb] = 1


