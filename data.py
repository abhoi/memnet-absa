from nltk import word_tokenize
from collections import Counter
from nltk.corpus import stopwords
import string
import math

from keras.preprocessing.text import text_to_word_sequence

import pandas as pd
import pickle
import numpy as np
import os
import csv
import xml.etree.ElementTree as ET
import html
# import HTMLParser
from html.parser import HTMLParser
import re

stop = set(stopwords.words('english'))

# aspect dictionary
aspect_dict = {}

def load_embedding_file(embed_file_name, word_set):
    ''' loads embedding file and returns a dictionary (word -> embedding) for the words existing in the word_set '''
    embeddings = {}
    with open(embed_file_name, 'r') as embed_file:
        for line in embed_file:
            content = line.strip().split()
            word = content[0]
            if word in word_set:
                embedding = np.array(content[1:], dtype=float)
                embeddings[word] = embedding
    return embeddings


def get_dataset_resources(data_file_name, sent_word2idx, target_word2idx, word_set, max_sent_len):
    ''' updates word2idx and word_set '''
    if len(sent_word2idx) == 0:
        sent_word2idx["<pad>"] = 0

    tech_reviews, food_reviews = load_and_clean()

    text = np.array(tech_reviews['text'])
    aspects = np.array(tech_reviews['aspect_term'])
    t_sentences = np.array(map(lambda x, y: replace_with_token(x, y), text, aspects))

    word_count = []
    sent_word_count = []
    target_count = []

    words = []
    sentence_words = []
    target_words = []

    for i in range(t_sentences.shape[0]):
    	sentence = t_sentences[i]
    	target = aspects[i].lower()

    	sentence.replace("$t$", "")
    	sentence = sentence.lower()
    	sentence_splitted = text_to_word_sequence(sentence)
    	target_splitted = text_to_word_sequence(target)
    	
    	max_sent_len = max(max_sent_len, len(sentence_splitted))
    	sentence_words.extend(sentence_splitted)
    	target_words.extend([target])
    	words.extend(sentence_splitted + target_splitted)

    	#max_sent_len = max(max_sent_len, len(sentence.split()))
    	#sentence_words.extend(sentence.split())
    	#target_words.extend([target])
    	#words.extend(sentence.split() + target.split())

    sent_word_count.extend(Counter(sentence_words).most_common())
    target_count.extend(Counter(target_words).most_common())
    word_count.extend(Counter(words).most_common())

    for word, _ in sent_word_count:
    	if word not in sent_word2idx:
    		sent_word2idx[word] = len(sent_word2idx)

    for target, _ in target_count:
    	if target not in target_word2idx:
    		target_word2idx[target] = len(target_word2idx)

	for word, _ in word_count:
		if word not in word_set:
			word_set[word] = 1

    # with open(data_file_name, 'r') as data_file:
    #     lines = data_file.read().split('\n')
    #     for line_no in range(0, len(lines) - 1, 3):
    #         sentence = lines[line_no]
    #         target = lines[line_no + 1]

    #         sentence.replace("$T$", "")
    #         sentence = sentence.lower()
    #         target = target.lower()
    #         max_sent_len = max(max_sent_len, len(sentence.split()))
    #         sentence_words.extend(sentence.split())
    #         target_words.extend([target])
    #         words.extend(sentence.split() + target.split())

    #     sent_word_count.extend(Counter(sentence_words).most_common())
    #     target_count.extend(Counter(target_words).most_common())
    #     word_count.extend(Counter(words).most_common())

    #     for word, _ in sent_word_count:
    #         if word not in sent_word2idx:
    #             sent_word2idx[word] = len(sent_word2idx)

    #     for target, _ in target_count:
    #         if target not in target_word2idx:
    #             target_word2idx[target] = len(target_word2idx)

    #     for word, _ in word_count:
    #         if word not in word_set:
    #             word_set[word] = 1
    print('resources calculation finished')
    return max_sent_len


def get_embedding_matrix(embeddings, sent_word2idx, target_word2idx, edim):
    ''' returns the word and target embedding matrix '''
    word_embed_matrix = np.zeros([len(sent_word2idx), edim], dtype=float)
    target_embed_matrix = np.zeros([len(target_word2idx), edim], dtype=float)

    for word in sent_word2idx:
        if word in embeddings:
            word_embed_matrix[sent_word2idx[word]] = embeddings[word]

    for target in target_word2idx:
        for word in target:
            if word in embeddings:
                target_embed_matrix[target_word2idx[target]] += embeddings[word]
        target_embed_matrix[target_word2idx[target]] /= max(1, len(target.split()))

    print(type(word_embed_matrix))
    return word_embed_matrix, target_embed_matrix


def load_and_clean():
    # read into pandas csv
    tech_reviews = pd.read_csv('data/data_1_train.csv', quoting=csv.QUOTE_NONE, error_bad_lines=False, skipinitialspace=True)
    food_reviews = pd.read_csv('data/data_2_train.csv', quoting=csv.QUOTE_NONE, error_bad_lines=False)

    # rename columns to remove whitespaces
    tech_reviews.columns = ['example_id', 'text', 'aspect_term', 'term_location', 'class']
    food_reviews.columns = ['example_id', 'text', 'aspect_term', 'term_location', 'class']

    # replace _ with whitespace and [comma] with ,
    tech_reviews['text'] = tech_reviews['text'].str.replace('_ ', '')
    food_reviews['text'] = food_reviews['text'].str.replace('_ ', '')
    tech_reviews['text'] = tech_reviews['text'].str.replace("\[comma\]", ',')
    food_reviews['text'] = food_reviews['text'].str.replace("\[comma\]", ',')

    print('tech_reviews shape: ' + str(tech_reviews.shape))
    print('food_reviews shape: ' + str(food_reviews.shape))

    return tech_reviews, food_reviews

def sub_list_finder(sentence, aspect):
    # implement dictionary implementation of sub_list_finder with aspect_dict
    global aspect_dict
    sentence_length = len(sentence)
    aspect_sequence = np.zeros(shape=sentence_length)
    for i in range(sentence_length):
        if aspect[0] == sentence[i]:
            sub_list = sentence[i:i + len(aspect)]
            if aspect == sub_list:
                aspect_index = aspect_dict.get(' '.join(aspect))
                for j in range(len(aspect)):
                    aspect_sequence[i + j] = aspect_index
    return aspect_sequence

def tokenize_t(sentence, aspect):
    # implement dictionary implementation of sub_list_finder with aspect_dict
    global aspect_dict
    sentence_length = len(sentence)
    aspect_sequence = list(sentence)

    for i in range(sentence_length):
    	if aspect[0] == sentence[i]:
    		sub_list = sentence[i:i + len(aspect)]
    		if aspect == sub_list:
    			for j in range(len(aspect)):
    				aspect_sequence[i + j] = '$t$'
    			break
    print(aspect_sequence)
    print(aspect)
    return aspect_sequence

def replace_with_token(sentence, aspect):
	s = sentence.replace(aspect, '$t$')
	return s

def get_aspect_sequences(word_sequence, aspects):
    key_index = 1
    global aspect_dict
    for i in aspects:
        if ' '.join(i) not in aspect_dict:
            aspect_dict[' '.join(i)] = key_index
            key_index += 1
    aspect_sequence = np.array(map(lambda x, y: tokenize_t(x, y), word_sequence,
                                   aspects))  # create aspect sequences based on word_sequence and aspects
    return aspect_sequence

def get_dataset(data_file_name, sent_word2idx, target_word2idx, embeddings, MODE):
    ''' returns the dataset'''
    sentence_list = []
    location_list = []
    target_list = []
    polarity_list = []

    tech_reviews, food_reviews = load_and_clean()

    tech_reviews = tech_reviews.sample(frac=1).reset_index(drop=True)

    text = np.array(tech_reviews['text'])
    aspects = np.array(tech_reviews['aspect_term'])
    polarities = np.array(tech_reviews['class'])
    t_sentences = np.array(map(lambda x, y: replace_with_token(x, y), text, aspects))

    target_error_counter = 0
    if MODE == 'train':
        lower_bound = 0
        upper_bound = int(math.ceil(text.shape[0] * 0.8))
    else:
        lower_bound = int(math.ceil(text.shape[0] * 0.8)) + 1
        upper_bound = text.shape[0]


    print('lower_bound: ' + str(lower_bound))
    print('upper_bound: ' + str(upper_bound))

    # for i in range(t_sentences.shape[0]):
    for i in range(lower_bound, upper_bound):
    	sentence = t_sentences[i].lower()
    	target = aspects[i].lower()
    	polarity = polarities[i] + 1

    	# sent_words = sentence.split()
    	sent_words = text_to_word_sequence(sentence)
    	target_words = text_to_word_sequence(target)
    	# target_words = target.split()
    	# try:
    	# 	# target_location = sent_words.index("$t$")
    	# 	for i, s in enumerate(sent_words):
    	# 		if "$t$" in s:
    	# 			target_location = i
    	# 			print(target_location)
    	# except:
    	# 	print("sentence does not contain target element tag")
    	# 	exit()
    	target_location = -1
    	for idx, s in enumerate(sent_words):
    		# if "$t$" in s:
    		if s == 't':
    			target_location = idx

    	if target_location == -1:
    		print(sentence)
    		print(sent_words)
    		print(target)
    		print(target_words)
    		for idx, s in enumerate(sent_words):
    			target_temp = target_words[0]
    		print(sent_words.index(target_temp))
    		print('target_location = -1')
    		target_error_counter += 1

        is_included_flag = 1
        id_tokenised_sentence = []
        location_tokenised_sentence = []

        for index, word in enumerate(sent_words):
        	# word = word.translate(None, string.punctuation)
        	# word = word.replace('"', '')
        	# word = word.replace('.', '')
        	# word = word.translate(None, r'\"\.\,')
        	# if word == "$t$":
        	#if "$t$" in word:
        	if word == 't':
        		continue
        	try:
        		word_index = sent_word2idx[word]
        	except:
        		print(word)
        		print("id not found for word in the sentence")
        		exit()

        	location_info = abs(index - target_location)

        	if word in embeddings:
        		id_tokenised_sentence.append(word_index)
        		location_tokenised_sentence.append(location_info)

		is_included_flag = 0
		for word in target_words:
			if word in embeddings:
				is_included_flag = 1
				break

        try:
        	target_index = target_word2idx[target]
        except:
        	print(target)
        	print("id not found for target")
        	exit()

		if not is_included_flag:
			print(sentence)
			continue

        sentence_list.append(id_tokenised_sentence)
        location_list.append(location_tokenised_sentence)
        target_list.append(target_index)
        polarity_list.append(polarity)
    # word_sequence = np.array(map(lambda x: text_to_word_sequence(x), text))  # text to word sequence
    # aspects = np.array(map(lambda x: text_to_word_sequence(x), aspects))  # aspects to word sequence
    # aspect_sequences = get_aspect_sequences(word_sequence, aspects) # sequences with aspects replaced with token $t$

    # with open(data_file_name, 'r') as data_file:
    #     lines = data_file.read().split('\n')
    #     for line_no in range(0, len(lines) - 1, 3):
    #         sentence = lines[line_no].lower()
    #         target = lines[line_no + 1].lower()
    #         polarity = int(lines[line_no + 2])

    #         sent_words = sentence.split()
    #         target_words = target.split()
    #         try:
    #             target_location = sent_words.index("$t$")
    #         except:
    #             print("sentence does not contain target element tag")
    #             exit()

    #         is_included_flag = 1
    #         id_tokenised_sentence = []
    #         location_tokenised_sentence = []

    #         for index, word in enumerate(sent_words):
    #             if word == "$t$":
    #                 continue
    #             try:
    #                 word_index = sent_word2idx[word]
    #             except:
    #                 print("id not found for word in the sentence")
    #                 exit()

    #             location_info = abs(index - target_location)

    #             if word in embeddings:
    #                 id_tokenised_sentence.append(word_index)
    #                 location_tokenised_sentence.append(location_info)

    #             # if word not in embeddings:
    #             #   is_included_flag = 0
    #             #   break

    #         is_included_flag = 0
    #         for word in target_words:
    #             if word in embeddings:
    #                 is_included_flag = 1
    #                 break

    #         try:
    #             target_index = target_word2idx[target]
    #         except:
    #             print(target)
    #             print("id not found for target")
    #             exit()

    #         if not is_included_flag:
    #             print(sentence)
    #             continue

    #         # print(id_tokenised_sentence)
    #         # print(sent_words)
    #         # print(location_tokenised_sentence)
    #         # print(target_index)
    #         # print(polarity)
    #         # print(target)
    #         # print(target_words)
    #         # print(location_info)

    #         sentence_list.append(id_tokenised_sentence)
    #         location_list.append(location_tokenised_sentence)
    #         target_list.append(target_index)
    #         polarity_list.append(polarity)
    print('target_error_counter: ' + str(target_error_counter))
    # print(np.array(sentence_list).shape)
    # print(np.array(location_list).shape)
    # print(np.array(target_list).shape)
    # print(np.array(polarity_list).shape)
    return sentence_list, location_list, target_list, polarity_list
