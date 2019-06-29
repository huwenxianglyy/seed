import json
import pickle
import re
from collections import Counter
from config import *
import numpy as np
import jieba
import os
import shutil
# import torch
from nltk.corpus import stopwords
import nltk
from stanfordcorenlp import StanfordCoreNLP


args = ConfigParameter()

def read_file(path):
    with open(path, "r", encoding='utf-8') as f1:
        data = f1.read().splitlines()
    return data
    # 返回一个list, 每个元素是一行字符


def dumpData4Gb(data, openPath):
    with open(openPath, "wb") as file:
        pickle.dump(data, file, protocol=4)


def split2sentence(d):
    s = "".join(list(map(lambda x: x.text, d.characters)))
    s = s.replace('\n', ' ')
    first_word = re.findall(r"[\.\?\!]\s?[A-Z]+\w+", s)
    sent = re.split(r'[\.\?\!]\s?[A-Z]+\w+', s)
    new_sents = [w + z for w, z in zip(first_word, sent[1:])]
    new_sents.insert(0, sent[0])

    sent_identifiers = ['.', '?', '!']
    for char in sent_identifiers:
        for i, pre_sent in enumerate(new_sents):
            if pre_sent.startswith(char):
                pre_sent = pre_sent.replace(char, '', 1)
                new_sents[i - 1] = new_sents[i - 1] + char
                for chara in pre_sent:
                    if chara.find(' ') == 0 and chara == ' ':
                        pre_sent = pre_sent.replace(' ', '', 1)
                        new_sents[i - 1] = new_sents[i - 1] + ' '
                        break
                    break
                new_sents[i] = pre_sent
    return new_sents


# def count_words_in_each_sentence(sentence):
#     nlp = StanfordCoreNLP('E:\TOOLS\stanford-corenlp-full-2018-10-05', lang='en')
#     return nlp.word_tokenize(sentence), len(nlp.word_tokenize(sentence))


def retrieve_entities(d):
    for each_t in d.entities:
        start = int(each_t.pos[0][0])
        end = int(each_t.pos[-1][-1])
        entity_in_sentence = list(filter(lambda x: x.s_start2end_index[0] <= start
                                                   and x.s_start2end_index[-1] >= end, d.sentences))
        assert len(entity_in_sentence) != 0
        entity_in_sentence[0].entitys.append(each_t)


def retrieve_relations(d):
    for relation in d.relations:
        e1 = relation.entity1
        e2 = relation.entity2
        realtion_in_sentence = list(filter(lambda x: e1 in x.entitys and e2 in x.entitys, d.sentences))
        if len(realtion_in_sentence) == 0:
            relation.skip_sentence = 1
            d.skip_sentence_relation.append(relation)
        else:
            realtion_in_sentence[0].relations.append(relation)


def two_e_have_multi_r(ls, r2e_dict, doc_r_obj):
    new = []
    for i in ls:
        new.append(' '.join(sorted(i)))
    count = Counter(new)
    multi_r_list = []
    for ele, num in count.items():
        assert num <= 2
        if num == 2:
            multi_r_list.append((list(filter(lambda x: r2e_dict[x.id] == ele, doc_r_obj)), num))
    return multi_r_list


# def count_words_in_each_sentence(sents_text):
#     nltk.download('punkt')
#     text = nltk.word_tokenize(sents_text)
#     # nlp = StanfordCoreNLP('E:\TOOLS\stanford-corenlp-full-2018-10-05', lang='en')
#     # return nlp.word_tokenize(sents_text), len(nlp.word_tokenize(sents_text))
#     return text, len(text)


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def loadData(openPath):
    with open(openPath, "rb", ) as file:
        data = pickle.load(file)
        return data


def minibatchesNdArray(test_ids, test_input_ids, test_input_masks, test_segment_ids, test_labels, words,
                       minibatch_size):
    iterNum = len(test_ids) // minibatch_size
    for i in range(iterNum):
        start = i * minibatch_size
        end = (i + 1) * minibatch_size
        yield test_ids[start:end], \
              test_input_ids[start:end], \
              test_input_masks[start:end], \
              test_segment_ids[start:end], \
              test_labels[start:end], \
              words[start:end]


def pos_index(x):

    if x < -args.pos_limit:
        return 0
    if x >= -args.pos_limit and x <= args.pos_limit:
        return x + args.pos_limit
    if x > args.pos_limit:
        return 2 * args.pos_limit


def get_w_pos_in_e(pos, en_pos, lens):
    assert len(en_pos) > 0
    for i in range(lens):
        if len(en_pos) == 1:
            pos[i] = pos_index(i - en_pos[0])
        else:
            if i < en_pos[0]:
                pos[i] = pos_index(i - en_pos[0])
            if i > en_pos[-1]:
                pos[i] = pos_index(i - en_pos[-1])
        if -1 in pos:
            w_pos_in_e = [i for i, x in enumerate(pos) if x == -1]
            ls = list(range(-len(w_pos_in_e), 0))
            for z, p in enumerate(w_pos_in_e):
                pos[p] = ls[z]
    return pos


def load_word_vec(self):
    global args
    wordMap = {}
    wordMap['PAD'] = len(wordMap)
    wordMap['UNK'] = len(wordMap)
    word_embed = []
    for line in tqdm(open(self.embedding_file, encoding='utf-8')):
        content = line.strip().split()
        if len(content) != self.embedding_size + 1:
            continue
        wordMap[content[0]] = len(wordMap)
        word_embed.append(np.asarray(content[1:], dtype=np.float32))
        # word_embed 列表，列表中每个元素是np数组，np数组的shape=(300,)

    word_embed = np.stack(word_embed)
    # word_embed变为np数组，shape=(372909, 300)
    embed_mean, embed_std = word_embed.mean(), word_embed.std()

    pad_embed = np.random.normal(embed_mean, embed_std, (2, self.embedding_size))
    word_embed = np.concatenate((pad_embed, word_embed), axis=0)
    word_embed = word_embed.astype(np.float32)
    # wordMap：一个词向量中的字典，word2id
    # word_embed：一个词向量np数组，shape(words_size, 300)
    # <class 'numpy.ndarray'>
    # (372911, 300)
    #
    # pickle.dump(word_embed, open('./embedding/embedding.pickle', 'wb'))
    return wordMap, word_embed, word_embed.shape[0]
