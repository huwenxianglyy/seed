# from entity_attention_model import entity_attention_model
# from utils import utils
import itertools
import torch
import os
import random
import utils
import numpy as np
from tqdm import tqdm
from config import *
from sklearn.metrics import classification_report

args = ConfigParameter()


def set_seed():
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(2019)
    random.seed(2019)
    torch.manual_seed(2019)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(2019)


def get_feed(data, samples, padding=False, shuffle=True):
    sentence_dict = {}
    # all_sent_ids = []
    # all_sents = []
    # all_pos1 = []
    # all_pos2 = []
    # all_labels = []
    for sample in samples:
        en1 = utils.clean_str(sample.entity1.text).split()
        en2 = utils.clean_str(sample.entity2.text).split()
        r_obj, s_obj = sample.relation, sample.sentence
        sentence = utils.clean_str(s_obj.text).split()

        en1_pos = []
        en2_pos = []
        for i in range(len(sentence)):
            for j in range(len(en1)):
                if sentence[i] == en1[j]:
                    en1_pos.append(i)
            for k in range(len(en2)):
                if sentence[i] == en2[k]:
                    en2_pos.append(i)
        if len(en1_pos) == 0 or len(en2_pos) == 0:
            print(s_obj.text)
            # Stem Cell Maintenance Is Regulated by AP2AP2 was originally identified as a floral homeotic gene encoding A-function of the ABC model of organ identity specification (Bowman et al., 1989, 1991).
            print(sample.entity1.text)    # AP2
            print(sample.entity2.text)    # Stem Cell Maintenance

        length = min(args.sen_len, len(sentence))
        words = []
        pos1_temp = [-1] * length
        pos2_temp = [-1] * length
        for a in en1_pos:
            pos1_temp[a] = 32
        for b in en2_pos:
            pos2_temp[b] = 32

        for i in range(length):
            words.append(data[0].word2id.get(sentence[i], data[0].word2id['UNK']))
        pos1 = utils.get_w_pos_in_e(pos1_temp, en1_pos, length)
        pos2 = utils.get_w_pos_in_e(pos2_temp, en2_pos, length)

        if length < args.sen_len:
            for i in range(length, args.sen_len):
                # range(10, 20)的意思是10,11, ... ,19
                words.append(data[0].word2id['PAD'])
            pos1 = pos1 + (args.sen_len - length) * [0]
            pos2 = pos2 + (args.sen_len - length) * [0]
        sentence_dict[sample.id] = (words, pos1, pos2)

    rel = [0] * len(config['relation_type2id'])
    try:
        sample.id, types = line.strip().split('\t')
        type_list = types.split()
        for tp in type_list:
            if len(type_list) > 1 and tp == '0':
                # if a sentence has multiple relations, we only consider non-NA relations
                continue
            rel[int(tp)] = 1
    except:
        sent_id = line.strip()
    #
    #     all_sent_ids.append(sent_id)
    #     all_sents.append(sentence_dict[sent_id][0])
    #     all_pos1.append(sentence_dict[sent_id][1])
    #     all_pos2.append(sentence_dict[sent_id][2])
    #     # sentence_dict是一个字典，
    #     # 每个键是这样的ID。TRAIN_SENT_ID_000001
    #     # 每个值是一个元组，为(words, pos1, pos2)。其中分别为
    #     # word_embedding的索引
    #     # entity1的位置embedding的索引
    #     # entity2的位置embedding的索引
    #
    #     # 所以，all_sents的形状是(28w, 60)
    #     # all_pos1、2的形状是(28w, 60)
    #     all_labels.append(np.reshape(np.asarray(rel, dtype=np.float32), (-1, self.num_classes)))
    #     # all_labels的shape为(28w, 35)
    #
    # self.data_size = len(all_sent_ids)
    # self.datas = all_sent_ids
    #
    # # all_sents = np.concatenate(all_sents, axis=0)
    # # all_pos1 = np.concatenate(all_pos1, axis=0)
    # # all_pos2 = np.concatenate(all_pos2, axis=0)
    # all_sents = torch.LongTensor(all_sents)
    # all_pos1 = torch.LongTensor(all_pos1)
    # all_pos2 = torch.LongTensor(all_pos2)
    # # 现在的all_sents的为torch.Size([17241060])
    # all_labels = np.concatenate(all_labels, axis=0)
    # all_labels = torch.LongTensor(all_labels)
    # # 原来(28w，35)，现在的all_labels的shape为(28w*35,)  详细见demo5
    #
    # data_order = list(range(self.data_size))
    # if shuffle:
    #     np.random.shuffle(data_order)
    # if padding:
    #     if self.data_size % self.batch_size != 0:
    #         data_order += [data_order[-1]] * (self.batch_size - self.data_size % self.batch_size)
    #
    # for i in tqdm(range(len(data_order) // self.batch_size)):
    #     idx = data_order[i * self.batch_size: (i + 1) * self.batch_size]
    #
    #     # all_sents[idx] 的shape:(batch_size, 60)
    #     # all_labels[idx] 的shape：(batch_size, 35)
    #     # yield返回的是一个tuple，长度为4
    #     yield all_sents[idx], all_pos1[idx], all_pos2[idx], all_labels[idx]


if __name__ == '__main__':
    set_seed()

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    train = utils.loadData('./saved_data/train.bin')
    dev = utils.loadData('./saved_data/dev.bin')

    train_sample = list(itertools.chain(*list(map(lambda x: x.samples, train))))
    dev_sample = list(itertools.chain(*list(map(lambda x: x.samples, dev))))

    args.vocab_size = len(train[0].word2id)  # train中有2631个单词
    assert args.vocab_size == 2631
    args.vocab_size = len(dev[0].word2id)  # dev中有1832个单词
    assert args.vocab_size == 1832

    get_feed(train, train_sample)
    # feed_dev_sent_dict = get_feed(dev, dev_sample)
