"""
Data loader for TACRED json files.
"""


import tqdm
import json
import random
import torch
import numpy as np
from stanfordcorenlp import StanfordCoreNLP
from gcn_utils import constant, helper, vocab
nlp=StanfordCoreNLP("/home/huwenxiang/deeplearn/stanford-corenlp/stanford-corenlp-full-2018-10-05",lang="en")


def escape_text(text):
    text = text.replace("\\", "\\\\")
    text.replace("'", "\\'")
    return text


class DataLoader(object):
    """
    Load data from json files, preprocess and prepare batches.
    """
    def __init__(self, filename, batch_size, opt, vocab, seed_data=None,evaluation=False):
        self.batch_size = batch_size
        self.opt = opt
        self.vocab = vocab
        self.eval = evaluation
        self.label2id = constant.LABEL_TO_ID
        # 这里用一个方法将我们的sample 格式转换成 他们需要的格式。
        if filename !=None:
            with open(filename) as infile:
                data = json.load(infile)
        else:
            data=[]
            iii=0
            for s in tqdm.tqdm(seed_data):

                # 这里进行采样
                if np.random.rand(1) > 0.02 and s.relation is None:
                    continue
                iii+=1
                d_map={}
                id=s.entity1.doc_id+"-"+str(s.sentence.id)
                d_map["id"]=id
                token=nlp.word_tokenize(s.sentence.text)
                obj_type=s.entity2.type
                subj_type=s.entity1.type
                relation=s.relation.relation_type if s.relation is not None else "None"
                dependency=nlp.dependency_parse(s.sentence.text)
                stanford_deprel=[0]*len(dependency)
                stanford_head = [0]*len(dependency)
                for depen in dependency:
                    stanford_head[depen[-1]-1]=depen[1]
                    stanford_deprel[depen[-1]-1]=depen[0]
                stanford_ner=list(map(lambda x:x[1],nlp.ner(s.sentence.text)))
                stanford_pos=list(map(lambda x:x[1],nlp.pos_tag(s.sentence.text)))
                d_map["stanford_ner"]=stanford_ner
                d_map["stanford_pos"]=stanford_pos
                d_map["stanford_deprel"]=stanford_deprel
                d_map["stanford_head"]=stanford_head
                d_map["token"]=token
                d_map["obj_type"]=obj_type
                d_map["subj_type"]=subj_type
                d_map["relation"]=relation
                # 最麻烦的是拿到位置。

                sentence_start = s.sentence.s_start2end_index[0]
                words_pos = {}


                orig_text=s.sentence.text
                for i, tok in enumerate(token):
                    try:
                        start=orig_text.index(tok)
                        end=start+len(tok)
                        words_pos[start+sentence_start] = i
                        orig_text=orig_text[end:]
                        sentence_start+=len(tok)+start
                    except:
                        raise ("分词后的token在原文找打不到")

                for i in range(len(s.sentence.text)):
                    current_pos=i+s.sentence.s_start2end_index[0]
                    try:
                        if current_pos not in words_pos.keys():
                            words_pos[current_pos]=words_pos[current_pos-1]
                    except:
                        pass



                e1_pos=s.entity1.pos
                e2_pos=s.entity2.pos
                e1_start = e1_pos[0][0]
                e1_end = int(e1_pos[-1][-1])-1
                e2_start = e2_pos[0][0]
                e2_end = int(e2_pos[-1][-1])-1

                entity1_start=words_pos[int(e1_start)]
                entity1_end=words_pos[int(e1_end)]
                entity2_start=words_pos[int(e2_start)]
                entity2_end=words_pos[int(e2_end)]
                d_map["subj_start"]=entity1_start
                d_map["subj_end"]=entity1_end
                d_map["obj_start"]=entity2_start
                d_map["obj_end"]=entity2_end
                # for t in token[entity1_start:entity1_end+1]:
                #     if t  not in s.entity1.text:
                #         print("entity1:"+t+"---"+s.entity1.text)
                #
                # for t in token[entity2_start:entity2_end+1]:
                #     if t  not in s.entity2.text:
                #         print("entity2:"+t+"---"+s.entity2.text)

                if stanford_head.count(0)>1:
                    continue


                if stanford_deprel[entity1_start]==0 or stanford_deprel[entity1_end]==0 or  stanford_deprel[entity2_start]==0 or stanford_deprel[entity2_end]==0:
                    print(1)

                # assert stanford_deprel[entity1_start]!=0
                # assert stanford_deprel[entity1_end]!=0
                # assert stanford_deprel[entity2_start]!=0
                # assert stanford_deprel[entity2_end]!=0


                data.append(d_map)




        self.raw_data = data
        data = self.preprocess(data, vocab, opt)

        # shuffle for training
        if not evaluation:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]
        self.id2label = dict([(v,k) for k,v in self.label2id.items()])
        self.labels = [self.id2label[d[-1]] for d in data] # 中文关系标签
        self.num_examples = len(data)

        # chunk into batches
        data = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
        self.data = data
        print("{} batches created for {}".format(len(data), filename))

    def preprocess(self, data, vocab, opt):
        """ Preprocess the data and convert to ids. """
        processed = []
        for d in data:
            tokens = list(d['token'])
            if opt['lower']:
                tokens = [t.lower() for t in tokens]
            # anonymize tokens
            ss, se = d['subj_start'], d['subj_end']
            os, oe = d['obj_start'], d['obj_end']
            tokens[ss:se+1] = ['SUBJ-'+d['subj_type']] * (se-ss+1)# 对实体 转换成特定的type
            tokens[os:oe+1] = ['OBJ-'+d['obj_type']] * (oe-os+1)#
            tokens = map_to_ids(tokens, vocab.word2id)
            pos = map_to_ids(d['stanford_pos'], constant.POS_TO_ID)
            ner = map_to_ids(d['stanford_ner'], constant.NER_TO_ID)
            deprel = map_to_ids(d['stanford_deprel'], constant.DEPREL_TO_ID)
            head = [int(x) for x in d['stanford_head']]
            assert any([x == 0 for x in head])
            l = len(tokens)
            subj_positions = get_positions(d['subj_start'], d['subj_end'], l)
            obj_positions = get_positions(d['obj_start'], d['obj_end'], l)
            subj_type = [constant.SUBJ_NER_TO_ID[d['subj_type']]]
            obj_type = [constant.OBJ_NER_TO_ID[d['obj_type']]]
            relation = self.label2id[d['relation']]
            processed += [(tokens, pos, ner, deprel, head, subj_positions, obj_positions, subj_type, obj_type, relation)]
        return processed

    def gold(self):
        """ Return gold labels as a list. """
        return self.labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        """ Get a batch with index. """
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]
        batch_size = len(batch)
        batch = list(zip(*batch))
        assert len(batch) == 10

        # sort all fields by lens for easy RNN operations
        lens = [len(x) for x in batch[0]]
        batch, orig_idx = sort_all(batch, lens)

        # word dropout
        if not self.eval:
            words = [word_dropout(sent, self.opt['word_dropout']) for sent in batch[0]]
        else:
            words = batch[0]

        # convert to tensors
        words = get_long_tensor(words, batch_size)
        masks = torch.eq(words, 0)
        pos = get_long_tensor(batch[1], batch_size)
        ner = get_long_tensor(batch[2], batch_size)
        deprel = get_long_tensor(batch[3], batch_size)
        head = get_long_tensor(batch[4], batch_size)
        subj_positions = get_long_tensor(batch[5], batch_size)
        obj_positions = get_long_tensor(batch[6], batch_size)
        subj_type = get_long_tensor(batch[7], batch_size)
        obj_type = get_long_tensor(batch[8], batch_size)

        rels = torch.LongTensor(batch[9])

        return (words, masks, pos, ner, deprel, head, subj_positions, obj_positions, subj_type, obj_type, rels, orig_idx)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

def map_to_ids(tokens, vocab):
    ids = [vocab[t] if t in vocab else constant.UNK_ID for t in tokens]
    return ids

def get_positions(start_idx, end_idx, length):
    """ Get subj/obj position sequence. """
    return list(range(-start_idx, 0)) + [0]*(end_idx - start_idx + 1) + \
            list(range(1, length-end_idx))

def get_long_tensor(tokens_list, batch_size):
    """ Convert list of list of tokens to a padded LongTensor. """
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.LongTensor(batch_size, token_len).fill_(constant.PAD_ID)
    for i, s in enumerate(tokens_list):
        tokens[i, :len(s)] = torch.LongTensor(s)
    return tokens

def sort_all(batch, lens):
    """ Sort all fields by descending order of lens, and return the original indices. """
    unsorted_all = [lens] + [range(len(lens))] + list(batch)
    sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
    return sorted_all[2:], sorted_all[1]

def word_dropout(tokens, dropout):
    """ Randomly dropout tokens (IDs) and replace them with <UNK> tokens. """
    return [constant.UNK_ID if x != constant.UNK_ID and np.random.random() < dropout \
            else x for x in tokens]

