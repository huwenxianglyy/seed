import tensorflow as tf
from entity_attention_model import entity_attention_model
# from utils import utils
import itertools
# import torch
import os
import random
import utils
import numpy as np
from tqdm import tqdm
from config import *
from sklearn.metrics import classification_report
import re
import collections
import tqdm









def set_seed():
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(2019)
    random.seed(2019)
    # torch.manual_seed(2019)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(2019)


def add_splitWord2List(split_words,last_pos,temple):
    search=re.search(".*(\'s|\'ve|\'re|\'d|\'ll|n\'t)$", temple)
    if search is not None:
        text=search.group(1)
        start=search.start(1)
        split_words[last_pos]=temple[0:start]
        split_words[last_pos+start]=text
        return
    iters=re.split("\'",temple)
    for text in iters:
        if text=="":
            split_words[last_pos] = "'"
            last_pos+=1
        else:
            split_words[last_pos] = text
            last_pos+=len(text)




def createInputForModel(data,samples):
    entity1=[]
    entity2=[]
    entity1_pos=[]
    entity2_pos=[]
    real_relation=[]
    sentences=[]
    sentences_len=[]
    error_sample_num=0
    entity1_type=[]
    entity2_type=[]
    relation_entity1_type=[]
    relation_entity2_type=[]


    word2idIndex=args.word2Id

    for sample in tqdm.tqdm(samples):
        sentence_start_end = sample.sentence.s_start2end_index
        e1_pos=sample.entity1.pos
        e2_pos=sample.entity2.pos
        en1 = utils.clean_str(sample.entity1.text).split()
        en2 = utils.clean_str(sample.entity2.text).split()
        r_obj, s_obj = sample.relation, sample.sentence
        bucket=collections.OrderedDict()
        entity1_type.append(config["entity_type2id"][sample.entity1.type])
        entity2_type.append(config["entity_type2id"][sample.entity2.type])
        relation_entity1_type.append(config["entity_all_relation_type2id"]["None" if r_obj is None else r_obj.entity1_type])
        relation_entity2_type.append(config["entity_all_relation_type2id"]["None" if r_obj is None else r_obj.entity2_type])

        for  i, c in enumerate(s_obj.text):
            bucket[i+sentence_start_end[0]]=c
        temple=""
        last_pos=0
        split_words=collections.OrderedDict()
        for k,c in bucket.items():
            if c==" " or c==' ':
                if len(temple)>0:
                    add_splitWord2List(split_words, last_pos, temple)
                    temple=""
                continue
            if re.search("[\^!@#$%&*()\[\]=+_:,\"’‘、–\-/≈∷≥—>∼%“”]",c) is not None:
                if len(temple) > 0:
                    add_splitWord2List(split_words, last_pos, temple)
                split_words[k] =c
                temple = ""
                continue

            if len(temple)==0:
                last_pos=k
            temple+=c
        if len(temple) > 0:
            add_splitWord2List(split_words, last_pos, temple)

        words=[]
        words_pos={}
        for i,k in enumerate(split_words.keys()):
            words.append(split_words[k])
            words_pos[k]=i

        e1_start=e1_pos[0][0]# 这个是使用全局的位置
        e2_start=e2_pos[0][0]# 这个是使用全局的位置
        try:
            e1_temp=words[words_pos[int(e1_start)]:]
            e2_temp=words[words_pos[int(e2_start)]:]


        except:
            error_sample_num+=1
            print(error_sample_num)
            continue




        relation_type=0
        if r_obj is not None:
            relation_type=config["relation_type2id"][r_obj.relation_type]
        real_relation.append(relation_type)


        word_indexs=[]
        for i in range(args.sen_len):
            if i<len(words):
                word_indexs.append(word2idIndex.get(words[i], word2idIndex[args.UNK]))
            else:
                word_indexs.append(word2idIndex.get(word2idIndex[args.PAD]))
        sentences.append(word_indexs)
        sentences_len.append(len(words))

        # 下面加入实体的信息
        entity1_index=[]
        entity2_index=[]
        for i in range(args.entity_len):
            if i < len(en1):
                entity1_index.append(word2idIndex.get(en1[i], word2idIndex[args.UNK]))
            else:
                entity1_index.append(word2idIndex.get(word2idIndex[args.PAD]))

        entity1.append(entity1_index)

        for i in range(args.entity_len):
            if i < len(en2):
                entity2_index.append(word2idIndex.get(en2[i], word2idIndex[args.UNK]))
            else:
                entity2_index.append(word2idIndex.get(word2idIndex[args.PAD]))
        entity2.append(entity2_index)

        # 下面加入位置信息
        en1_pos=[]
        en2_pos=[]
        en1_start=words_pos[int(e1_start)] # 这个位置是相对于分词后的位置。
        en2_start=words_pos[int(e2_start)]

        for i in range(len(word_indexs)):
            pos1 = i - en1_start+args.pos_limit # 这里以 args.pos_limit 为中心 计算位置，最大的位置编码 2*args.pos_limit
            pos2 = i - en2_start+args.pos_limit
            if pos1<=0:
                pos1=0
            if pos2<=0:
                pos2=0
            if pos1>=2*args.pos_limit-1:
                pos1=2*args.pos_limit-1
            if pos2 >= 2*args.pos_limit-1:
                pos2 = 2*args.pos_limit-1
            en1_pos.append(pos1)
            en2_pos.append(pos2)


        for i in range(en1_start,en1_start+len(en1)):
            en1_pos[i]=args.pos_limit

        for i in range(en2_start,en2_start+len(en2)):
            en2_pos[i]=args.pos_limit

        # en1_pos[en1_start:en1_start+len(en1)]=args.pos_limit
        # en2_pos[en2_start:en2_start+len(en2)]=args.pos_limit
        entity1_pos.append(en1_pos)
        entity2_pos.append(en2_pos)


    return np.array(sentences,dtype=np.int32),np.array(entity1,dtype=np.int32),\
           np.array(entity2,dtype=np.int32),np.array(real_relation,dtype=np.int32),\
           np.array(entity1_pos,dtype=np.int32),np.array(entity2_pos,dtype=np.int32),\
           np.array(sentences_len,dtype=np.int32),np.array(entity1_type,dtype=np.int32), \
            np.array(entity2_type, dtype=np.int32),np.array(relation_entity1_type, dtype=np.int32), \
            np.array(relation_entity2_type, dtype=np.int32)






if __name__ == '__main__':
    set_seed()
    # embedd=utils.read_file(args.embedding_file)

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    train = utils.loadData('../saved_data/train.bin')
    dev = utils.loadData('../saved_data/dev.bin')

    train_sample = list(itertools.chain(*list(map(lambda x: x.samples, train))))
    dev_sample = list(itertools.chain(*list(map(lambda x: x.samples, dev))))

    args.vocab_size = len(train[0].word2id)  # train中有2631个单词
    assert args.vocab_size == 2631
    args.vocab_size = len(dev[0].word2id)  # dev中有1832个单词
    assert args.vocab_size == 1832







    # get_feed(train, train_sample)
    train_sentences, train_entity1,train_entity2,train_real_relation,train_e1_pos,train_e2_pos,train_seq_len,train_entity1_type,train_entitiy2_type,train_r_e1_type,train_r_e2_type=createInputForModel(train, train_sample)
    dev_sentences, dev_entity1,dev_entity2,dev_real_relation,dev_e1_pos,dev_e2_pos,dev_seq_len,dev_entity1_type,dev_entitiy2_type,dev_r_e1_type,dev_r_e2_type=createInputForModel(train, dev_sample)

    input_e1 = tf.placeholder(shape=[None,5],dtype=tf.int32)
    input_e2 = tf.placeholder(shape=[None,5],dtype=tf.int32)
    input_sentence = tf.placeholder(shape=[None,args.sen_len],dtype=tf.int32)

    input_e1_position=tf.placeholder(shape=[None,args.sen_len],dtype=tf.int32)
    input_e2_position=tf.placeholder(shape=[None,args.sen_len],dtype=tf.int32)
    input_e1_type=tf.placeholder(shape=[None,1],dtype=tf.int32)
    input_e2_type=tf.placeholder(shape=[None,1],dtype=tf.int32)
    input_e1_r_type=tf.placeholder(shape=[None,1],dtype=tf.int32)
    input_e2_r_type=tf.placeholder(shape=[None,1],dtype=tf.int32)
    seq_len = tf.placeholder(shape=[None,1],dtype=tf.int32)
    input_realtion = tf.placeholder(shape=[None,1],dtype=tf.int32)




    # 下面建立各种embedding
    position_diff_emb = tf.get_variable('pos1_embedding',                               # 位置向量
                                     [args.pos_limit*2, args.pos_dim], trainable=False) # 后期可以多设一个Pad 向量，全部为0
                                                                                        # 后期可以 设置pos1 和 pos2 来表示到实体1的距离和实体2的距离
    word_emb =  tf.get_variable(initializer=args.word2Vec,
                                         name='word_embedding',trainable=False)
    relation_emb=tf.get_variable(shape=[len(config["relation_type"]),args.relation_dim],name='relation_emb',trainable=False)



    entity_type_emb=tf.get_variable(shape=[len(config["entity_type"]),args.entity_type_dim],name='entity_type_emb',trainable=True)




    def get_feed(sentences=None, entity1=None,entity2=None,
                 e1_pos=None,e2_pos=None,seq_len=None,entity1_type=None,
                 entitiy2_type=None,r_e1_type=None,r_e2_type=None,sentence_len=None,real_relation=None):
        # 可以feed的 变量
        # input_e1
        # input_e2
        # input_sentence
        # input_e1_position
        # input_e2_position
        # input_e1_type
        # input_e2_type
        # input_e1_r_type
        # input_e2_r_type
        # seq_len
        # input_realtion

        if real_relation is None:
            feed = {input_sentence: sentences,input_e1:entity1,input_e2:entity2,
                    input_e1_position: e1_pos, input_e2_position: e2_pos,seq_len:sentence_len,
                    input_e1_type:entity1_type,input_e2_type:entitiy2_type
                    }

        else:
            feed = {input_sentence: sentences, input_e1: entity1, input_e2: entity2,
                    input_e1_position: e1_pos, input_e2_position: e2_pos, seq_len: sentence_len,
                    input_e1_type: entity1_type, input_e2_type: entitiy2_type,
                    input_realtion:real_relation
                    }
        return feed





    model=entity_attention_model(input_e1, input_e2, input_sentence, input_e1_position,
                  input_e2_position,input_e1_type,input_e2_type,
                 input_e1_r_type,input_e2_r_type,
                  position_diff_emb, word_emb,entity_type_emb, seq_len,True)

    outputs=model.outputs_final
    full_c=tf.layers.Dense(config["relation_type"])
    logits=full_c(outputs)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=input_realtion, name="soft_loss")
    total_loss = tf.reduce_mean(loss, name="loss")


    predict = tf.argmax(tf.nn.softmax(logits), axis=1, name="predictions")
    acc = tf.reduce_mean(tf.cast(tf.equal(input_realtion, tf.cast(predict, dtype=tf.int64)), "float"), name="accuracy")


    train_op = tf.train.AdamOptimizer(args.lr).minimize(total_loss)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer)
        for train_step in range(args.num_epochs):
            shuffIndex = np.random.permutation(np.arange(len(train_sample)))
            shuffIndex = shuffIndex[0:args.batch_size]

            feed = get_feed(real_relation=train_real_relation[shuffIndex], sentences=train_sentences[shuffIndex],
                                e1_pos=train_e1_pos[shuffIndex],e2_pos=train_e2_pos[shuffIndex],
                            entity1=train_entity1[shuffIndex],entity2=train_entity2[shuffIndex]
                            ,entity1_type=train_entity1_type[shuffIndex],
                            entitiy2_type=train_entitiy2_type[shuffIndex],sentence_len=train_seq_len[shuffIndex])

            acc, total_loss=sess.run([train_op,acc,total_loss],feed_dict=feed)



