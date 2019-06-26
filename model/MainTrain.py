from entity_attention_model import entity_attention_model
import utils
import itertools
import tensorflow as tf
import os
import utils
import numpy as np
from  config import  config
from  sklearn.metrics import  classification_report


def getFeed():
    pass


num_train_steps=100

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
train=utils.loadData("../saved_data/train.bin")
dev=utils.loadData("../saved_data/dev.bin")


train_sample=list(itertools.chain(*list(map(lambda x:x.samples,train))))
dev_sample=list(itertools.chain(*list(map(lambda x:x.samples,train))))


input_e1=tf.placeholder()
input_e2=tf.placeholder()
input_sentence=tf.placeholder()
input_position=tf.placeholder()
input_e1_start=tf.placeholder()
input_e2_start=tf.placeholder()
input_e1_end=tf.placeholder()
input_e2_end=tf.placeholder()
position_emb=tf.get_variable()
position_diff_emb=tf.get_variable()
word_emb=tf.get_variable()
is_train=True





model=entity_attention_model(input_e1,input_e2,input_sentence,input_position,
                 input_e1_start,input_e2_start,input_e1_end,input_e2_end,position_emb,position_diff_emb,word_emb,is_train)
output=model.outputs

# todo 这里通过 output 构建损失函数



with tf.Session() as sess:
    for i in range(num_train_steps):
        shuffIndex = np.random.permutation(np.arange(len(train_sample)))
        shuffIndex=shuffIndex[0:config["batch_size"]]
        feed = getFeed(train_input_ids[shuffIndex], train_input_masks[shuffIndex], train_segment_ids[shuffIndex], train_labels[shuffIndex])
        train_s_labels=train_labels[shuffIndex]


        _,los,pred,lengths=sess.run([self.train_op,self.total_loss,self.pred_ids,self.lengths],feed_dict=feed)

        classification_report(feed_dict[self.input_y], pre_labels)
        print(i)



