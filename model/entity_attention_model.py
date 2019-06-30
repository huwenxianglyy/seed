import tensorflow as  tf
from tensorflow.contrib import rnn
from config import  *

args.filter_size=50
args.rnn_cell_size=50
class entity_attention_model(object):
    def __init__(self, input_e1, input_e2, input_sentence, input_e1_position,
                  input_e2_position,input_e1_type,input_e2_type,
                 input_e1_r_type,input_e2_r_type,
                  position_diff_emb, word_emb,entity_type_emb, seq_len, is_train):
        '''
        :param input_e1:
        :param input_e2:
        :param input_sentence: 输入的sentence 序列 [batch_size,156]
        :param input_position: 输入的 位置 序列 [batch_size,156]
        :param input_e1_start: e1的start [batch_size,1]
        :param input_e2_start: e2的start [batch_size,1]
        :param input_e1_end:  e1的end [batch_size,1]
        :param input_e2_end:  e2的end [batch_size,1]
        :param position_emb:  position_emb [156,position_dim]
        :param position_diff_emb:  distance_emb [78,position_dim]
        :param word_emb: word_emb
        '''

        self.input_sentence=input_sentence
        self.input_e1 = input_e1
        self.input_e2 = input_e2
        self.input_e1_position = input_e1_position
        self.input_e2_position = input_e2_position
        self.input_e1_type = input_e1_type
        self.input_e2_type = input_e2_type
        self.input_e1_r_type = input_e1_r_type
        self.input_e2_r_type = input_e2_r_type


        self.entity_type_emb = entity_type_emb



        self.position_diff_emb = position_diff_emb
        self.word_emb = word_emb
        self.seq_len = seq_len

        self.cell_type = "lstm"
        self.is_training = is_train
        self.hidden_unit = args.rnn_cell_size
        self.droupout_rate = 0.7

        self.create()

    def create(self):
        self.emb_sentence = tf.nn.embedding_lookup(self.word_emb, self.input_sentence)  # [batch,156,300]
        self.e1_position = tf.nn.embedding_lookup( self.position_diff_emb,self.input_e1_position) # [batch,156,10]
        self.e2_position = tf.nn.embedding_lookup(self.position_diff_emb, self.input_e2_position)  # [batch,156,10]
        self.senAndPosition = tf.concat([self.emb_sentence,self.e1_position,self.e2_position], axis=-1)

        self.e1_emb=tf.nn.embedding_lookup(self.word_emb, self.input_e1)  #
        self.e2_emb=tf.nn.embedding_lookup(self.word_emb, self.input_e2)  #


        self.e1_type_emb=tf.squeeze(tf.nn.embedding_lookup(self.entity_type_emb, self.input_e1_type),axis=1)  #
        self.e2_type_emb=tf.squeeze(tf.nn.embedding_lookup(self.entity_type_emb, self.input_e2_type),axis=1) #


        with tf.variable_scope("cnn_layer"):# 主要对两个实体进行卷积获取特征
            kernel = tf.get_variable(shape=[1, args.word_dim, args.filter_size], dtype=tf.float32, name="kernel",  #
                                     initializer=tf.random_normal_initializer())
            e1_output = tf.nn.conv1d(self.e1_emb, kernel, 1, 'VALID')  # 这里的形状应该是【N，5,50】

            e1_output = tf.layers.max_pooling1d(e1_output, args.entity_len, strides=1, padding="valid")
            e1_output = tf.reshape(e1_output, [-1, args.filter_size])  # tf.squeeze()
            self.e1_output = tf.concat([e1_output,self.e1_type_emb], axis=-1)  # tf.squeeze()




            e2_output = tf.nn.conv1d(self.e2_emb, kernel, 1, 'VALID')  # 这里的形状应该是【N，5,50】
            e2_output = tf.layers.max_pooling1d(e2_output, args.entity_len, strides=1, padding="valid")
            e2_output = tf.reshape(e2_output, [-1, args.filter_size])  # tf.squeeze()
            self.e2_output = tf.concat([e2_output,self.e2_type_emb], axis=-1)  # tf.squeeze()


        with tf.variable_scope('rnn_layer'):
            cell_fw = self._witch_cell()
            cell_bw = self._witch_cell()
            if self.droupout_rate is not None and self.is_training:
                cell_bw = rnn.DropoutWrapper(cell_bw, output_keep_prob=self.droupout_rate)
                cell_fw = rnn.DropoutWrapper(cell_fw, output_keep_prob=self.droupout_rate)

            outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, self.senAndPosition,
                                                         dtype=tf.float32)
            self.outputs = tf.concat(outputs, axis=2)


        # 这里做二次attention

        dense=tf.layers.Dense(args.filter_size+args.entity_type_dim) # todo 这里可以加入激活函数
        self.e1_l1_out=tf.expand_dims(dense(self.e1_output), axis=1)
        self.e2_l1_out=tf.expand_dims(dense(self.e2_output), axis=1)



        self.e1_att_logits=tf.einsum('bed,bdf->bef', self.e1_l1_out, tf.transpose(self.outputs, [0, 2, 1]))
        e1_attention_softmax = tf.nn.softmax(self.e1_att_logits, axis=2)
        self.e1_att_outputs = tf.einsum('bed,bdf->bef', e1_attention_softmax, self.outputs)



        self.e2_att_logits=tf.einsum('bed,bdf->bef', self.e2_l1_out, tf.transpose(self.outputs, [0, 2, 1]))
        e2_attention_softmax = tf.nn.softmax(self.e2_att_logits, axis=2)
        self.e2_att_outputs = tf.einsum('bed,bdf->bef', e2_attention_softmax, self.outputs)

        outputs=(self.e1_att_outputs+self.e2_att_outputs)/2

        self.outputs_final = tf.squeeze(outputs, 1) # 这里的维度是 N，300,768  # 原版是 用self passage_setentce





    def _witch_cell(self):
        """
        RNN 类型
        :return:
        """
        cell_tmp = None
        if self.cell_type == 'lstm':
            cell_tmp = rnn.BasicLSTMCell(self.hidden_unit)
        elif self.cell_type == 'gru':
            cell_tmp = rnn.GRUCell(self.hidden_unit)
        # 是否需要进行dropout
        if self.droupout_rate is not None and self.is_training:
            cell_tmp = rnn.DropoutWrapper(cell_tmp, output_keep_prob=self.droupout_rate)
        return cell_tmp
