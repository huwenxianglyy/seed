import tensorflow as  tf
from tensorflow.contrib import rnn


class entity_attention_model(object):
    def __init__(self, input_e1, input_e2, input_sentence, input_position,
                 input_e1_start, input_e2_start, input_e1_end, input_e2_end,
                 position_emb, position_diff_emb, word_emb, seq_len, is_train):
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

        self.input_e1 = input_e1
        self.input_e2 = input_e2
        self.input_sentence = input_sentence
        self.input_position = input_position
        self.input_e1_start = input_e1_start
        self.input_e2_start = input_e2_start
        self.input_e1_end = input_e1_end
        self.input_e2_end = input_e2_end
        self.position_emb = position_emb
        self.position_diff_emb = position_diff_emb
        self.word_emb = word_emb
        self.seq_len = seq_len

        self.cell_type = "lstm"
        self.is_training = is_train
        self.hidden_unit = 512
        self.droupout_rate = 0.7

        self.create()

    def create(self):
        self.emb_sentence = tf.nn.embedding_lookup(self.word_emb, self.input_sentence)  # [batch,156,300]
        self.emb_position = tf.nn.embedding_lookup(self.position_emb, self.input_position)  # [batch,156,10]
        self.senAndPosition = tf.concat(self.emb_sentence, self.position_emb, axis=-1)
        with tf.variable_scope('rnn_layer'):
            cell_fw = self._witch_cell()
            cell_bw = self._witch_cell()
            if self.droupout_rate is not None and self.is_training:
                cell_bw = rnn.DropoutWrapper(cell_bw, output_keep_prob=self.droupout_rate)
                cell_fw = rnn.DropoutWrapper(cell_fw, output_keep_prob=self.droupout_rate)

            # if self.num_layers > 1:
            #     cell_fw = rnn.MultiRNNCell([cell_fw] * self.num_layers, state_is_tuple=True)
            #     cell_bw = rnn.MultiRNNCell([cell_bw] * self.num_layers, state_is_tuple=True)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, self.senAndPosition,
                                                         dtype=tf.float32)
            self.outputs = tf.concat(outputs, axis=2)

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
