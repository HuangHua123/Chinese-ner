# encoding = utf-8
import numpy as np
import tensorflow as tf
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from tensorflow.contrib.layers.python.layers import initializers
from .utils import result_to_json
from .data_utils import create_input, iobes_iob,iob_iobes


class Model(object):

    #初始化模型参数
    def __init__(self, config):

        self.config = config
        
        self.lr = config["lr"]  #0.001
        self.char_dim = config["char_dim"]  #100
        self.lstm_dim = config["lstm_dim"]  #100
        self.seg_dim = config["seg_dim"]  #20

        self.num_tags = config["num_tags"]  #51
        self.num_chars = config["num_chars"]#样本中总字数2
        self.num_segs = 4

        self.global_step = tf.Variable(0, trainable=False)
        self.best_dev_f1 = tf.Variable(0.0, trainable=False)
        self.best_test_f1 = tf.Variable(0.0, trainable=False)
        self.initializer = initializers.xavier_initializer()
        
        

        # 为模型添加占位符

        self.char_inputs = tf.placeholder(dtype=tf.int32,
                                          shape=[None, None],
                                          name="ChatInputs")
        self.seg_inputs = tf.placeholder(dtype=tf.int32,
                                         shape=[None, None],
                                         name="SegInputs")

        self.targets = tf.placeholder(dtype=tf.int32,
                                      shape=[None, None],
                                      name="Targets")
        # dropout
        self.dropout = tf.placeholder(dtype=tf.float32,
                                      name="Dropout")

        used = tf.sign(tf.abs(self.char_inputs))
        length = tf.reduce_sum(used, reduction_indices=1)  #列相加  例如[[1,2,3,],[4,5,6]](2*3)= [6,15](2)
        self.lengths = tf.cast(length, tf.int32) 
        self.batch_size = tf.shape(self.char_inputs)[0]
        self.num_steps = tf.shape(self.char_inputs)[-1]
        
        
        #使用膨胀卷积做backbone网络
        self.model_type = config['model_type']

        #膨胀卷积参数
        self.layers = [
            {
                'dilation': 1
            },
            {
                'dilation': 1
            },
            {
                'dilation': 2
            },
        ]
        self.filter_width = 3
        self.num_filter = self.lstm_dim  #100
        self.embedding_dim = self.char_dim + self.seg_dim   #120
        self.repeat_times = 4
        self.cnn_output_width = 0
        
        # embeddings for chinese character and segmentation representation
        embedding = self.embedding_layer(self.char_inputs, self.seg_inputs, config)

        if self.model_type == 'bilstm':
            # apply dropout before feed to lstm layer
            model_inputs = tf.nn.dropout(embedding, self.dropout)

            # bi-directional lstm layer
            model_outputs = self.biLSTM_layer(model_inputs, self.lstm_dim, self.lengths)

            # logits for tags
            self.logits = self.project_layer_bilstm(model_outputs)
        
        elif self.model_type == 'idcnn':
            # apply dropout before feed to idcnn layer
            model_inputs = tf.nn.dropout(embedding, self.dropout)  #batch_size*maxlength*[100+20]

            # ldcnn layer
            model_outputs = self.IDCNN_layer(model_inputs)

            # logits for tags
            self.logits = self.project_layer_idcnn(model_outputs)
        
        else:
            raise KeyError

        # loss of the model
        self.loss = self.loss_layer(self.logits, self.lengths)  #self.logits= 60 * max * 51

        with tf.variable_scope("optimizer"):
            optimizer = self.config["optimizer"]
            if optimizer == "sgd":
                self.opt = tf.train.GradientDescentOptimizer(self.lr)
            elif optimizer == "adam":
                self.opt = tf.train.AdamOptimizer(self.lr)
            elif optimizer == "adgrad":
                self.opt = tf.train.AdagradOptimizer(self.lr)
            else:
                raise KeyError

            # apply grad clip to avoid gradient explosion
            grads_vars = self.opt.compute_gradients(self.loss)
            capped_grads_vars = [[tf.clip_by_value(g, -self.config["clip"], self.config["clip"]), v]
                                 for g, v in grads_vars]  #config["clip"]梯度截断【-5,5】范围内
            self.train_op = self.opt.apply_gradients(capped_grads_vars, self.global_step)

        # saver of the model
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    def embedding_layer(self, char_inputs, seg_inputs, config, name=None):
        """
        :param char_inputs: 句子的char_2_ID表示
        :param seg_inputs: segmentation feature
        :param config: wither use segmentation feature
        :return: [1, num_steps, embedding size], 
        """
        #高:3 血:22 糖:23 和:24 高:3 血:22 压:25 char_inputs=[3,22,23,24,3,22,25]
        #高血糖 和 高血压 seg_inputs 高血糖=[1,2,3] 和=[0] 高血压=[1,2,3]  seg_inputs=[1,2,3,0,1,2,3]
        embedding = []
        self.char_inputs_test=char_inputs
        self.seg_inputs_test=seg_inputs
        with tf.variable_scope("char_embedding" if not name else name), tf.device('/cpu:0'):
            self.char_lookup = tf.get_variable(
                    name="char_embedding",
                    shape=[self.num_chars, self.char_dim],
                    initializer=self.initializer)
            #输入char_inputs='常' 对应的字典的索引/编号/value为：8
            #self.char_lookup=[2677*100]的向量，char_inputs字对应在字典的索引/编号/key=[1]
            embedding.append(tf.nn.embedding_lookup(self.char_lookup, char_inputs))
            if config["seg_dim"]: #20
                with tf.variable_scope("seg_embedding"), tf.device('/cpu:0'):
                    self.seg_lookup = tf.get_variable(
                        name="seg_embedding",
                        #shape=[4*20]
                        shape=[self.num_segs, self.seg_dim],
                        initializer=self.initializer)
                    embedding.append(tf.nn.embedding_lookup(self.seg_lookup, seg_inputs))
            embed = tf.concat(embedding, axis=-1)  #(60*max_length*[100+20])
        self.embed_test=embed
        self.embedding_test=embedding
        return embed

    def biLSTM_layer(self, model_inputs, lstm_dim, lengths, name=None):
        """
        :param lstm_inputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, 2*lstm_dim]
        """
        with tf.variable_scope("char_BiLSTM" if not name else name):
            lstm_cell = {}
            for direction in ["forward", "backward"]:
                with tf.variable_scope(direction):
                    lstm_cell[direction] = tf.contrib.rnn.CoupledInputForgetGateLSTMCell(
                        lstm_dim,
                        use_peepholes=True,
                        initializer=self.initializer,
                        state_is_tuple=True)
            outputs, final_states = tf.nn.bidirectional_dynamic_rnn(
                lstm_cell["forward"],
                lstm_cell["backward"],
                model_inputs,
                dtype=tf.float32,
                sequence_length=lengths)
        return tf.concat(outputs, axis=2)

    #IDCNN layer
    def IDCNN_layer(self, model_inputs, 
                    name=None):
        """
        :param idcnn_inputs: [batch_size, num_steps, emb_size] 
        :return: [batch_size, num_steps, cnn_output_width]
        """
        #tf.expand_dims会向tensor中插入一个维度，插入位置就是参数代表的位置（维度从0开始）。
        model_inputs = tf.expand_dims(model_inputs, 1)  #  60,1,max,120
        self.model_inputs_test=model_inputs
        reuse = False
        if self.dropout == 1.0:
            reuse = True
        with tf.variable_scope("idcnn" if not name else name):
            #filter_weights.shape=[1*3*120*100]
            filter_weights = tf.get_variable(
                "idcnn_filter",
                shape=[1, self.filter_width, self.embedding_dim,
                       self.num_filter],
                initializer=self.initializer)
            
            """
            shape of input = [batch, in_height, in_width, in_channels]
            shape of filter = [filter_height, filter_width, in_channels, out_channels]
            """
            layerInput = tf.nn.conv2d(model_inputs,
                                      filter_weights,
                                      strides=[1, 1, 1, 1],
                                      padding="SAME",
                                      name="init_layer",use_cudnn_on_gpu=True)
            self.layerInput_test=layerInput
            finalOutFromLayers = []
            
            totalWidthForLastDim = 0
            for j in range(self.repeat_times): #4
                for i in range(len(self.layers)):#3
                    #1,1,2
                    dilation = self.layers[i]['dilation']
                    isLast = True if i == (len(self.layers) - 1) else False
                    with tf.variable_scope("atrous-conv-layer-%d" % i,
                                           reuse=True
                                           if (reuse or j > 0) else False):
                        #w 卷积核的高度，卷积核的宽度，图像通道数，卷积核个数
                        w = tf.get_variable(
                            "filterW",
                            shape=[1, self.filter_width, self.num_filter,
                                   self.num_filter],
                            initializer=tf.contrib.layers.xavier_initializer())
                        if j==1 and i==1:
                            self.w_test_1=w
                        if j==2 and i==1:
                            self.w_test_2=w                            
                        b = tf.get_variable("filterB", shape=[self.num_filter])

                        conv = tf.nn.atrous_conv2d(layerInput,
                                                   w,
                                                   rate=dilation,
                                                   padding="SAME")
                        self.conv_test=conv 
                        conv = tf.nn.bias_add(conv, b)
                        conv = tf.nn.relu(conv)
                        if isLast:
                            finalOutFromLayers.append(conv)
                            totalWidthForLastDim += self.num_filter
                        layerInput = conv  
            finalOut = tf.concat(axis=3, values=finalOutFromLayers)
            keepProb = 1.0 if reuse else 0.5
            finalOut = tf.nn.dropout(finalOut, keepProb)
            
            finalOut = tf.squeeze(finalOut, [1])
            finalOut = tf.reshape(finalOut, [-1, totalWidthForLastDim])
            self.cnn_output_width = totalWidthForLastDim
            return finalOut

    def project_layer_bilstm(self, lstm_outputs, name=None):
        """
        hidden layer between lstm layer and logits
        :param lstm_outputs: [batch_size, num_steps, emb_size] 
        :return: [batch_size, num_steps, num_tags]
        """
        with tf.variable_scope("project"  if not name else name):
            with tf.variable_scope("hidden"):
                W = tf.get_variable("W", shape=[self.lstm_dim*2, self.lstm_dim],
                                    dtype=tf.float32, initializer=self.initializer)   #200 * 100

                b = tf.get_variable("b", shape=[self.lstm_dim], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())            #100
                output = tf.reshape(lstm_outputs, shape=[-1, self.lstm_dim*2])
                hidden = tf.tanh(tf.nn.xw_plus_b(output, W, b))

            # project to score of tags
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[self.lstm_dim, self.num_tags],
                                    dtype=tf.float32, initializer=self.initializer)   #100 * 51

                b = tf.get_variable("b", shape=[self.num_tags], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())                #51

                pred = tf.nn.xw_plus_b(hidden, W, b)                

            return tf.reshape(pred, [-1, self.num_steps, self.num_tags])  # 60 * max *51 
    
    #Project layer for idcnn by crownpku
    #Delete the hidden layer, and change bias initializer
    def project_layer_idcnn(self, idcnn_outputs, name=None):
        """
        :param lstm_outputs: [batch_size, num_steps, emb_size] 
        :return: [batch_size, num_steps, num_tags]
        """
        with tf.variable_scope("project"  if not name else name):
            
            # project to score of tags
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[self.cnn_output_width, self.num_tags],
                                    dtype=tf.float32, initializer=self.initializer)            #400 * 51

                b = tf.get_variable("b",  initializer=tf.constant(0.001, shape=[self.num_tags]))   #51

                pred = tf.nn.xw_plus_b(idcnn_outputs, W, b)

            return tf.reshape(pred, [-1, self.num_steps, self.num_tags])

    def loss_layer(self, project_logits, lengths, name=None):
        """
        calculate crf loss
        :param project_logits: [-1, num_steps, num_tags]   60 * max * 51
        :return: scalar loss
        """
        with tf.variable_scope("crf_loss"  if not name else name):
            small = -1000.0
            # pad logits for crf loss
            start_logits = tf.concat([small * tf.ones(shape=[self.batch_size, 1, self.num_tags]), tf.zeros(shape=[self.batch_size, 1, 1])], axis=-1)   #60 * 1 * 52 
            pad_logits = tf.cast(small * tf.ones([self.batch_size, self.num_steps, 1]), tf.float32)   #60 * max * 1  （全-1000）
            logits = tf.concat([project_logits, pad_logits], axis=-1)    # 60 * max * 51   concat  60 * max * 1   axis=-1  result = 60 * max * 52
            logits = tf.concat([start_logits, logits], axis=1)  #start_logits(60*1*52)  logits(60*max*52)  axis=1   result=60*max+1*52
            targets = tf.concat(
                [tf.cast(self.num_tags*tf.ones([self.batch_size, 1]), tf.int32), self.targets], axis=-1)  #60 * 1+max

            self.trans = tf.get_variable(
                "transitions",
                shape=[self.num_tags + 1, self.num_tags + 1],
                initializer=self.initializer)                           # 52 * 52
            #crf_log_likelihood在一个条件随机场里面计算标签序列的log-likelihood
            #inputs: 一个形状为[batch_size, max_seq_len, num_tags] 的tensor,
            #一般使用BILSTM处理之后输出转换为他要求的形状作为CRF层的输入. 
            #tag_indices: 一个形状为[batch_size, max_seq_len] 的矩阵,其实就是真实标签. 
            #sequence_lengths: 一个形状为 [batch_size] 的向量,表示每个序列的长度. 
            #transition_params: 形状为[num_tags, num_tags] 的转移矩阵    
            #log_likelihood: 标量,log-likelihood 
            #transition_params: 形状为[num_tags, num_tags] 的转移矩阵               
            log_likelihood, self.trans = crf_log_likelihood(
                inputs=logits,   #60* max+1 * 52
                tag_indices=targets, #60 *  1+max
                transition_params=self.trans,# 52 * 52
                sequence_lengths=lengths+1)
            return tf.reduce_mean(-log_likelihood)

    def create_feed_dict(self, is_train, batch):
        """
        :param is_train: Flag, True for train batch
        :param batch: list train/evaluate data 
        :return: 结构化数据

        """
        _, chars, segs, tags = batch
        feed_dict = {
            self.char_inputs: np.asarray(chars),
            self.seg_inputs: np.asarray(segs),
            self.dropout: 1.0,
        }
        if is_train:
            feed_dict[self.targets] = np.asarray(tags)
            feed_dict[self.dropout] = self.config["dropout_keep"]
        return feed_dict

    def run_step(self, sess, is_train, batch):
        """
        :param sess: session to run the batch
        :param is_train: a flag indicate if it is a train batch
        :param batch: a dict containing batch data
        :return: batch result, loss of the batch or logits
        """
        feed_dict = self.create_feed_dict(is_train, batch)
        if is_train:
            global_step, loss,_,char_lookup_out,seg_lookup_out,char_inputs_test,seg_inputs_test,embed_test,embedding_test,\
                model_inputs_test,layerInput_test,conv_test,w_test_1,w_test_2,char_inputs_test= sess.run(
                [self.global_step, self.loss, self.train_op,self.char_lookup,self.seg_lookup,self.char_inputs_test,self.seg_inputs_test,\
                 self.embed_test,self.embedding_test,self.model_inputs_test,self.layerInput_test,self.conv_test,self.w_test_1,self.w_test_2,self.char_inputs],
                feed_dict)
            return global_step, loss
        else:
            lengths, logits = sess.run([self.lengths, self.logits], feed_dict)
            return lengths, logits

    def decode(self, logits, lengths, matrix):
        """
        :param logits: [batch_size, num_steps, num_tags]float32, logits
        :param lengths: [batch_size]int32, real length of each sequence
        :param matrix: transaction matrix for inference
        :return:
        """
        # 推断最终标签使用维特比算法
        paths = []
        small = -1000.0
        start = np.asarray([[small]*self.num_tags +[0]])  #（1，52）
        for score, length in zip(logits, lengths):
            score = score[:length]  #  假如length为45 那么我们的score 为 45*51
            pad = small * np.ones([length, 1])  #45 * 1
            logits = np.concatenate([score, pad], axis=1)   #45 * 52
            logits = np.concatenate([start, logits], axis=0)  #46 * 52
            path, _ = viterbi_decode(logits, matrix)  # 46 *52   52  *52

            paths.append(path[1:])
        return paths

    def evaluate(self, sess, data_manager, id_to_tag):
        """
        :param sess: session  to run the model 
        :param data: list of data
        :param id_to_tag: index to tag name
        :return: evaluate result
        """
        results = []
        trans = self.trans.eval()
        for batch in data_manager.iter_batch():
            strings = batch[0]
            tags = batch[-1]
            lengths, scores = self.run_step(sess, False, batch)
            batch_paths = self.decode(scores, lengths, trans)
            for i in range(len(strings)):
                result = []
                string = strings[i][:lengths[i]]
                gold = iobes_iob([id_to_tag[int(x)] for x in tags[i][:lengths[i]]])
                pred = iobes_iob([id_to_tag[int(x)] for x in batch_paths[i][:lengths[i]]])
                #gold = iob_iobes([id_to_tag[int(x)] for x in tags[i][:lengths[i]]])
                #pred = iob_iobes([id_to_tag[int(x)] for x in batch_paths[i][:lengths[i]]])                
                for char, gold, pred in zip(string, gold, pred):
                    result.append(" ".join([char, gold, pred]))
                results.append(result)
        return results

    def evaluate_line(self, sess, inputs, id_to_tag):   #inputs 是4维 跟训练一样
        trans = self.trans.eval(session=sess)   #52 * 52
        lengths, scores = self.run_step(sess, False, inputs) #scores （batch_size*max_length*num_tags)
        batch_paths = self.decode(scores, lengths, trans)
        tags = [id_to_tag[idx] for idx in batch_paths[0]]
        return result_to_json(inputs[0][0], tags)
