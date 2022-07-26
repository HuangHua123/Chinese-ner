import os
import csv
import jieba
import pickle
import pathlib
import itertools
import numpy as np
import pandas as pd
import jieba.posseg as pseg
from collections import OrderedDict



from server.plugins.ner.utils.utils import get_logger, make_path
from server.plugins.ner.utils.loader import char_mapping, tag_mapping
from server.plugins.ner.utils.utils import save_config, load_config, test_ner
from server.plugins.ner.utils.loader import load_sentences, update_tag_scheme
from server.plugins.ner.utils.loader import augment_with_pretrained, prepare_dataset
from server.plugins.ner.utils.data_utils import create_input, input_from_line, BatchManager


class NER_Object(object):

    def __init__(self, job_context, **kwargs):

        
        self.tag_schema = "iobes"
        self.script = "conlleval"

        self.lower = False
        self.zeros = True
        self.pre_emb = True
        self.clean = True
        
        self.model_type = job_context["model_type"]
        self.clip = job_context["clip"]
        self.seg_dim = job_context["seg_dim"]
        self.char_dim = job_context["char_dim"]
        self.lstm_dim = job_context["lstm_dim"]

        self.lr = job_context["lr"]
        self.dropout = job_context["dropout"]      
        
        if 'test_prefix' in job_context:
            self.base_dir = job_context['test_prefix']+'tmp'
        else:
            self.base_dir = '/tmp'
        self.max_epoch = job_context["max_epoch"]
        self.optimizer = job_context["optimizer"]
        self.batch_size = job_context["batch_size"]
        self.steps_check = job_context["steps_check"]

        #创建/tmp/ckpt 文件夹，里面存放模型训练，推理时候的中间文件。
        # pathlib.Path('/tmp/ckpt').mkdir(parents=True, exist_ok=True)
        self.map_file = os.path.join(self.base_dir, 'ckpt/maps.pkl')
        self.config_file = os.path.join(
            self.base_dir, 'ckpt/config_file')
        self.id_to_tag_path = os.path.join(
            self.base_dir, 'ckpt/id_to_tag.txt')
        self.tag_to_id_path = os.path.join(
            self.base_dir, 'ckpt/tag_to_id.txt')

        #此为开源的word2vec文件。
        self.emb_file = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), 'vec.txt')

        #创建/tmp/pre_train_data 文件夹，里面存放数据与处理后的训练数据，测试数据。
        # pathlib.Path('/tmp/pre_train_data').mkdir(parents=True, exist_ok=True)

        self.pre_train_data = os.path.join(self.base_dir, 'pre_train_data')
        self.dev_file = os.path.join(
            self.base_dir, 'pre_train_data/example.dev')
        self.test_file = os.path.join(
            self.base_dir, 'pre_train_data/example.test')
        self.train_file = os.path.join(
            self.base_dir, 'pre_train_data/example.train')

        # self.source_data_txt、self.source_data_csv这两个文件是前端传过来的，tmp/Datasets里面是很多txt文件，tmp/train_data里面是实体与实体名称对应的csv文件。
        

        self.source_data_txt = os.path.join(self.base_dir, 'Datasets')
        self.source_data_csv = os.path.join(
            self.base_dir, 'train_data.csv')

        #创建/tmp/log 文件夹，里面存放日志信息。
        # pathlib.Path('/tmp/log').mkdir(parents=True, exist_ok=True)
        # pathlib.Path('/tmp/result').mkdir(parents=True, exist_ok=True)

        self.log_dir = os.path.join(self.base_dir, 'log')
        self.ckpt_path = os.path.join(self.base_dir, 'ckpt')
        self.result_path = os.path.join(self.base_dir, 'result')
        self.log_file = os.path.join(self.log_dir, 'train.log')


    def config_model(self, char_to_id, tag_to_id):
        '''
            配置信息
        '''
        config = OrderedDict()
        config["model_type"] = self.model_type
        config["num_chars"] = len(char_to_id)
        config["char_dim"] = self.char_dim
        config["num_tags"] = len(tag_to_id)
        config["seg_dim"] = self.seg_dim
        config["lstm_dim"] = self.lstm_dim
        config["batch_size"] = self.batch_size

        config["emb_file"] = self.emb_file
        config["clip"] = self.clip
        config["dropout_keep"] = 1.0 - self.dropout
        config["optimizer"] = self.optimizer
        config["lr"] = self.lr
        config["tag_schema"] = self.tag_schema
        config["pre_emb"] = self.pre_emb
        config["zeros"] = self.zeros
        config["lower"] = self.lower
        return config


    def pickle_map_file(self,):
        '''
            加载数据集信息。
        '''
        with open(self.map_file, "rb") as f:
            char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)
        return char_to_id, id_to_char, tag_to_id, id_to_tag


    def load_config_(self,):
        '''
            加载配置文件信息。
        '''
        config = load_config(self.config_file)
        return config


    def get_logger_(self,):
        '''
            日志信息记录。
        '''
        log_path = self.log_file
        logger = get_logger(log_path)
        return logger


    def gen_data(self,):
        '''
            预处理前端传过来数据，数据切分为3部分，训练，测试，推理。
        '''
        logger = logger = self.get_logger_()
        dict_csv_file = self.source_data_csv
        try:
            df = pd.read_csv(dict_csv_file, header=None)
        except:
            logger.error("NOT FIND CSV file!")
        df.columns = ['entity', 'name']
        biaoji = set(df["name"].tolist())


        dev = open(self.dev_file, 'w', encoding='utf8')
        train = open(self.train_file, 'w', encoding='utf8')
        test = open(self.test_file, 'w', encoding='utf8')
        fuhao = set(['。', '?', '？', '!', '！'])
        dics = csv.reader(open(dict_csv_file, 'r', encoding='utf8'))
        for row in dics:
            if len(row) == 2:
                jieba.add_word(row[0].strip(), tag=row[1].strip())
                jieba.suggest_freq(row[0].strip())
        split_num = 0
        for file in os.listdir(self.source_data_txt):
            if file.split(".")[-1] == "txt":
                fp = open(os.path.join(self.source_data_txt, file),
                          'r', encoding='utf8')
                for line in fp:
                    split_num += 1
                    words = pseg.cut(line)
                    for key, value in words:
                        if value.strip() and key.strip():
                            index = str(1) if split_num % 15 < 2 else str(
                                2) if split_num % 15 > 1 and split_num % 15 < 4 else str(3)
                            if value not in biaoji:
                                value = 'O'
                                for achar in key.strip():
                                    if achar and achar.strip() in fuhao:
                                        string = achar+" "+value.strip()+"\n"+"\n"
                                        dev.write(string) if index == '1' else test.write(
                                            string) if index == '2' else train.write(string)
                                    elif achar.strip() and achar.strip() not in fuhao:
                                        string = achar + " " + value.strip() + "\n"
                                        dev.write(string) if index == '1' else test.write(
                                            string) if index == '2' else train.write(string)

                            elif value.strip() in biaoji:
                                begin = 0
                                for char in key.strip():
                                    if begin == 0:
                                        begin += 1
                                        string1 = char+' '+'B-'+value.strip()+'\n'
                                        if index == '1':
                                            dev.write(string1)
                                        elif index == '2':
                                            test.write(string1)
                                        elif index == '3':
                                            train.write(string1)
                                        else:
                                            pass
                                    else:
                                        string1 = char + ' ' + 'I-' + value.strip() + '\n'
                                        if index == '1':
                                            dev.write(string1)
                                        elif index == '2':
                                            test.write(string1)
                                        elif index == '3':
                                            train.write(string1)
                                        else:
                                            pass
                            else:
                                continue
        dev.close()
        train.close()
        test.close()


    def pre_data(self,):
        '''
            make_path():创建训练及推理过程中所需文件夹。
            self.gen_data()：前端传回来的数据进行预处理，切分为三部分，train/dev/test
        '''
        
        make_path(self.result_path, self.ckpt_path, self.log_dir, self.pre_train_data)
        logger = self.get_logger_()
        
        self.gen_data()

        train_sentences = load_sentences(
            self.train_file, self.lower, self.zeros) #加载训练数据

        update_tag_scheme(train_sentences, self.tag_schema)  #转换标注格式
        test_sentences = load_sentences(self.test_file, self.lower, self.zeros)  #加载测试数据
        update_tag_scheme(test_sentences, self.tag_schema)  #转换标注格式
        if os.path.isfile(self.map_file):
            os.remove(self.map_file)
        if self.pre_emb:
            dico_chars_train = char_mapping(train_sentences, self.lower)[0]
            dico_chars, char_to_id, id_to_char = augment_with_pretrained(
                dico_chars_train.copy(),
                self.emb_file,
                list(itertools.chain.from_iterable(
                    [[w[0] for w in s] for s in test_sentences])
                ),logger
            )
        else:
            _c, char_to_id, id_to_char = char_mapping(
                train_sentences, self.lower)

        _t, tag_to_id, id_to_tag = tag_mapping(
            train_sentences, self.id_to_tag_path, self.tag_to_id_path)
        with open(self.map_file, "wb") as f:
            pickle.dump([char_to_id, id_to_char, tag_to_id, id_to_tag], f)

        if os.path.isfile(self.config_file):
            os.remove(self.config_file)
        self.config = self.config_model(char_to_id, tag_to_id)
        save_config(self.config, self.config_file)

    def train_data_loader(self,):
        char_to_id, id_to_char, tag_to_id, id_to_tag = self.pickle_map_file()
        train_sentences = load_sentences(
            self.train_file, self.lower, self.zeros)
        update_tag_scheme(train_sentences, self.tag_schema)

        train_data = prepare_dataset(
            train_sentences, char_to_id, tag_to_id, self.lower
        )
        train_manager = BatchManager(train_data, self.batch_size)

        logger = self.get_logger_()

        return train_manager, logger, char_to_id, id_to_char, tag_to_id, id_to_tag

    def dev_data_loader(self,):
        char_to_id, id_to_char, tag_to_id, id_to_tag = self.pickle_map_file()
        dev_sentences = load_sentences(self.dev_file, self.lower, self.zeros)
        update_tag_scheme(dev_sentences, self.tag_schema)

        dev_data = prepare_dataset(
            dev_sentences, char_to_id, tag_to_id, self.lower
        )
        dev_manager = BatchManager(dev_data, self.batch_size)

        return dev_manager

    def test_data_loader(self,):
        char_to_id, id_to_char, tag_to_id, id_to_tag = self.pickle_map_file()
        test_sentences = load_sentences(self.test_file, self.lower, self.zeros)
        update_tag_scheme(test_sentences, self.tag_schema)
        test_data = prepare_dataset(
            test_sentences, char_to_id, tag_to_id, self.lower
        )
        self.test_manager = BatchManager(test_data, self.batch_size)

        return test_manager
