import os
import sys
import numpy as np
import tensorflow as tf
import pathlib
import zipfile
from server.plugins.ner.utils.model import Model
from server.plugins.ner.utils.utils import test_ner
from server.plugins.ner.NER_Object import NER_Object
from server.plugins.ner.utils.utils import create_model, save_model
from server.plugins.ner.utils.data_utils import load_word2vec, input_from_line


class NER(NER_Object):

    def __init__(self, job_context, **kwargs):

        super().__init__(job_context)
        self.asserts = {}
        self.job_context = job_context

    def init_model(self):
        ckpt_zip_file = os.path.join(self.base_dir,'model.alphamind.h5')
        with zipfile.ZipFile(ckpt_zip_file, 'r') as zipf: 
            zipf.extractall(self.ckpt_path)

        config = self.load_config_()
        logger = self.get_logger_()
        self.char_to_id, self.id_to_char, self.tag_to_id, self.id_to_tag = self.pickle_map_file()

        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        tf.reset_default_graph()
        self.sess = tf.Session(config=tf_config)
        self.model = create_model(
            self.sess, Model, self.ckpt_path, load_word2vec, config, self.id_to_char, logger)

    def pre_train(self):

        self.pre_data()

    def train(self):
        config = self.load_config_()
        train_manager, logger, char_to_id, id_to_char, tag_to_id, id_to_tag = self.train_data_loader()
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        steps_per_epoch = train_manager.len_data
        tf.reset_default_graph()
        with tf.Session(config=tf_config) as sess:
            model = create_model(sess, Model, self.ckpt_path,
                                 load_word2vec, config, id_to_char, logger)
            logger.info("start training")
            loss = []
            for i in range(self.max_epoch):
                logger.info("Training the %d epoch" % i)
                for batch in train_manager.iter_batch(shuffle=True):
                    step, batch_loss = model.run_step(sess, True, batch)
                    loss.append(batch_loss)
                    if step % self.steps_check == 0:
                        iteration = step // steps_per_epoch + 1
                        logger.info("iteration:{} step:{}/{}, "
                                    "NER loss:{:>9.6f}".format(
                                        iteration, step % steps_per_epoch, steps_per_epoch, np.mean(loss)))
                        loss = []
                logger.info("The {0} epoch finished!".format(i))
            if i == self.max_epoch - 1:
                logger.info("Training end, model in storage.....")
                save_model(sess, model, self.ckpt_path, logger)
                logger.info("Model saved！")
        self.job_context['models_path'] = self.ckpt_path
        
        ckpt_zip_file = os.path.join(self.base_dir,'model.alphamind.h5')
        f = zipfile.ZipFile(ckpt_zip_file, 'w', zipfile.ZIP_DEFLATED)
        for root, dirs, files in os.walk(self.ckpt_path):
            fpath = root.replace(self.ckpt_path,'')
            for name in files:
                path_name = os.path.join(root,name)
                f.write(path_name,os.path.join(fpath, name))
        f.close()
        
    def evaluate(self):

        def evaluate_(sess, model, name, data, id_to_tag, logger):
            logger.info("evaluate:{}".format(name))
            ner_results = model.evaluate(sess, data, id_to_tag)
            eval_lines = test_ner(ner_results, self.result_path)
            for line in eval_lines:
                logger.info(line)
            f1 = float(eval_lines[1].strip().split()[-1])
            if name == "dev":
                best_test_f1 = model.best_dev_f1.eval()
                if f1 > best_test_f1:
                    tf.assign(model.best_dev_f1, f1).eval()
                    logger.info("new best dev f1 score:{:>.3f}".format(f1))
            return f1
        config = self.load_config_()
        logger = self.get_logger_()
        char_to_id, id_to_char, tag_to_id, id_to_tag = self.pickle_map_file()
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        tf.reset_default_graph()
        dev_manager = self.dev_data_loader()
        with tf.Session(config=tf_config) as sess:
            model = create_model(sess, Model, self.ckpt_path,
                                 load_word2vec, config, id_to_char, logger)
            best_score = evaluate_(
                sess, model, "dev", dev_manager, id_to_tag, logger)
            logger.info("Evaluate score：{}".format(best_score))
        return best_score, best_score

    def predict(self, data):
        result = self.model.evaluate_line(self.sess, input_from_line(
            data['content'], self.char_to_id), self.id_to_tag)
        return result
