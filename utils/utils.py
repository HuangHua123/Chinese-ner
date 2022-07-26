import os
import json
import shutil
from logzero import setup_logger

import tensorflow as tf
from .conlleval import return_report



def get_logger(log_file):
    logger = setup_logger(name="mylogger1",logfile=log_file)
    return logger



def test_ner(results, path):
    """
        运行脚本评估模型
    """
    output_file = os.path.join(path, "ner_predict.utf8")
    with open(output_file, "w",encoding='utf8') as f:
        to_write = []
        for block in results:
            for line in block:
                to_write.append(line + "\n")
            to_write.append("\n")

        f.writelines(to_write)
    eval_lines = return_report(output_file)
    return eval_lines


def print_config(config, logger):
    """
        打印config
    """
    for k, v in config.items():
        logger.info("{}:\t{}".format(k.ljust(15), v))


def make_path(result_path, ckpt_path, log, pre_train_data):
    """
        创建训练及推理时文件夹
    """
    if not os.path.isdir(result_path):
        os.makedirs(result_path)
    if not os.path.isdir(ckpt_path):
        os.makedirs(ckpt_path)
    if not os.path.isdir(log):
        os.makedirs(log)
    if not os.path.isdir(pre_train_data):
        os.makedirs(pre_train_data)


def clean(params):
    """
    Clean current folder
    remove saved model and training log
    """
    if os.path.isfile(params.vocab_file):
        os.remove(params.vocab_file)

    if os.path.isfile(params.map_file):
        os.remove(params.map_file)

    if os.path.isdir(params.ckpt_path):
        shutil.rmtree(params.ckpt_path)

    if os.path.isdir(params.summary_path):
        shutil.rmtree(params.summary_path)

    if os.path.isdir(params.result_path):
        shutil.rmtree(params.result_path)

    if os.path.isdir("log"):
        shutil.rmtree("log")

    if os.path.isdir("__pycache__"):
        shutil.rmtree("__pycache__")

    if os.path.isfile(params.config_file):
        os.remove(params.config_file)

    if os.path.isfile(params.vocab_file):
        os.remove(params.vocab_file)


def save_config(config, config_file):
    """
        保存模型配置
        参数以json格式存储
    """
    with open(config_file, "w", encoding="utf8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)


def load_config(config_file):
    """
        模型的负载配置
        参数以json格式存储
    """
    with open(config_file, encoding="utf8") as f:
        return json.load(f)


def convert_to_text(line):
    """
    Convert conll data to text
    """
    to_print = []
    for item in line:

        try:
            if item[0] == " ":
                to_print.append(" ")
                continue
            word, gold, tag = item.split(" ")
            if tag[0] in "SB":
                to_print.append("[")
            to_print.append(word)
            if tag[0] in "SE":
                to_print.append("@" + tag.split("-")[-1])
                to_print.append("]")
        except:
            print(list(item))
    return "".join(to_print)


def save_model(sess, model, path, logger):

    checkpoint_path = os.path.join(path, "ner.ckpt")
    model.saver.save(sess, checkpoint_path)
    logger.info("model saved")


def create_model(session, Model_class, path, load_vec, config, id_to_char, logger):

    # 创建模型，重用参数（如果存在）

    model = Model_class(config)

    ckpt = tf.train.get_checkpoint_state(path)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        logger.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        logger.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        if config["pre_emb"]:
            emb_weights = session.run(model.char_lookup.read_value())
            emb_weights = load_vec(config["emb_file"],id_to_char, config["char_dim"], emb_weights, logger) #预训练Word2vec加进去。
            session.run(model.char_lookup.assign(emb_weights))
            logger.info("Load pre-trained embedding.")
    return model


def result_to_json(string, tags):
    item = {"string": string, "entities": []}
    entity_name = ""
    entity_start = 0
    idx = 0
    for char, tag in zip(string, tags):
        if tag[0] == "S":
            item["entities"].append({"word": char, "start": idx, "end": idx+1, "type":tag[2:]})
        elif tag[0] == "B":
            entity_name += char
            entity_start = idx
        elif tag[0] == "I":
            entity_name += char
        elif tag[0] == "E":
            entity_name += char
            item["entities"].append({"word": entity_name, "start": entity_start, "end": idx + 1, "type": tag[2:]})
            entity_name = ""
        else:
            entity_name = ""
            entity_start = idx
        idx += 1
    return item




