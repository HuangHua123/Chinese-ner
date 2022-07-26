import os
import re
import codecs

from .data_utils import create_dico, create_mapping, zero_digits
from .data_utils import iob2, iob_iobes, get_seg_features


def load_sentences(path, lower, zeros):
    """
		加载句子，行必须至少包含一个单词及其标记。
		句子之间用空行隔开。
    """
    sentences = []
    sentence = []
    num = 0
    for line in open(path, 'r',encoding='utf8'):
        num+=1
        line = zero_digits(line.rstrip()) if zeros else line.rstrip()
        if not line:
            if len(sentence) > 0:
                sentences.append(sentence)
                sentence = []
        else:
            if line[0] == " ":
                line = "$" + line[1:]
                word = line.split()
            else:
                word= line.split( )
            assert len(word) == 2
            sentence.append(word)
    if len(sentence) > 0:
        if 'DOCSTART' not in sentence[0][0]:
            sentences.append(sentence)
    return  sentences


def update_tag_scheme(sentences, tag_scheme):
    """
		检查并更新句子标记方案。
		只接受IOB1和IOB2方案。
    """
    for i, s in enumerate(sentences):
        tags = [w[-1] for w in s]
        # 检查标签是否以IOB格式给出
        if not iob2(tags):
            s_str = '\n'.join(' '.join(w) for w in s)
            raise Exception('Sentences should be given in IOB format! ' +
                            'Please check sentence %i:\n%s' % (i, s_str))
        if tag_scheme == 'iob':
            # 如果格式是IOB1，则转换为IOB2
            for word, new_tag in zip(s, tags):
                word[-1] = new_tag
        elif tag_scheme == 'iobes':
            new_tags = iob_iobes(tags)
            for word, new_tag in zip(s, new_tags):
                word[-1] = new_tag
        else:
            raise Exception('Unknown tagging scheme!')


def char_mapping(sentences, lower):
    """
		创建字典和单词映射，按频率排序。
    """
    chars = [[x[0].lower() if lower else x[0] for x in s] for s in sentences]
    dico = create_dico(chars) 
    dico["<PAD>"] = 10000001   
    dico['<UNK>'] = 10000000
    char_to_id, id_to_char = create_mapping(dico)
    return dico, char_to_id, id_to_char


def tag_mapping(sentences,id_to_tag_path,tag_to_id_path):
    """
        创建按频率排序的字典和标记映射。
    """
    
    f=open(tag_to_id_path,'w',encoding='utf8')
    f1=open(id_to_tag_path,'w',encoding='utf8')
    tags=[]
    for s in sentences:
        ts=[]
        for char in s:
            tag=char[-1]
            ts.append(tag)
        tags.append(ts)
    
    dico = create_dico(tags)
    tag_to_id, id_to_tag = create_mapping(dico)
    for k,v in tag_to_id.items():
        f.write(k+":"+str(v)+"\n")
    for k,v in id_to_tag.items():
        f1.write(str(k) + ":" + str(v) + "\n")
    return dico, tag_to_id, id_to_tag


def prepare_dataset(sentences, char_to_id, tag_to_id, lower=False, train=True):
    """
		准备数据集。返回包含以下内容的词典列表：
		-单词索引
		-单词字符索引
		-标记索引
    """
    none_index = tag_to_id["O"]

    def f(x):
        return x.lower() if lower else x
    data = []
    for s in sentences:
        string = [w[0] for w in s]
        chars = [char_to_id[f(w) if f(w) in char_to_id else '<UNK>']
                 for w in string]
        segs = get_seg_features("".join(string))
        if train:
            tags = [tag_to_id[w[-1]] for w in s]
        else:
            tags = [none_index for _ in chars]
        data.append([string, chars, segs, tags])

    return data


def augment_with_pretrained(dictionary, ext_emb_path, chars, logger):
    """
		用预先训练过的单词扩充词典。
		如果“words”为“None”，则添加每个预训练嵌入的单词
		否则，我们只添加`单词`（通常是开发和测试集中的单词。）
    """
    logger.info('Loading pretrained embeddings from %s...' % ext_emb_path)
    assert os.path.isfile(ext_emb_path)

    # 从文件加载预训练的嵌入从文件加载预训练的嵌入
    pretrained = set([
        line.rstrip().split()[0].strip()
        for line in codecs.open(ext_emb_path, 'r', 'utf-8')
        if len(ext_emb_path) > 0
    ])

	# 我们要么把预先训练好的文件里的每一个字都加进去，
	# 或者只有“单词”列表中给出的单词
	# 我们可以指定一个预训练嵌入
    if chars is None:
        for char in pretrained:
            if char not in dictionary:
                dictionary[char] = 0
    else:
        for char in chars:
            if any(x in pretrained for x in [
                char,
                char.lower(),
                re.sub('\d', '0', char.lower())
            ]) and char not in dictionary:
                dictionary[char] = 0

    word_to_id, id_to_word = create_mapping(dictionary)
    return dictionary, word_to_id, id_to_word



