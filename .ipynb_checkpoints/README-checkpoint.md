## 简介
    命名实体识别（Named Entity Recognition，NER）可以自动的从文本数据中识别出特定类型的命名实体。即在文本中标识命名实体并划分到相应的实体类型中,通常实体类型包括人名、地名、组织机构名、日期等。
    举例说明,“当地时间 14 日下午,叙利亚一架军用直升机在阿勒坡西部乡村被一枚恶意飞弹击中。”这句话中包含的实体有:日期实体“14 日下午”、组织机构实体“叙利亚”、地名实体“ 阿勒坡西部乡村”、装备实体“军用直升机”和“飞弹”。


## 模型
    随着深度学习不断发展，深度学习模型也开始广泛应用于命名实体识别之中。句子比较长的时候，选用Bert模型；句子较短的时候，选择LSTM，BiLSTM，CNN模型。在句子不怎么长的情况下，用BiLSTM+CRF模型。
    本项目提供膨胀卷积+CRF，BiLSTM+CRF模型供用户选择
    
## demo类别
    数据分为10个标签类别，分别为: 
    地址（address），
    书名（book），
    公司（company），
    游戏（game），
    政府（goverment），
    电影（movie），
    姓名（name），
    组织机构（organization），
    职位（position），
    景点（scene）
    
## NER标签体系
    这里记录两种比较常用的NER的标签体系：BIO和BIOES。
**1.BIO：**

    B: begin，实体名称的开头字符。弱实体名称为单字，也是用这个符号。
    I:inside，实体名称的中间或末尾字符。
    O:outside，不是实体名称的字符。
    
**2.BIOES：**

    E:end，实体名称的结尾字符。
    S:single,单字成一个命名实体。
    
    本项目使用BIOES标注格式

## 参数
```py
job_context = {
    
    'test_prefix':'./',     #   tmp文件夹所在目录
    'model_type':'idcnn',    #   backbone，可选“idcnn", "bilstm"
    'clip':5,           #   梯度截断参数，取值范围在[-5,5]之间，大于5取值5，小于-5取值-5。
    'seg_dim':20,        #   序列标签的维度，先把句子分词，若为单个字则序列为0，若为多个字组成的词语,则词语序列为
                    #   tmp = [2]*len(word),tmp[0] = 1, tmp[-1] = 3
                    #   例：我 /爱 /中国，那么序列为[0,0,1,3]
    'char_dim':100,      #   每个汉字的编码维度
    'lstm_dim':100,      #   LSTM中cell的维度
    'dropout': 0.5,      #   丢弃网络中神经元的比例
    'batch_size': 60,    #    每次放入模型中数据的尺寸
    'lr': 0.001,       #    学习率，影响模型收敛的参数
    'optimizer': 'adam',  #   优化器，可选优化器 “sgd", "adam", "adgrad"
    'max_epoch': 50,    #    最大训练次数
    "steps_check": 100  #    每100个step打印一次loss信息
    
}
```

## 环境需求
    python==3.6.8
    tensorflow==1.13.1
    numpy==1.18.5
    jieba==0.42.1
    pandas==1.0.3

## 数据
    **tmp/Datasets:  原始数据文件夹，不主动创建，需要被提供，该文件夹下子文件格式为 "xxx.txt"**
    **tmp/train_data.csv:原始数据文件，不主动创建，需要被提供，该文件是实体与实体名称对应的csv文件。**

    csv文件夹格式为：（第一列为实体，第二列为实体英文缩写）
            美国,address
            布鲁京斯研究所桑顿中国中心,organization
            李成,name
            研究部主任,position
            unicef,organization
            联合国儿童基金会,organization
            罗马,organization
            英雄联盟,game
            吴湖帆,name
            吴待秋,name
            ......
    
    **tmp/ckpt:  训练中创建的文件夹，主动创建，不需要被提供，该文件夹下存放模型训练，推理过程中的文件。
    tmp/log: 训练中创建的文件夹，主动创建，不需要被提供，该文件夹下存放日志文件。
    tmp/pre_train_data:  训练中创建的文件夹，主动创建，不需要被提供，该文件夹下存放处理后的训练数据，测试数据。
    tmp/result:  训练中创建的文件夹，主动创建，不需要被提供，该文件夹下存放测试结果文件。**

## 使用
    初始化实例：
        train_model = NER(job_context)
    预处理数据：
        train_model.pre_train()
        预处理数据后，数据存入tmp/pre_train_data文件夹，含有（example.dev、example.test、example.train）三个文件。
    模型训练：
        train_model.train()
        模型训练后，模型文件存入tmp/ckpt文件夹，名称为 ner.ckpt.***
    模型推理：
        train_model.evaluate()
        模型推理使用的推理数据为example.dev，返回结果为best_score的数值。

## 预测
    初始化实例：
        predict_model = NER(job_context)
    初始化权重：
        predict_model.init_model()
    传入数据预测：
        result = predict_model.predict({
                'content': '九寨沟真是一个好地方'
            })
        返回结果：
            {'string': '九寨沟真是一个好地方', 'entities': [{'word': '九寨沟', 'start': 0, 'end': 3, 'type': 'scene'}]}




