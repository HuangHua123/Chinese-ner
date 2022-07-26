from server.plugins.ner.NER import NER

job_context = {
    
    'test_prefix':'./',     #   tmp文件夹所在目录
    'model_type':'idcnn',   #   backbone，可选“idcnn", "bilstm"
    'clip':5,           #   梯度截断参数，取值范围在[-5,5]之间，大于5取值5，小于-5取值-5。
    'seg_dim':20,        #   序列标签的维度
    'char_dim':100,      #   每个汉字的编码维度
    'lstm_dim':100,      #   LSTM中cell的维度
    'dropout': 0.5,      #   丢弃网络中神经元的比例
    'batch_size': 60,    #    每次放入模型中数据的尺寸
    'lr': 0.001,       #    学习率，影响模型收敛的参数
    'optimizer': 'adam',  #   优化器，可选优化器 “sgd","adam","adgrad"
    'max_epoch': 1,    #    最大训练次数
    "steps_check": 100  #    每100个step打印一次loss信息
    
}
train_model = NER(job_context)
train_model.pre_train()
train_model.train()
train_model.evaluate()

predict_model = NER(job_context)
predict_model.init_model()
result = predict_model.predict({
    'content': '九寨沟真是一个美丽的地方。'
})
print(result)
