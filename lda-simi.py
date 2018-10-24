import codecs
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from gensim import corpora, models
import numpy as np
import jieba

##构建embedding字典
def get_dict():
    train = []
    fp = codecs.open(r'C:\Users\yuwei\Desktop\逆天龙神.txt', 'r', encoding='utf8')#文本文件，输入需要提取主题的文档
    stopwords = codecs.open('stopwords','r',encoding='utf-8').readlines()
    stopwords = [w.strip() for w  in stopwords]
    for line in fp:
        line = list(jieba.cut(line))
        train.append([w for w in line if w not in stopwords])
    dictionary = Dictionary(train)
    return dictionary,train

##训练lda模型
def train_model():
    dictionary=get_dict()[0]
    train=get_dict()[1]
    corpus = [ dictionary.doc2bow(text) for text in train]
    lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=7,minimum_probability=0.001)
    #模型的保存/ 加载
    lda.save('test_lda.model')


#计算两个文档的相似度
def lda_sim(s1,s2):
    stopwords = codecs.open('stopwords', 'r', encoding='utf8').readlines()
    stopwords = [w.strip() for w in stopwords]
    lda = models.ldamodel.LdaModel.load('test_lda.model')

    test_doc = []
    for line in s1:
        line = jieba.lcut(line.strip())
        test_doc.extend( w for w in line if w not in stopwords)

    print(test_doc)

    dictionary=get_dict()[0]
    doc_bow = dictionary.doc2bow(test_doc) # 文档转换成bow

    print(doc_bow)
    doc_lda = lda[doc_bow]  # 得到新文档的主题分布
    # 输出新文档的主题分布
    print("第一篇文档的主题分布：")
    print(doc_lda)

    list_doc1 = [i[1] for i in doc_lda]
    print('list_doc1',list_doc1)


    test_doc2 = []
    for line in s2:
        result = jieba.lcut(line.strip())
        test_doc2.extend(w for w in result if w not in stopwords)

    print(test_doc2)

    doc_bow2 = dictionary.doc2bow(test_doc2)  # 文档转换成bow
    print(doc_bow2)
    doc_lda2 = lda[doc_bow2]  # 得到新文档的主题分布
    # 输出新文档的主题分布
    print("第二篇文档的主题分布：")
    print(doc_lda2)
    list_doc2 = [i[1] for i in doc_lda2]

    print('list_doc2',list_doc2)
    try:
        sim = np.dot(list_doc1, list_doc2) / (np.linalg.norm(list_doc1) * np.linalg.norm(list_doc2))
    except ValueError:
        print(ValueError)
        sim=0
    #得到文档之间的相似度，越大表示越相近
    return sim


#train_model()
doc1 = open('doc1','r',encoding='utf-8')
doc2 = open('doc2','r',encoding='utf-8')

result = lda_sim(doc1,doc2)

print(result)