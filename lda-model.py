from gensim.corpora import Dictionary
from gensim.models import LdaModel
import codecs
import jieba


doc1 = codecs.open('doc1','r',encoding='utf-8')

stopwords = codecs.open('stopwords','r',encoding='utf8').readlines()
stopwords = [ w.strip() for w in stopwords ]
train_set = []
for line in doc1:
    line = list(jieba.cut(line))
    train_set.append([ w for w in line if w not in stopwords ])

print(train_set)
# 构建训练语料
dictionary = Dictionary(train_set)
corpus = [ dictionary.doc2bow(text) for text in train_set]

# lda模型训练
lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=7)
print(lda.print_topics(10))