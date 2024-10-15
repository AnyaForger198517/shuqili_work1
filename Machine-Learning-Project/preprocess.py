# coding:utf-8
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import jieba
import pandas as pd
import re
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import os

# 去除字数过少评价和重复评价
def drop_less_dup():
    raw_data = pd.read_csv("./ChnSentiCorp_htl_all.csv")
    # 原始数据集（7766，2） 第一列存储类别，1为好，0为差，第二列存储评价文本
    print("原始数据集包含{neg}个负样本，{pos}个正样本".format(neg=raw_data[raw_data.label == 0].shape[0],
                                                             pos=raw_data[raw_data.label == 1].shape[0]))
    # 删除评价字数少于8的行数据
    index = []
    for i in range(raw_data.values.shape[0]):
        tmp = str(raw_data.values[i][1])
        if len(tmp) < 8:
            index.append(i)

    data = raw_data.drop(index=index, axis=0)
    # 去除可能的重复评价
    data.drop_duplicates(inplace=True)
    print("简单处理后数据集包含{neg}个负样本，{pos}个正样本".format(neg=data[data.label == 0].shape[0],
                                                                   pos=data[data.label == 1].shape[0]))
    return data


def per_process(sentences):
    # 无用标点符号
    del_mark = "[0-9a-zA-Z\s+\.\!\/_,$%^*()?;；：【】+\"\'\[\]\\]+|[+——！，;:。？《》、~@#￥%……&*（）“”.=-]+"
    # 停用词
    stopwords = set(open('./cn_stopwords.txt', encoding='utf-8').read().split())
    # 去除标点
    post_del = re.sub(del_mark, ' ', sentences)
    # 分词结果列表
    seg_list = list(jieba.cut(post_del))
    seg_list = [word for word in seg_list if word != ' ' and word not in stopwords]
    return seg_list

def all_process(data):
    mark = "[\[\]\，,\'\"’“]+"
    res = []
    for t in range(data.values.shape[0]):
        res.append(per_process(data.values[t, 1]))
    # print(type(str(res[1])))
    if os.path.exists(os.path.join(os.getcwd(), "Corpus.txt")):
        # 如果文件已经存在，进行提醒
        print("Corpus.txt文件在之前训练过程中已经生成！如需重新生成请手动删除！")
    else:
        # 生成语料库文件
        with open("Corpus.txt", "a+", encoding="utf-8") as f:
            for i in range(len(res)-1):
                f.write(re.sub(mark, " ", str(res[i]))+"\n")
            f.write(re.sub(mark, " ", str(res[len(res)-1])))
            print("已生成Corpus.txt文件")
    return res

# 根据分词结果生成词云频率字典
def frequency(res):
    frequency_dict = {}
    for sentence in res:
        for word in sentence:
            if word in frequency_dict.keys():
                frequency_dict[word] += 1
            else:
                frequency_dict[word] = 1
    return frequency_dict


# 文本特征提取（使用两种模型抽取特征）
# 1.word2vec模型(不一次性读入所有句子，使用迭代器按行读入)，使用深度学习算法
# 不一次性读入所有句子，内存负担太重
# Iterate over a file that contains sentences: one line = one sentence.
# Words must be already preprocessed and separated by whitespace.
def word2vec():
    # sentences = MySentences(os.getcwd())  # 自定义内存友好的迭代器
    # 也可以直接使用LineSentence
    model = Word2Vec(LineSentence("Corpus.txt"), min_count=1, epochs=50, vector_size=80, seed=42)
    # 返回词向量字典用于划分数据集
    return model.wv

'''
2.TF-IDF模型，使用传统机器学习算法（得到的向量可能会非常稀疏？）
字词的重要性随着它在文档中出现的次数成正比增加，但同时会随着它在语料库中出现的频率成反比下降。
原理：简单的解释为，一个单词在一个文档中出现次数很多，同时在其他文档中出现此时较少，那么我们认为这个单词对该文档是非常重要的
tf-idf 倾向于过滤掉常见的词语，保留重要的词语
将文本中的词语转换为词频矩阵
'''
class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for line in open(os.path.join(self.dirname, "Corpus.txt"), encoding="utf-8"):
            yield str(line.split())

def tf_idf_feature():

    # 创建词频计算类
    vectorizer = CountVectorizer(stop_words=None, lowercase=False)

    # 计算词语出现的次数
    sentences = MySentences(os.getcwd())
    X = vectorizer.fit_transform(sentences.__iter__())

    # 获取所有文档中的关键词
    word = vectorizer.get_feature_names_out()
    # print("不错" in word)

    # 查看词频结果
    df_word = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
    # 查看其中一个词在各个评论中出现的次数（去重，只看大概）
    print("\"不错\"在各个评论中出现的次数（去重后结果）\n", df_word["不错"].drop_duplicates())

    # 类调用,使用idf防止除0
    transformer = TfidfTransformer(smooth_idf=True, norm='l2', use_idf=True)
    # 将计算好的词频矩阵X统计成TF-IDF值
    tfidf = transformer.fit_transform(X)
    # 查看计算的tf-idf(最后使用到的特征向量)
    df_word_tfidf = pd.DataFrame(tfidf.toarray(), columns=vectorizer.get_feature_names_out())
    # print(type(df_word_tfidf))
    # 查看“龙门石窟”的特征
    print("\"龙门石窟\"的TF-IDF特征：\n", df_word_tfidf["龙门石窟"])
    print("\n")
    print("查看\"龙门石窟\"的词频特征：\n", df_word_tfidf["龙门石窟"].drop_duplicates())
    print("\n")

    # return np.array(df_word_tfidf)
    return df_word_tfidf