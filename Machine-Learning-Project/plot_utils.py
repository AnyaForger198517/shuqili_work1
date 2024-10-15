from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

def wordcloud(word_dict):
    wc = WordCloud(
        background_color="white",
        font_path="C:/Windows/Fonts/simkai.ttf",
        max_words=600,
    )
    wc.generate_from_frequencies(word_dict)
    # print(word_dict)
    plt.axis("off")
    # plt.imshow(wc, interpolation='bilinear')
    wc.to_file(os.path.join(os.getcwd(), "wordcloud.png"))
    return None

'''
目标：给出相应任务结果，训练集和测试集精度，以及特征降维后的可视化
使用5种传统机器学习方法进行二分类：决策树、SVM、KNN、朴素贝叶斯、逻辑回归
'''
# 总共画5个图就行，分3个画图函数（精度、召回率、准确度），存储为三张图片，前两张图片含有2个子图，都是训练集和测试集对比
def plotting_precision(name_list, pos_train, pos_test, neg_train, neg_test):
    x = np.arange(5)
    width = 0.25  # 柱子的宽度
    plt.figure(figsize=(12,9))
    # 计算每个柱子在x轴上的位置，保证x轴刻度标签居中
    # x - width/2，x + width/2即每组数据在x轴上的位置

    # 正类精度
    plt.subplot(121)
    plt.bar(x - width / 2, pos_train, width=0.25, label='train', color="cornflowerblue")
    plt.bar(x + width / 2, pos_test, width=0.25, label='test', color="lightpink")
    plt.xlabel("Method")
    plt.ylabel('Scores')
    plt.xticks(range(x.shape[0]), name_list)
    plt.title("positive-label precision comparison")
    plt.legend()

    # 负类精度
    plt.subplot(122)
    plt.bar(x - width / 2, neg_train, width=0.25, label='train', color="cornflowerblue")
    plt.bar(x + width / 2, neg_test, width=0.25, label='test', color="lightpink")
    plt.xlabel("Method")
    plt.ylabel('Scores')
    plt.xticks(range(x.shape[0]), name_list)
    plt.title("negative-label precision comparison")
    plt.legend()

    # 保存并显示图片
    plt.savefig("./ML_Res/precision comparison.png")
    # plt.show()


def plotting_recall(name_list, pos_train, pos_test, neg_train, neg_test):
    x = np.arange(5)
    width = 0.25  # 柱子的宽度
    plt.figure(figsize=(12,9))
    # 计算每个柱子在x轴上的位置，保证x轴刻度标签居中
    # x - width/2，x + width/2即每组数据在x轴上的位置

    # 正类召回率
    plt.subplot(121)
    plt.bar(x - width / 2, pos_train, width=0.25, label='train', color="royalblue")
    plt.bar(x + width / 2, pos_test, width=0.25, label='test', color="orangered")
    plt.xlabel("Method")
    plt.ylabel('Scores')
    plt.xticks(range(x.shape[0]), name_list)
    plt.title("positive-label recall comparison")
    plt.legend()

    # 负类召回率
    plt.subplot(122)
    plt.bar(x - width / 2, neg_train, width=0.25, label='train', color="royalblue")
    plt.bar(x + width / 2, neg_test, width=0.25, label='test', color="orangered")
    plt.xlabel("Method")
    plt.ylabel('Scores')
    plt.xticks(range(x.shape[0]), name_list)
    plt.title("negative-label recall comparison")
    plt.legend()

    # 保存并显示图片
    plt.savefig("./ML_Res/recall comparison.png")
    # plt.show()
    plt.clf()

# 准确度用折线图画吧，放在同一张图片里
def plotting_acc(name_list, train_acc, test_acc):
    x = np.arange(5)
    plt.axis("on")
    plt.xlabel("Method")
    plt.ylabel('Scores')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xticks(range(x.shape[0]), name_list)
    plt.title("accuracy comparison")

    # 训练集准确度
    plt.plot(x, train_acc, color="r", label="training_set")
    # 测试集准确度
    plt.plot(x, test_acc, color="b",  label="test_set")

    # 保存并显示图片
    plt.legend()
    plt.savefig("./ML_Res/accuracy comparison.png")
    plt.clf()

def plotting_loss_acc(train_loss, test_acc, epoch):
    plt.figure(figsize=(12, 9))
    plt.subplot(121)
    plt.plot(np.arange(epoch), train_loss, color='r', label="train_loss")
    plt.xlabel("epoch")
    plt.ylabel("train_loss")
    plt.title("epoch--train_loss")
    plt.legend()

    plt.subplot(122)
    plt.plot(np.arange(epoch), test_acc, color='b', label="test_acc")
    plt.xlabel("epoch")
    plt.ylabel("test_acc")
    plt.title("epoch--test_acc")
    plt.legend()

    plt.savefig("./DL_Res/loss and acc.png")
    plt.clf()



# 降维可视化，呃呃呃感觉没啥用，唉😔
def visualize(X, y):
    # 颜色设置
    color = ['r', 'b']

    # PCA降维（一种矩阵分解的方法）
    pca = PCA(n_components=3, random_state=42)
    result = pca.fit_transform(X)
    # 归一化处理
    scaler = MinMaxScaler(feature_range=(-2., 2.), copy=True)
    result = scaler.fit_transform(result)
    # 可视化展示
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    # 标题，范围，标签设置
    ax.set_title('PCA process')
    ax.set_xlim((-2.1, 2.1))
    ax.set_ylim((-2.1, 2.1))
    ax.set_zlim((-2.1, 2.1))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # 绘制散点图（直接在对应位置上画出标签值）
    for i in range(len(result)):
        ax.text(result[i, 0], result[i, 1], result[i, 2], str(y[i]),
                color=color[y[i]], fontdict={'weight': 'bold', 'size': 9})

    # plt.show()
    plt.savefig("./DL_Res/PCA_Visualization.png")
    plt.clf()