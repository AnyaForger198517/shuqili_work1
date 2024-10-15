import numpy as np
import plot_utils
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
import preprocess

# 读取数据
def read_data(file_label, file_corpus, test_size):
    all_texts = []
    # print(file_label.values)
    # 注意使用values函数得到去除对应原本行数的信息，否则后续会有KeyError!!
    all_labels = file_label.values
    for line in open(file_corpus, 'r', encoding='utf-8'):
        all_texts.append(line.split())
        # print(line.split())
        # print(type(line.split()))

    # 分割数据集
    train_X, test_X, train_Y, test_Y = train_test_split(all_texts, all_labels, random_state=1,
                                                        test_size=test_size, shuffle=True, stratify=all_labels)

    # 注意打乱样本顺序，训练过程中也进行二分类
    return train_X, train_Y, test_X, test_Y


# 构建词编码，即一个分词对应一个数字
def build_corpus(texts):
    word_2_index = {'UNK': 0, 'PAD': 1}
    for text in texts:
        for word in text:
            if word not in word_2_index:
                word_2_index[word] = len(word_2_index)
    return word_2_index, list(word_2_index)


# 如果要使用word2vec，就要根据word2vec的结果重新构建word_2_index，保证索引结果的正确性
def build_corpus_w2v(word_vec):
    word_2_index = {'UNK': 0, 'PAD': 1}
    for key, index in word_vec.key_to_index.items():
        word_2_index[key] = index+2
    return word_2_index, list(word_2_index)

# 构建数据类，重写自己的数据类
class TextDataset(Dataset):
    def __init__(self, all_texts, all_labels, word_2_index, max_len, ):
        self.all_texts = all_texts
        self.all_labels = all_labels
        self.word_2_index = word_2_index
        # max_len是限制文本最大长度（即一个评论的分词数量）
        self.max_len = max_len

    # 按照索引item获取每个元素的内容，注意返回值都是tensor
    def __getitem__(self, item):
        text = self.all_texts[item][:self.max_len]
        # dict.get函数返回指定键的值，如果指定键不存在，那么在下例中返回0
        text_idx = [self.word_2_index.get(i, 0) for i in text]
        # PADDING!如果某个评论的分词数量过少，小于max_len，那么在保证不同评论的输入向量大小和输出大小维度各自相同情况下，必须padding
        text_idx = text_idx + [1] * (self.max_len - len(text))
        label = int(self.all_labels[item])
        return torch.tensor(text_idx), torch.tensor(label)

    def __len__(self):
        return len(self.all_texts)


# 构建模型
class Block(nn.Module):
    def __init__(self, out_channel, max_lens, kernel_s, embed_num):
        super(Block, self).__init__()
        # 这里out_channel是卷积核的个数，此处kernel_s指的是卷积核的宽度，embed_num即长度，也是词嵌入的维度
        self.cnn = nn.Conv2d(1, out_channel, kernel_size=(kernel_s, embed_num))
        self.act = nn.ReLU()
        # 使用一维最大池化层，其中kernel_size指的是滑动窗口大小
        # self.maxp1 = nn.MaxPool1d(kernel_size=(max_lens - kernel_s + 1))
        self.maxp1 = nn.AvgPool1d(kernel_size=(max_lens - kernel_s + 1))

    def forward(self, emb):
        # emb's size: torch.Size([32, 1, 32, 50])达到输入通道要求
        # print("emb's size:", emb.shape)
        output = self.cnn(emb)
        # output's size: ([32, 2, 31, 1])
        output1 = self.act(output)

        # 一维上的最大池化，所以([32, 2, 31, 1])最后的1需要去掉
        output1 = output1.squeeze(3)
        output2 = self.maxp1(output1)
        # output2's size: ([32, 2, 1])
        # 返回之前将最后维度删除，便于后续concat。数据量足够的情况下，返回维度大小torch.Size([32, 2])
        return output2.squeeze(dim=-1)


class TextCnnModel(nn.Module):
    def __init__(self, vocab_num, out_channel, max_lens, embed_num, class_num):
        super(TextCnnModel, self).__init__()
        self.emb = nn.Embedding(vocab_num, embed_num)
        # 使用已经训练好的词向量
        # UNK  and  PAD embedding! self_define??
        UNK = torch.rand(embed_num)
        PAD = torch.zeros(embed_num)
        word_vector.vectors = np.insert(word_vector.vectors, 0, PAD)
        word_vector.vectors = np.insert(word_vector.vectors, 0, UNK)
        self.emb.weight.data.copy_(torch.from_numpy(word_vector.vectors.reshape(-1, embed_num)))
        self.emb.weight.requires_grad = False
        # 使用三层同级卷积操作，之后concat输入全连接层进行分类
        self.block1 = Block(out_channel, max_lens, 2, embed_num)
        self.block2 = Block(out_channel, max_lens, 3, embed_num)
        self.block3 = Block(out_channel, max_lens, 4, embed_num)
        self.block4 = Block(out_channel, max_lens, 5, embed_num)

        self.classifier = nn.Linear(4 * out_channel, class_num)
        self.loss_fn = nn.CrossEntropyLoss()
        # self.loss_fn = nn.MSELoss(reduction="sum")
        # self.loss_fn = nn.BCELoss()
    def get_emb(self):
        return self.emb

    def forward(self, batch_idx, batch_label=None):
        '''
        batch_idx: torch.Size([32, 32])
        output: torch.Size([32, 32, 50])
        b1: torch.Size([32, 2])

        :param batch_idx:
        :param batch_label:
        :return:loss
        '''
        output = self.emb(batch_idx)
        output = output.unsqueeze(dim=1)
        b1 = self.block1(output)
        b2 = self.block2(output)
        b3 = self.block3(output)
        b4 = self.block4(output)

        feature = torch.cat([b1, b2, b3, b4], dim=1)
        # 全连接层分类
        pre = self.classifier(feature)

        # 如果是训练阶段
        if batch_label is not None:
            # print("pre_size:", pre.shape)
            # print("batch_label_size:", batch_label.shape)
            # pre_size: torch.Size([40, 2])
            # batch_label_size: torch.Size([40])
            # loss = self.loss_fn(torch.argmax(pre, dim=-1).float(), batch_label.float())
            loss = self.loss_fn(pre, batch_label)
            return loss
        else:
            # 如果是测试阶段
            return torch.argmax(pre, dim=-1)


if __name__ == '__main__':
    batch_size = 32
    max_len = 32
    epochs = 50
    out_channel = 3
    class_num = 2
    embed_num = 80
    lr = 2e-5
    test_size = 0.3
    file_corpus = "Corpus.txt"
    file_label = preprocess.drop_less_dup()['label']

    # 使用word2vec词向量，直接应用到训练过程中，不在进行词向量上的权重更新
    word_vector = preprocess.word2vec()
    # print(word_vector.vectors)
    train_text, train_label, test_text, test_label = read_data(file_label, file_corpus, test_size)
    # word_2_index, _ = build_corpus(train_text)
    word_2_index, _ = build_corpus_w2v(word_vector)
    # print("len of word_2_index:", len(word_2_index))# 27727
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_set = TextDataset(train_text, train_label, word_2_index, max_len)
    train_loader = DataLoader(train_set, batch_size)

    test_set = TextDataset(test_text, test_label, word_2_index, max_len)
    test_loader = DataLoader(test_set, batch_size)

    model = TextCnnModel(len(word_2_index), out_channel, max_len, embed_num, class_num).to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr)

    train_loss = []
    test_acc = []
    for e in range(epochs):
        # 训练模式
        model.train()
        # tqdm执行过程中使用了Dataloader类中的__iter__和__next__函数，即构造了迭代器进行batch-size==32大小的索引
        loss = 0
        loss_all = 0
        for batch_idx, batch_label in tqdm(train_loader):
            loss = model(batch_idx.to(device), batch_label.to(device))
            loss_all += loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        train_loss.append(loss_all/len(train_text))
        print(f'epoch:{e},loss={loss:.3f}')

        # 测试模式
        model.eval()
        right_num = 0
        pre = []
        for batch_idx, batch_label in tqdm(test_loader):
            pre = model(batch_idx.to(device))
            batch_label = batch_label.to(device)
            right_num += torch.sum(torch.tensor(pre == batch_label))
        acc = right_num / len(test_text)
        test_acc.append(acc)
        print(f'acc = {acc * 100:.3f}%')

    if os.path.exists(os.path.join(os.getcwd(), "DL_Res")):
        print("使用深度学习方法获得的图像结果数据文件已经存在！")
    else:
        # 画出对应训练过程的损失和测试精度
        os.mkdir(os.path.join(os.getcwd(), "DL_Res"))
        train_loss = torch.tensor(train_loss).detach().numpy()
        test_acc = torch.tensor(test_acc).detach().numpy()
        plot_utils.plotting_loss_acc(train_loss, test_acc, epochs)

        # 画出特征降维后的可视化
        embedding = model.get_emb()
        plot_X = []
        plot_y = test_label
        for i in range(len(test_text)):
            index, _ = test_set.__getitem__(i)
            plot_X.append(embedding(index))
        '''
        UserWarning: Creating a tensor from a list of numpy.ndarrays is extremely slow. 
        Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor. 
        '''
        plot_X = np.array(plot_X)
        # plot_X = torch.tensor(plot_X)
        '''
        ValueError: Found array with dim 3. PCA expected <= 2.
        '''
        c, x, y = plot_X.shape
        plot_X = plot_X.reshape(c, x * y)
        plot_utils.visualize(plot_X, plot_y)
