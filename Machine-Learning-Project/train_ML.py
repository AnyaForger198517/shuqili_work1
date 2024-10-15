import os
import preprocess
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn import neighbors
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from plot_utils import plotting_acc, plotting_recall, plotting_precision, wordcloud


if __name__ == '__main__':
    # 中文分词
    data = preprocess.drop_less_dup()
    res = preprocess.all_process(data)

    # 生成词云图
    wordcloud(preprocess.frequency(res))

    # 划分数据集,使用机器学习方法所需训练集和测试集(分层抽样)
    train_X, test_X, train_Y, test_Y = train_test_split(preprocess.tf_idf_feature(), data.label, random_state=1,
                                                        test_size=0.3, shuffle=True, stratify=data.label)

    # 各种方法得到的评价指标以字典形式返回，用列表存储，便于后续画图处理
    result_train_all = []
    result_test_all = []

    # 决策树分类方法模型
    dtc = DecisionTreeClassifier()
    dtc.fit(train_X, train_Y)
    pre = dtc.predict(test_X)
    print("决策树分类开始")
    train_pre = dtc.predict(train_X)
    # 训练集相关报告
    result_train = classification_report(train_pre, train_Y, target_names=["bad", "good"], output_dict=True)
    # 测试集相关报告
    result_test = classification_report(test_Y, pre, target_names=["bad", "good"], output_dict=True)
    print("决策树分类结束")
    result_train_all.append(result_train)
    result_test_all.append(result_test)

    # SVM分类方法模型
    # 设置参数dual指明是否使用对偶问题求解防止FutureWarning版本问题!
    SVM = svm.LinearSVC(dual=True) #支持向量机分类器LinearSVC
    SVM.fit(train_X, train_Y)
    pre = SVM.predict(test_X)
    print("支持向量机分类开始")
    train_pre = SVM.predict(train_X)
    # 训练集相关报告
    result_train = classification_report(train_pre, train_Y, target_names=["bad", "good"], output_dict=True)
    # 测试集相关报告
    result_test = classification_report(test_Y, pre, target_names=["bad", "good"], output_dict=True)
    print("支持向量机分类结束")
    result_train_all.append(result_train)
    result_test_all.append(result_test)

    '''
    最近邻分类
                  precision    recall  f1-score   support
    
               0       0.25      0.98      0.40       435
               1       1.00      0.74      0.85      4995
    
        accuracy                           0.76      5430
       macro avg       0.62      0.86      0.62      5430
    weighted avg       0.94      0.76      0.82      5430
    
                  precision    recall  f1-score   support
    
               0       0.83      0.17      0.28       732
               1       0.72      0.98      0.83      1596
    
        accuracy                           0.73      2328
       macro avg       0.78      0.58      0.55      2328
    weighted avg       0.76      0.73      0.66      2328
    
    以上都用到了混淆矩阵，其中accuracy是所有样本中分类正确的比例
    macro avg 是各个指标的平均值，weighted avg是加权平均值
    '''

    # KNN分类方法模型
    knn = neighbors.KNeighborsClassifier() #n_neighbors=11
    knn.fit(train_X, train_Y)
    pre = knn.predict(test_X)
    print("最近邻分类开始")
    train_pre = knn.predict(train_X)
    # 训练集相关报告
    result_train = classification_report(train_pre, train_Y, target_names=["bad", "good"], output_dict=True)
    # 测试集相关报告
    result_test = classification_report(test_Y, pre, target_names=["bad", "good"], output_dict=True)
    print("最近邻分类结束")
    result_train_all.append(result_train)
    result_test_all.append(result_test)


    # 朴素贝叶斯分类方法模型
    nb = MultinomialNB()
    nb.fit(train_X, train_Y)
    pre = nb.predict(test_X)
    print("朴素贝叶斯分类开始")
    train_pre = nb.predict(train_X)
    # 训练集相关报告
    result_train = classification_report(train_pre, train_Y, target_names=["bad", "good"], output_dict=True)
    # 测试集相关报告
    result_test = classification_report(test_Y, pre, target_names=["bad", "good"], output_dict=True)
    print("朴素贝叶斯分类结束")
    result_train_all.append(result_train)
    result_test_all.append(result_test)

    # 逻辑回归分类方法模型
    LR = LogisticRegression(solver='liblinear')
    LR.fit(train_X, train_Y)
    pre = LR.predict(test_X)
    print("逻辑回归分类开始")
    train_pre = LR.predict(train_X)
    # 训练集相关报告
    result_train = classification_report(train_pre, train_Y, target_names=["bad", "good"], output_dict=True)
    # 测试集相关报告
    result_test = classification_report(test_Y, pre, target_names=["bad", "good"], output_dict=True)
    print("逻辑回归分类结束")
    result_train_all.append(result_train)
    result_test_all.append(result_test)

    # 各方法正类precision静度条形图
    pos_precision_train = [i["good"]["precision"] for i in result_train_all]
    pos_precision_test = [i["good"]["precision"] for i in result_test_all]
    # 各方法负类precision静度条形图
    neg_precision_train = [i["bad"]["precision"] for i in result_train_all]
    neg_precision_test = [i["bad"]["precision"] for i in result_test_all]
    # 各方法正类recall召回率条形图
    pos_recall_train = [i["good"]["recall"] for i in result_train_all]
    pos_recall_test = [i["good"]["recall"] for i in result_test_all]
    # 各方法负类recall召回率条形图
    neg_recall_train = [i["bad"]["recall"] for i in result_train_all]
    neg_recall_test = [i["bad"]["recall"] for i in result_test_all]
    # 各方法accuracy准确度条形图
    acc_all_train = [i["accuracy"] for i in result_train_all]
    acc_all_test = [i["accuracy"] for i in result_test_all]


    # 制作相应结果表格数据
    measure_train = ['pos_precision_train', 'neg_precision_train', 'pos_recall_train', 'neg_recall_train', 'acc_all_train']
    measure_test = ['pos_precision_test', 'neg_precision_test', 'pos_recall_test', 'neg_recall_test', 'acc_all_test']
    method_name = ["Decision_Tree", "SVM", "KNN", "Bayes", "logistic_regression"]
    train_data = [pos_precision_train, neg_precision_train, pos_recall_train, neg_recall_train, acc_all_train]
    test_data = [pos_precision_test, neg_precision_test, pos_recall_test, neg_recall_test, acc_all_test]
    print("----------------------------------------------------")
    print("Train")
    print("----------------------------------------------------")
    print("\t\t\t\t\t\tDecision_Tree\t\t\t\tSVM\t\t\t\tKNN\t\t\t\tBayes\t\t\t\tlogistic_regression")
    for i in range(5):# 控制指标
        print(measure_train[i], end="\t\t\t")
        for j in range(5):# 控制方法
            print(train_data[i][j], end="\t")
        print("\n")
    print("----------------------------------------------------")
    print("Test")
    print("----------------------------------------------------")
    print("\t\t\t\t\t\tDecision_Tree\t\t\t\tSVM\t\t\t\tKNN\t\t\t\tBayes\t\t\t\tlogistic_regression")
    for i in range(5):# 控制指标
        print(measure_test[i], end="\t\t\t")
        for j in range(5):# 控制方法
            print(test_data[i][j], end="\t")
        print("\n")



    # 警告：FutureWarning: The default value of `dual` will change from `True` to `'auto'` in 1.5. Set the value of `dual` explicitly to suppress the warning.

    # print(result_train_all)
    # print(result_test_all)

    # 调用plot_utils函数画出相应精度、召回率、准确度比较图
    # method_name = ["Decision_Tree", "SVM", "KNN", "Bayes", "logistic_regression"]
    if os.path.exists(os.path.join(os.getcwd(), "ML_Res")):
        print("使用机器学习方法获得的图像结果数据文件已经存在！")
    else:
        os.mkdir(os.path.join(os.getcwd(), "ML_Res"))
        plotting_acc(method_name, acc_all_train, acc_all_test)
        plotting_recall(method_name, pos_recall_train, pos_recall_test, neg_recall_train, neg_recall_test)
        plotting_precision(method_name, pos_precision_train, pos_precision_test, neg_precision_train, neg_precision_test)

