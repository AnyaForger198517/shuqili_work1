### 项目结构简要说明 ###
***
**Final_Homework**
- ML_Res    
        accuracy comparison.png 训练集和测试集的精确度折线图  
        precision comparison.png 训练集和测试集的精度条形统计图  
        recall comparison.png 训练集和测试集的召回率条形统计图  


- DL_Res  
        loss and acc.png 训练过程的损失和测试准确度  
        PCA_Visualization PCA数据降维可视化结果  


- wordcloud.png 根据分词结果绘制的词云图  
  

- ChnSentiCorp_htl_all.csv 原始酒店评论数据文件 
- cn_stopwords.txt 分词使用的停用词
- Corpus.txt 分词结果  


- plot_utils.py 画图函数所在文件
- preprocess.py 文本预处理（去重、分词、计算词频率等操作）
- train_ML.py 使用5个传统机器学习方法进行的文本情感二分类
- train_DL 使用深度学习方法（TextCNN）进行的文本情感二分类  

### 项目运行说明 ###
***
在项目路径下打开终端输入python train_ML.py执行机器学习文本二分类过程  
在项目路径下打开终端输入python train_DL.py执行深度学习文本二分类过程 

<font color="red">注意：生成词云前需确保本机在C:/Windows/Fonts路径下存在simkai.ttf字体配置文件，为避免代码在  
他人主机上无法运行，已将相关词云生成代码注释，若仍生成词云，需将train_ML.py文件20行代码取消注释重新运行。</font>  

### 项目使用package版本说明 ###
***
gensim              4.3.2  
ipython             8.18.1  
jieba               0.42.1  
matplotlib          3.8.2  
numpy               1.26.3  
packaging           23.2  
pandas              2.2.1  
Pillow              10.0.1  
pip                 23.3.1  
prompt-toolkit      3.0.43  
scikit-learn        1.4.1.post1  
scipy               1.12.0  
tqdm                4.66.4  
wheel               0.41.2  
wordcloud           1.9.3  

