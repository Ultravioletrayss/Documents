# ---------- 文本清洗 ----------
def clean_txt(str_in):
    """文本清洗：去除非字母 + 小写"""
    # 参数
    # str_in : str 输入文本
    # return : str 清洗后文本
    import re
    return re.sub("[^A-Za-z']+", " ", str_in).strip().lower()


# ---------- 文件读取 ----------
def file_opener(p_in, f_n):
    """读取单个文件 → 清洗文本"""
    # 参数
    # p_in  : str 文件路径
    # f_n   : str 文件名
    # return: str 文件内容
    try:
        with open(p_in + f_n, "r", encoding="UTF8") as f:
            l_t = f.read()
        return clean_txt(l_t)
    except:
        print("Can't open", p_in + f_n)
        return ""


def file_crawler(p_in):
    """遍历文件夹 → 构建 DataFrame"""
    # 参数
    # p_in  : str 根目录路径
    # return: pd.DataFrame {'body':文本,'label':标签}
    import os
    import pandas as pd
    m_pd = pd.DataFrame()
    for root, dirs, files in os.walk(p_in, topdown=False):
        if "output" in root:
            continue
        x = root.split("/")  # ['C:', 'Users', 'ultra', 'Desktop', 'data', 'fishing']
        lab_t = x[-1]        # 'fishing'
        tmp_p = root + "/"   # tmp_p = "C:/Users/ultra/Desktop/data/fishing/"
        for name in files:
            tmp_txt = file_opener(tmp_p, name)
            if tmp_txt != "":
                t_pd = pd.DataFrame({"body": tmp_txt, "label": lab_t}, index=[0])
                m_pd = pd.concat([m_pd, t_pd], ignore_index=True)
    return m_pd


# ---------- 文本处理 ----------
def rem_sw(str_in):
    """去停用词"""
    # 参数
    # str_in : str 输入文本
    # return : str 去停用词文本
    from nltk.corpus import stopwords
    sw = set(stopwords.words('english'))
    # 1️⃣ 拆分字符串为单词列表
    words = str_in.split()
    # 例如: str_in = "This is a test"
    # words = ['This', 'is', 'a', 'test']
    
    # 2️⃣ 初始化空列表存储过滤后的单词
    filtered_words = []
    
    # 3️⃣ 遍历每个单词
    for w in words:
        # 4️⃣ 判断是否是停用词
        if w not in sw:
            # 5️⃣ 如果不是停用词，就加入列表
            filtered_words.append(w)
    # filtered_words = ['This', 'test'] (假设 'is' 和 'a' 是停用词)
    
    # 6️⃣ 用空格拼接回字符串
    clean_text = " ".join(filtered_words)#"This" + " " + "test"  →  "This test"
    # clean_text = "This test"
    
    # 7️⃣ 返回最终文本
    return clean_text


def ps_lemma(str_in, sw_in):
    """词干 / 词形还原"""
    # 参数
    # str_in : str 输入文本
    # sw_in  : str 'ps'词干,'lemma'词形还原
    # return : str 处理后的文本
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    if sw_in == "ps":
        ps = PorterStemmer()
        words = str_in.split()    # 先把文本切成单词列表
        new_words = []             # 创建空列表
        for w in words:            # 遍历每个单词
            new_words.append(ps.stem(w))  # 词干化并加入新列表
        return " ".join(new_words)
    else:
        lemma = WordNetLemmatizer()
        words=str_in.split()
        new_words=[]
        for w in words:
            new_words.append(lemma.lemmatize(w))
        return " ".join(new_words)


# ---------- 词频统计 ----------
def word_freq_redux(c_in):
    """词频统计"""
    # 参数
    # c_in  : str 文本
    # return: dict 词频
    import collections
    #c_in = "cat dog cat mouse"
    words=c_in.split()
    ## ['cat', 'dog', 'cat', 'mouse']
    ct=collections.Counter(words)
    ## Counter({'cat': 2, 'dog': 1, 'mouse': 1})Counter 返回的是一个 特殊的字典类型，支持一些额外方法（比如 .most_common()）。
    results=dict(ct)
    return results


def all_dictionary(df_in, col_n):
    """全局 + 分类词频"""
    # 参数
    # df_in : pd.DataFrame
    # col_n : str 列名
    # return: dict {'all':{}, '类别':{}}
    m_dict = dict()
    c_str = df_in[col_n].str.cat(sep=" ")
    m_dict["all"] = word_freq_redux(c_str)
    for top in list(df_in["label"].unique()):
        t_t = df_in[df_in["label"] == top]
        m_dict[top] = word_freq_redux(t_t[col_n].str.cat(sep=" "))
    return m_dict


# ---------- IO ----------
def write_pickle(d_in, n_in, o_p):
    """保存 pickle"""
    # 参数
    # d_in : 任意对象
    # n_in : str 文件名
    # o_p  : str 输出路径
    import pickle, os
    os.makedirs(o_p, exist_ok=True)
    with open(o_p + n_in + '.pkl', 'wb') as f:
        pickle.dump(d_in, f)


def read_pickle(n_in, o_p):
    """读取 pickle"""
    # 参数
    # n_in : str 文件名
    # o_p  : str 文件路径
    # return: 对象
    import pickle
    with open(o_p + n_in + '.pkl', 'rb') as f:
        return pickle.load(f)


# ---------- 向量化 ----------
def vec_fun(df_in, l_in, m, n, sw_in, o_p):
    """文本 → 数值矩阵"""
    # 参数
    # df_in : pd.Series/list 文本
    # l_in  : pd.Series 标签
    # m,n   : int ngram范围
    # sw_in : str 'tf'词频,其他tfidf
    # o_p   : str 输出路径
    # return : DataFrame 数值矩阵, vectorizer
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    import pandas as pd
    if sw_in == "tf":
        cv = CountVectorizer(ngram_range=(m, n))
    else:
        cv = TfidfVectorizer(ngram_range=(m, n))
    x_d = pd.DataFrame(cv.fit_transform(df_in).toarray())
    write_pickle(cv, sw_in, o_p)
    x_d.columns = cv.get_feature_names_out()#给columns横着赋值
    print(cv.get_feature_names_out())
    x_d.index = l_in#l_in是series
    return x_d, cv


# ---------- 分析 ----------
def ran_wrd_draw(df_in, tok_in):
    """计算词概率"""
    # 参数
    # df_in : DataFrame 文本向量矩阵
    # tok_in: str 词
    import pandas as pd
    sum_cols = pd.DataFrame(df_in.sum(axis=0))
    sum_cols.columns = ["token"]
    try:
        tok_cnt = int(sum_cols.at[tok_in, 'token'])
        llh = tok_cnt / int(sum_cols.sum()["token"]) * 100
        print(tok_in, "likelihood", llh, "%")
    except:
        print(tok_in, "does not exist")
        pass


def cos_fun(df_in, lab_in):
    from sklearn.metrics.pairwise import cosine_similarity
    import pandas as pd
    
    cos_fun = pd.DataFrame(cosine_similarity(df_in, df_in))
    cos_fun.columns = lab_in
    cos_fun.index = lab_in
    
    stat_dictionary = dict()
    for t in lab_in.unique():
        tmp = cos_fun[t]
        tmp = tmp[cos_fun.index == t]
        tmp = tmp.mean(axis=0).mean()
        stat_dictionary[t] = float(tmp)
    return stat_dictionary

def chi_fun(df_in, lab_in, k_in, p_in, n_in, stat_sig):
    from sklearn.feature_selection import chi2, SelectKBest
    import pandas as pd
    feat_sel = SelectKBest(score_func=chi2, k=k_in)
    dim_data = pd.DataFrame(feat_sel.fit_transform(df_in, lab_in))
    p_val = pd.DataFrame(list(feat_sel.pvalues_))
    p_val.columns = ["pval"]
    feat_index = list(p_val[p_val.pval <= stat_sig].index)
    dim_data = dim_data[feat_index]
    feature_names = df_in.columns[feat_index]
    dim_data.columns = feature_names
    write_pickle(feat_sel, n_in, p_in)
    #(d_in, n_in, o_p)
    return dim_data, feat_sel

def pca_fun(df_in, e_v, p_in, n_in):
    from sklearn.decomposition import PCA
    import pandas as pd
    pca = PCA(n_components=e_v)
    x_data_t = pd.DataFrame(pca.fit_transform(df_in))
    exp_var = sum(pca.explained_variance_ratio_)
    print ("Explained variance", exp_var)
    write_pickle(pca, n_in, p_in)
    return x_data_t
# ---------- Embedding（老师模块） ----------
def extract_embeddings_pre(df_in, out_path_i, name_in):
    """
    【词向量 Embedding】
    👉 功能：
        将文本转换为 word2vec 平均向量

    👉 输入：
        df_in : 文本列（Series）
        name_in : 词向量文件路径

    👉 输出：
        数值矩阵 + 模型
    """
    import pandas as pd
    from nltk.data import find
    from gensim.models import KeyedVectors
    import pickle
    import numpy as np

    def get_score(var):
        tmp_arr = []
        for word in var:
            try:
                tmp_arr.append(list(my_model_t.get_vector(word)))
            except:
                pass
        return np.mean(np.array(tmp_arr), axis=0)

    word2vec_sample = str(find(name_in))

    my_model_t = KeyedVectors.load_word2vec_format(
        word2vec_sample, binary=False
    )

    tmp_out = df_in.str.split().apply(get_score)
    tmp_data = tmp_out.apply(pd.Series).fillna(0)

    pickle.dump(my_model_t, open(out_path_i + "embeddings.pkl", "wb"))
    pickle.dump(tmp_data, open(out_path_i + "embeddings_df.pkl", "wb"))

    return tmp_data, my_model_t



# ======================
# 主程序
# ======================

# 1. 路径设置
data_path = "C:/Users/ultra/Desktop/data/"
out_path = "C:/Users/ultra/Desktop/data/output/"

# ======================
# 第一阶段：数据准备 (顺序执行)
# ======================
print(">>> 正在读取数据...")
# 1. 爬数据
the_data = file_crawler(data_path)

# 2. 去停用词
the_data["body_sw"] = the_data["body"].apply(rem_sw)

# 3. 词干提取
the_data["body_sw_stem"] = the_data["body_sw"].apply(
    lambda x: ps_lemma(x, "ps")
)

# 4. 词形还原
the_data["body_sw_lemma"] = the_data["body_sw"].apply(
    lambda x: ps_lemma(x, "lemma")
)

# 5. 保存清洗后的数据
write_pickle(the_data, "the_data", out_path)

# 6. 词频统计 (仅用于查看，不影响后续流程)
all_d_sw_stem = all_dictionary(the_data, "body_sw_stem")


# ======================
# 第二阶段：特征提取 (二选一：并列关系)
# ======================
# 说明：这里有两个选择，一次只能运行一个。
# 选择 A：传统方法 (TF-IDF/词袋) -> 适合短文本，可解释性强
# 选择 B：词向量方法 (Word2Vec) -> 适合语义理解，包含上下文信息

x_data = None # 初始化变量

# --- 选择 A：传统向量化 (vec_fun) ---
# 如果你想用原来的方法，请确保下面这行代码没有被注释
print(">>> 正在执行：传统向量化 (TF/TF-IDF)...")
x_data, vectorizer = vec_fun(
    the_data["body_sw"], # 使用去停用词后的文本
    the_data["label"],
    1, 2,                # ngram范围 (1,2)
    "tf",                # 模式：'tf'为词频，其他为tfidf
    out_path
)

# --- 选择 B：词向量嵌入 (extract_embeddings_pre) ---
# 如果你想用老师新加的方法，请注释掉上面的 vec_fun，并取消下面这部分的注释
"""
print(">>> 正在执行：Word2Vec 词向量嵌入...")
# 注意：这里使用的是原始文本列 'body' 或者 'body_sw' 都可以，Word2Vec通常处理分词后的列表更好
# 但根据函数定义，它内部会做 split，所以传字符串即可
x_data, emb_mod = extract_embeddings_pre(
    the_data["body"], 
    out_path, 
    'models/word2vec_sample/pruned.word2vec.txt' # 确保这个模型文件路径存在
)
"""

# ======================
# 第三阶段：特征优化 (可选：顺序执行)
# ======================
# 说明：这一步是在 x_data 生成之后，对其进行“瘦身”或“提纯”。
# 只有当你确定了 x_data 是什么之后，才能运行这里。

# --- 优化选项 1：卡方检验特征选择 (chi_fun) ---
# 作用：剔除不重要的词，只保留对分类最有用的词
# 如果需要使用，请取消下方注释
# --- 选项 1：我想保留重要的词 (推荐用于文本分析) ---
"""
print(">>> 正在执行：卡方检验特征选择...")
x_data, chi_selector = chi_fun(
    x_data, 
    the_data["label"], 
    k_in=1000,       # 想要保留的特征数量 (例如1000个)
    p_in=out_path, 
    n_in="chi_model", 
    stat_sig=0.05    # 显著性水平
)
"""

# --- 优化选项 2：PCA 降维 (pca_fun) ---
# 作用：压缩数据维度，减少计算量
# 如果需要使用，请取消下方注释
#--- 选项 2：我想压缩数据 (推荐用于画图或深度学习前处理) ---
"""
print(">>> 正在执行：PCA 降维...")
x_data = pca_fun(
    x_data, 
    e_v=0.95,    # 保留 95% 的信息量
    p_in=out_path, 
    n_in="pca_model"
)
"""

# ======================
# 第四阶段：分析与输出 (顺序执行)
# ======================
print(">>> 流程结束，正在分析结果...")

# 8. 分析特定词概率
ran_wrd_draw(x_data, "fishes")

# 9. 计算余弦相似度 (查看文档间相似性)
c_s = cos_fun(x_data, the_data["label"])
print(">>> 相似度统计完成。")