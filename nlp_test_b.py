# -*- coding: utf-8 -*-
"""
邮件分类程序 (v2.0)
功能：基于朴素贝叶斯算法的垃圾邮件分类系统
"""

# -------------------------- 环境配置 --------------------------
# 基础库
import os
import sys
import warnings
import numpy as np  # ← 新增导入
# 数据处理
import pandas as pd
# 可视化
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import make_interp_spline
# 文本处理
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
# 机器学习
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV


# 配置警告过滤（实际应用中存在无法解决的警告）
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module="sklearn")

plt.rcParams.update({
    'savefig.dpi': 600 , # 图像输出质量
    'font.sans-serif': 'SimHei',  # 字体
    'axes.unicode_minus': False,  # 负号显示
    'figure.figsize': (10, 6),  # 画布尺寸
})

# -------------------------- 常量定义 --------------------------
# 基础英文停用词集合（当NLTK加载失败时使用）
BASIC_STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
    'yours', 'yourself', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
    'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
    'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
    'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
    'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
    'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
    'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
    'than', 'too', 'very', 'can', 'will', 'just', 'don', 'should', 'now'
}


# -------------------------- 核心函数 --------------------------
def initialize_nltk():
    """初始化NLTK停用词库
    功能逻辑：
    1. 多路径检测：用户目录 → 项目目录
    2. 自动下载：当本地资源缺失时自动下载
    3. 备用方案：下载失败时启用基础停用词
    """
    global STOPWORDS  # ← 新增全局变量
    STOPWORDS = BASIC_STOPWORDS  # ← 默认使用基础停用词

    try:
        # 多路径检测逻辑
        nltk.data.path = [
            os.path.expanduser("~/nltk_data"),  # 用户目录
            os.path.join(os.getcwd(), "nltk_data"),  # 项目目录
        ]

        valid = any(os.path.exists(os.path.join(p, "corpora/stopwords/english")) for p in nltk.data.path)
        if not valid:
            raise LookupError

        # 新增路径验证日志
        print(f" NLTK搜索路径：{nltk.data.path}")
        nltk.data.find('corpora/stopwords')
        if not os.path.exists(os.path.join(nltk.data.path[0], "corpora/stopwords/english")):
            raise LookupError  # ← 强制触发下载流程

        STOPWORDS = set(stopwords.words('english'))  # ← 成功时更新为正式停用词
        print(" NLTK停用词库加载成功")

    except LookupError:
        try:
            # 自动创建所有候选路径
            for path in nltk.data.path:
                os.makedirs(os.path.join(path, "corpora"), exist_ok=True)

            print("️ 正在下载停用词库...")
            nltk.download('stopwords', download_dir=nltk.data.path[0])  # 优先用户目录
            print(" 下载完成，路径：", os.path.join(nltk.data.path[0], "corpora/stopwords"))

            if os.path.exists(os.path.join(nltk.data.path[0], "corpora/stopwords/english")):
                print(" 停用词库验证通过")
            else:
                raise RuntimeError("文件下载后验证失败")

        except Exception as e:
            print(f"\n 警告：使用备用停用词（{len(BASIC_STOPWORDS)}个）")


def clean_text(text):
    """文本预处理流水线
    处理步骤：
    1. 清洗：移除非字母字符 → 转小写 → 去两端空格
    2. 分词：按空格切分文本
    3. 过滤：去除停用词
    4. 词干提取：Porter词干化处理
    Args:
        text (str): 原始文本
    Returns:
        str: 标准化文本
    """
    # 文本清洗
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower().strip()

    # 分词与停用词过滤
    stop_words = STOPWORDS  # ← 直接使用全局变量
    tokens = [PorterStemmer().stem(w) for w in text.split() if w not in stop_words]

    return ' '.join(tokens)


def analyze_features(vectorizer, model, n_top=20):
    """
    分析特征重要性
    实现原理：
    1. 获取特征名称（兼容新旧版sklearn）
    2. 按垃圾邮件类别的对数概率排序
    3. 输出TopN重要特征及其权重
    Args:
        vectorizer: 特征向量化器
        model: 训练好的模型
        n_top: 显示前N个重要特征
    """
    # 兼容性处理：新旧版本特征名获取  放弃兼容性，直接采用旧版
    features_names = vectorizer.get_feature_names()

    features = sorted(
        zip(features_names, model.feature_log_prob_[1]),
        key=lambda x: x[1],
        reverse=True
    )[:n_top]

    # 数据对齐处理
    vocab = [feat[0] for feat in features]
    weights = [feat[1] for feat in features]

   # return features
    return vocab, weights


def evaluate_model(model, x_test, y_test, thresholds=np.arange(0.5, 1, 0.02)):
    """
    多阈值模型评估（增强版）
    新增功能：
    1. 详细指标输出（包含各分类的精确率/召回率/F1）
    2. 基于加权F1的最佳阈值推荐
    3. 返回最佳阈值供后续使用
    """
    y_proba = model.predict_proba(x_test)[:, 1]

    best_score = 0

    reports = []

    # 评估自定义阈值
    for thresh in thresholds:
        y_pred = (y_proba > thresh).astype(int)
        report = classification_report(y_test, y_pred, output_dict=True)

        # 记录当前阈值报告
        reports.append((thresh, report))

        # 获取加权平均指标
        weighted_avg = report['weighted avg']
        current_f1 = weighted_avg['f1-score']

        # 更新最佳阈值
        if current_f1 > best_score:
            best_score = current_f1
            best_thresh = thresh

        # 详细指标输出
        print("\n" + "-" * 50)
        print(f" 阈值 {thresh:.2f} 详细指标 ".center(45))
        print("-" * 50)
        print(f"垃圾邮件识别:")
        print(
            f"  精确率: {report['1']['precision']:.3f} | 召回率: {report['1']['recall']:.3f} | F1: {report['1']['f1-score']:.3f}")
        print(f"正常邮件识别:")
        print(
            f"  精确率: {report['0']['precision']:.3f} | 召回率: {report['0']['recall']:.3f} | F1: {report['0']['f1-score']:.3f}")
        print(f"加权平均F1: {current_f1:.3f}")

    # 最佳阈值推荐
    print("\n" + "=" * 50)
    print(f" 最佳阈值推荐：{best_thresh:.2f} ".center(45))
    print(f" 对应加权F1分数：{best_score:.3f} ".center(45))
    print("=" * 50)

    # 在返回前添加数据收集
    f1_scores = [report['weighted avg']['f1-score'] for _, report in reports]
    return best_thresh, thresholds, f1_scores  # ← 修改返回值为三元组


def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    生成混淆矩阵可视化
    可视化要素：
    1. 蓝色系配色方案
    2. 数值标签显示
    3. 中文字符支持
    4. 自动保存为PNG格式
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure()
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Greens',
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={'size': 12}
    )
    plt.title('邮件分类混淆矩阵', fontweight='bold')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()


def plot_threshold_analysis(thresholds, f1_scores, best_thresh):
    """绘制阈值选择平滑曲线"""
    plt.figure()

    # 生成平滑曲线
    x_new = np.linspace(min(thresholds), max(thresholds), 300)
    spl = make_interp_spline(thresholds, f1_scores, k=3)  # 三次样条插值
    y_smooth = spl(x_new)

    plt.plot(x_new, y_smooth, 'b-', linewidth=2)
    plt.scatter(thresholds, f1_scores, c='red', s=50, zorder=5)
    plt.axvline(best_thresh, color='green', linestyle='--', linewidth=1.5)
    plt.ylim(
        max(0.5, min(f1_scores)*0.95),  # 强制最小显示0.5
        min(1.0, max(f1_scores)*1.05)  # 强制最大显示1.05倍
    )
    focus_range = 0.1  # ← 控制聚焦范围
    plt.xlim(
        max(0.3, best_thresh - focus_range),  # 确保不低于0.3
        min(0.9, best_thresh + focus_range)  # 确保不高于0.9
    )
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(0.05))  # X轴每0.05一个刻度
    plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.02))  # Y轴每0.02一个刻度

    plt.title('阈值选择分析', fontsize=14, pad=20)
    plt.xlabel('分类阈值', fontsize=12)
    plt.ylabel('加权F1分数', fontsize=12)
    plt.grid(True, alpha=0.3)

    # 移除箭头后的纯文本标注
    plt.text(best_thresh + 0.02, max(f1_scores) - 0.03,
             f'最佳阈值: {best_thresh:.2f}',
             fontsize=10, color='green')

    plt.tight_layout()
    plt.savefig('threshold_analysis.png')
    plt.show()


# -------------------------- 主程序 --------------------------
if __name__ == "__main__":
    try:
        print("\n" + "=" * 50)
        print(" 邮件分类系统 v2.0 ".center(45))
        print("=" * 50)



        # 初始化阶段
        initialize_nltk()


        # 数据加载与编码
        print("\n" + "=" * 50)
        print(" 数据加载与预处理 ".center(45))
        print("=" * 50)
        df = pd.read_csv("Email Classification.csv", encoding="utf-8-sig")
        df["Class"] = LabelEncoder().fit_transform(df["Class"])  # 标签编码
        print("数据集类别分布:\n", df["Class"].value_counts())

        # 文本预处理
        print("\n 文本预处理中...")
        df["Cleaned_Message"] = df["Message"].apply(clean_text)

        # 特征工程
        print("\n 特征工程处理中...")
        tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        min_df = 5,
        max_df = 0.85
        X = tfidf.fit_transform(df["Cleaned_Message"])
        y = df["Class"]

        # 特征工程后添加验证
        print("\n特征空间分析:")
        print(f"实际特征维度: {X.shape[1]}")
        print(f"词汇表覆盖率: {len(tfidf.get_feature_names()) / 25000:.1%}")

        # 添加二元语法有效性验证
        bigram_samples = [phrase for phrase in tfidf.get_feature_names() if ' ' in phrase][:10]
        print("示例二元语法特征:", bigram_samples)



        # 直接使用旧版函数
        tfidf_feature_names = tfidf.get_feature_names()  # ← 移除版本判断

        # 数据划分
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.3,
            random_state=42,
            stratify=y  # 保持类别分布
        )

        # 数据分布统计
        def print_distribution(name, y_data):
            ham = (y_data == 0).sum()
            spam = (y_data == 1).sum()
            print(f"{name}:")
            print(f"正常邮件(ham): {ham} ({ham/len(y_data):.1%})")
            print(f"垃圾邮件(spam): {spam} ({spam/len(y_data):.1%})")

        print_distribution("\n训练集分布", y_train)
        print_distribution("\n测试集分布", y_test)

        # 模型训练
        print("\n 模型训练中...")
        # 新增网格搜索配置
        param_grid = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10]}
        grid_search = GridSearchCV(
            estimator=MultinomialNB(),
            param_grid=param_grid,
            cv=5,
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_

        # 输出最佳参数
        print(f"✓ 模型训练完成，最佳alpha值: {grid_search.best_params_['alpha']}")

        # 垃圾邮件词频统计功能
        print("\n 正在统计垃圾邮件高频词汇...")
        spam_indices = y_train[y_train == 1].index
        spam_texts = df.loc[spam_indices, 'Cleaned_Message']

        # 移除停用词
        spam_texts = spam_texts.apply(
            lambda x: ' '.join([w for w in x.split() if w not in STOPWORDS])
        )

        # 统一使用特征工程中的词汇表
        count_vec = CountVectorizer(vocabulary=tfidf_feature_names)
        spam_counts = count_vec.fit_transform(spam_texts)
        spam_counts_array = spam_counts.sum(axis=0).A1  # 提前计算并复用

        # 提取训练集垃圾邮件文本
        spam_indices = y_train[y_train == 1].index
        spam_texts = df.loc[spam_indices, 'Cleaned_Message']

        # 兼容性处理：新旧版本特征名获取  放弃兼容性，直接使用旧版
        feature_names = (count_vec.get_feature_names())  # ← 放弃兼容处理，直接使用旧版

        word_counts = zip(feature_names, spam_counts.sum(axis=0).tolist()[0])  # ← 修改这里
        sorted_words = sorted(word_counts, key=lambda x: x[1], reverse=True)[:30]

        # 计算频率（基于训练集总数）
        train_total = X_train.shape[0]  # 训练集样本总数
        freq_data = [
            (word, count, count / train_total)
            for word, count in sorted_words
        ]

        # 模型评估
        evaluate_model(model, X_test, y_test)
        #best_threshold = evaluate_model(model, X_test, y_test)
        best_threshold, thresholds, f1_scores = evaluate_model(model, X_test, y_test)
        feature_vocab, feature_weights = analyze_features(tfidf, model, n_top=20)  # ← 修改为输出20个特征
        #top_features = analyze_features(tfidf, model)

        # 使用最佳阈值进行最终预测
        y_proba = model.predict_proba(X_test)[:, 1]
        final_pred = (y_proba > best_threshold).astype(int)

        # 生成最终评估报告
        print("\n" + "="*50)
        print(f" 最终评估报告（使用阈值 {best_threshold:.2f}） ".center(45))
        print("="*50)
        print(classification_report(y_test, final_pred))

        # 添加测试准确率输出
        print(f"\n测试集准确率：{np.mean(final_pred == y_test):.4f}")

        # 使用最佳阈值生成混淆矩阵
        plot_confusion_matrix(y_test, final_pred, ['ham', 'spam'])

        # 使用相同词汇表的CountVectorizer
        count_vec = CountVectorizer(vocabulary=feature_vocab)
        spam_counts = count_vec.fit_transform(spam_texts)
        spam_counts_array = np.asarray(spam_counts.sum(axis=0)).flatten()  # 确保转换为numpy数组

        # 调试用维度验证（添加在创建DataFrame之前）
        print(f"特征维度: {len(feature_vocab)}, 词频维度: {spam_counts.sum(axis=0).shape[1]}")
        print(f"词汇列长度: {len(feature_vocab)}")
        print(f"权重列长度: {len(feature_weights)}")
        print(f"词频统计维度: {spam_counts.sum(axis=0).shape}")

        # 创建数据框（确保数组长度一致）
        df_merge = pd.DataFrame({
            '词汇': feature_vocab,
            '权重': feature_weights,
            '出现次数': spam_counts_array[:len(feature_vocab)],  # 使用预计算结果
            '频率': spam_counts_array[:len(feature_vocab)] / X_train.shape[0]
        }).head(20)

        # 控制台输出预览
        print("\n top20_high_frequency_spam_keywords:")
        print(f"{'词汇':<15} | {'权重':<8} | {'出现次数':<8} | {'频率':<8}")
        print("-" * 45)
        for idx, row in df_merge.iterrows():
            print(f"{row['词汇']:15} | {row['权重']:8.3f} | {row['出现次数']:8} | {row['频率']:8.3%}")

        # 导出CSV
        df_merge.to_csv('top20_high_frequency_spam_keywords.csv', index=False, encoding='utf-8-sig')
        print(" 数据已导出至 top20_high_frequency_spam_keywords.csv")  # 更新提示信息


        # 图像输出
        print("\n 生成可视化结果...")
        #(y_test, model.predict(X_test), ['ham', 'spam'])
        plot_threshold_analysis(thresholds, f1_scores, best_threshold)
        print("\n 程序运行结束")

    except Exception as e:
        print(f"\n 程序异常终止: {str(e)}")
        sys.exit(1)