from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import jieba


# 特征提取
# 提取字典特征
def dict_demo():
    """
    字典特征提取
    :return: None
    """

    # 1 获取数据
    data = [{'city': '北京', 'temperature': 100},
            {'city': '上海', 'temperature': 60},
            {'city': '深圳', 'temperature': 30}]

    # 2 字典特征值提取
    # 2.1 实例化
    transfer = DictVectorizer(sparse=False)

    # 2.2 转换
    new_data = transfer.fit_transform(data)
    print(new_data)

    # 2.3 获取具体属性名字
    name = transfer.get_feature_names_out()
    print(f'名字是:{name}')


# 提取英文特征
def English_count_demo():
    """
    文本特征提取 英文
    :return:
    """
    # 1 获取数据
    data = ["life is short,i like python",
            "life is too long,i dislike python"]

    # 2 文本特征提取
    transfer = CountVectorizer(stop_words=['like'])  # 注意没有spare这个参数
    new_data = transfer.fit_transform(data)
    print(new_data)
    print(new_data.toarray())
    name = transfer.get_feature_names_out()
    print(f'名字是:{name}')


def cut_word(text):
    """
    中文分词
    :return:
    """
    return " ".join(list(jieba.cut(text)))


# 提取中文特征
def Chinese_count_demo():
    """
    中文特征提取
    :return:
    """
    # 获取数据
    data = ["⼀种还是⼀种今天很残酷，明天更残酷，后天很美好，但绝对⼤部分是死在明天晚上，所以每个⼈不要放弃今天。",
            "我们看到的从很远星系来的光是在⼏百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。",
            "如果只⽤⼀种⽅式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。"]

    # 文章分割
    temp_list = []
    for temp in data:
        temp_list.append(cut_word(temp))

    # 实例化 + 转换
    transfer = CountVectorizer()  # 注意没有spare这个参数
    new_data = transfer.fit_transform(temp_list)
    print(new_data)
    print(new_data.toarray())
    name = transfer.get_feature_names_out()
    print(f'名字是:{name}')


def tfidf_demo():
    """
    中文特征提取
    :return:
    """
    # 获取数据
    data = ["⼀种还是⼀种今天很残酷，明天更残酷，后天很美好，但绝对⼤部分是死在明天晚上，所以每个⼈不要放弃今天。",
            "我们看到的从很远星系来的光是在⼏百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。",
            "如果只⽤⼀种⽅式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。"]

    # 文章分割
    temp_list = []
    for temp in data:
        temp_list.append(cut_word(temp))

    # 实例化 + 转换
    transfer = TfidfVectorizer()
    # transfer = CountVectorizer()  # 注意没有spare这个参数
    new_data = transfer.fit_transform(temp_list)
    print(new_data)
    print(new_data.toarray())
    name = transfer.get_feature_names_out()
    print(f'名字是:{name}')


if __name__ == '__main__':
    dict_demo()
    English_count_demo()
    Chinese_count_demo()
    tfidf_demo()
