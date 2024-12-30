import pickle
import pandas as pd
import os
import re
import jieba

class DataProcessing:
    """
    该类用于统一管理数据的读取、清洗、文本分词、标签编码及保存等处理过程。
    主要处理流程如下：
    1. 读取数据文件（支持 csv/tsv/xlsx/json 格式）
    2. 提取特征（文本）和标签
    3. 对文本进行结构化清洗，替换标点符号
    4. 使用 jieba 进行分词
    5. 对标签进行数字编码
    6. 将分词后的文本和标签编码结果保存为 CSV 文件
    """

    def __init__(self, data_path, label_vocab_path=None, data_outlook=True):
        """
        初始化数据处理类，设置数据路径和标签词汇表的保存路径

        参数:
        -------
        data_path: str
            数据文件的路径，支持 csv/tsv/xlsx/json 等格式
        label_vocab_path: str, optional
            词汇表文件路径，如果传入则使用该路径加载词汇表
        data_outlook: bool, default True
            是否在读取完成后，随机打印 3 条示例数据
        """
        self.data_path = data_path
        self.data_outlook = data_outlook
        # 如果未提供词汇表路径，则根据数据路径自动生成
        self.label_vocab_path = label_vocab_path if label_vocab_path is not None else self.generate_label_vocab_path()

        # 初始化属性
        self.Text_data = None
        self.features = None
        self.labels = None

    def read_data(self):
        """
        读取数据文件，并将其存储在 self.Text_data 中。
        支持 csv、tsv、xlsx 和 json 格式。
        如果 data_outlook = True，则打印随机 3 行数据作为预览。

        返回:
        -------
        pandas.DataFrame
            读取的数据
        """
        extension = os.path.splitext(self.data_path)[1].lower()

        # 根据文件类型读取数据
        if extension in ['.csv', '.tsv']:
            self.Text_data = pd.read_csv(self.data_path, sep="\t" if extension == '.tsv' else ",", header=None, encoding='utf-8')
        elif extension == '.xlsx':
            self.Text_data = pd.read_excel(self.data_path, header=None)
        elif extension == '.json':
            self.Text_data = pd.read_json(self.data_path, encoding='utf-8')
        else:
            raise ValueError(f"不支持的文件类型: {extension}")

        # 打印随机的 3 行数据示例（如果 data_outlook 为 True）
        if self.data_outlook:
            print("随机输出 3 行数据示例(原始数据）：")
            print(self.Text_data.sample(3))

        return self.Text_data

    def extract_features_and_labels(self):
        """
        提取数据中的标签和文本特征。
        假设数据中的第一列为标签，第二列为文本内容。
        如果文件格式不符合要求，函数会尝试通过逗号分割第一列数据。

        返回:
        -------
        tuple: (features, labels)
            - features: 文本特征（列表形式）
            - labels: 标签（列表形式）
        """
        if not isinstance(self.Text_data, pd.DataFrame):
            raise TypeError("Text_data 必须是一个 pandas DataFrame 对象")

        # 删除 NA 值
        self.Text_data = self.Text_data.dropna(axis=0, how='any')

        # 如果数据没有两列，则尝试通过逗号分割并处理
        if self.Text_data.shape[1] < 2:
            print("DataFrame 的列数不足两列，将自动分割数据。")

            # 使用逗号分割数据
            self.Text_data[['label', 'text']] = self.Text_data[0].str.split(',', n=1, expand=True)
            self.Text_data = self.Text_data.drop(columns=[0])  # 删除原始列
            self.Text_data = self.Text_data.drop(index=[0]).reset_index(drop=True)  # 删除首行并重置索引

            # 再次检查 DataFrame 是否有两列
            if self.Text_data.shape[1] != 2:
                raise ValueError("数据分割后，DataFrame 的列数仍然不足两列。")

        # 提取标签和文本特征
        self.labels = self.Text_data.iloc[:, 0].tolist()
        self.features = self.Text_data.iloc[:, 1].tolist()

        print(f"提取了 {len(self.features)} 条文本特征和 {len(self.labels)} 个标签。")
        if self.data_outlook:
            print("部分特征(特征分割)：", self.features[:3])
            print("部分标签（特征分割）：", self.labels[:3])

        return self.features, self.labels

    def feature_structural_transformation(self):
        """
        对文本进行结构化处理，主要是标点符号的替换。
        1) 替换文本最开始出现的 `。` 为 [CLS]
        2) 其他位置的 `。` 替换为 [SEP]
        3) 连续的英文句号替换为 [ELLIPSIS]

        返回:
        -------
        pandas.DataFrame
            经过处理后的数据
        """
        self.extract_features_and_labels()

        if not isinstance(self.features, list):
            raise TypeError("features 必须是一个 list 对象")

        def process_text(text):
            """对文本进行正则化处理"""
            if pd.isnull(text):
                return text

            # 1) 如果文本最开始出现一个或多个 `。`，全部替换为 [CLS]
            leading = re.match(r'^[。]+', text)
            if leading:
                length_leading = len(leading.group(0))
                text = "[CLS]" + text[length_leading:]

            # 2) 文本其他位置出现的一个或多个 `。`（可能连写）统一替换为 [SEP]
            text = re.sub(r'[。]+', '[SEP]', text)

            # 3) 连续3个及以上的英文句号（"..."，"...."等）替换为 [ELLIPSIS]
            text = re.sub(r'\.{2,}', '[ELLIPSIS]', text)

            return text

        # 应用结构化清洗
        self.features = [process_text(feature) for feature in self.features]

        if self.data_outlook:
            print("部分特征(特征清洗)：", self.features[:3])
            print("部分标签（特征清洗）：", self.labels[:3])

        return self.Text_data

    def generate_label_vocab_path(self):
        """
        根据数据文件名生成词汇表保存路径

        返回:
        -------
        str
            词汇表保存的路径
        """
        file_name = os.path.basename(self.data_path)
        file_name_without_extension = os.path.splitext(file_name)[0]
        return os.path.join(os.path.dirname(self.data_path), f"{file_name_without_extension}_label_vocab.pkl")

    def labels_encoding(self):
        """
        将字符串类型的标签转换为数字编码。
        如果词汇表已经存在，则加载；否则会新建一个词汇表。

        返回:
        -------
        list
            数字编码后的标签列表
        """
        if os.path.exists(self.label_vocab_path):
            with open(self.label_vocab_path, 'rb') as f:
                label_vocab = pickle.load(f)
            print("加载现有的 label 词汇表。")
        else:
            label_vocab = {}
            print("词汇表不存在，正在创建新的词汇表。")
            for label in self.labels:
                if label not in label_vocab:
                    label_vocab[label] = len(label_vocab)
            with open(self.label_vocab_path, 'wb') as f:
                pickle.dump(label_vocab, f)

        label_encoding = [label_vocab[label] for label in self.labels]
        return label_encoding

    def features_tokenlize(self):
        """
        使用 Jieba 进行词级分词，并显示分词后的词列表。

        返回:
        -------
        list
            分词后的文本列表
        """
        if not isinstance(self.features, list):
            raise TypeError("features 必须是一个 list 对象")

        # 使用 Jieba 进行词级分词
        tokenized_features = [self.jieba_tokenize(feature) for feature in self.features]

        if self.data_outlook:
            print("部分分词后的文本（词级）：", tokenized_features[:1])

        return tokenized_features

    def jieba_tokenize(self, text):
        """
        使用 Jieba 对单个文本进行词级分词

        参数:
        -------
        text: str
            需要分词的文本

        返回:
        -------
        list
            分词后的单词列表
        """
        # 使用精确模式进行分词
        words = list(jieba.cut(text, cut_all=False))
        return words

    def save_tokenized_and_encoded_labels(self):
        """
        保存分词后的文本和标签编码到 CSV 文件。
        将每条文本的分词结果与对应的标签编码保存在同一个文件中。

        返回:
        -------
        str
            保存文件的路径
        """
        tokenized_features = self.features_tokenlize()
        encoded_labels = self.labels_encoding()

        # 创建输出文件名
        output_file = os.path.splitext(self.data_path)[0] + '_tokenized_and_encoded.csv'

        # 构建 DataFrame 并保存为 CSV 文件
        df = pd.DataFrame({
            'tokenized_text': [' '.join(feature) for feature in tokenized_features],
            'encoded_label': encoded_labels
        })
        df.to_csv(output_file, index=False)

        print(f"分词后的文本和标签编码已保存至: {output_file}")

        return output_file



# 使用代码
if __name__ == "__main__":
    data_path = "E:/TAR_LMMS/semester_analysis/data_SA/Nlpcc2014/WholeSentence_Nlpcc2014Train.tsv"

    # 初始化数据处理类
    data_processor = DataProcessing(data_path, label_vocab_path=None, data_outlook=True)

    # 读取数据
    data_processor.read_data()

    # 对文本进行结构化处理
    data_processor.feature_structural_transformation()

    # 保存分词后的文本和标签编码
    output_file = data_processor.save_tokenized_and_encoded_labels()

    # 输出保存路径
    print(f"结果已保存为文件：{output_file}")


