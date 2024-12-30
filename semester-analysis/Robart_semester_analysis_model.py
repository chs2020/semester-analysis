import os
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm  # 导入 tqdm 进度条库
import data_processing  # 确保 data_processing.py 存在，并包含 DataProcessing 类

class semester_analysis_model:
    """
    该类用于读取和处理训练数据，包括检查已处理文件的存在情况，
    并根据需要重新生成分词和标签编码后的文件。随后，它使用预训练的中文模型对分词后的文本进行词级别的嵌入（embedding），
    并将嵌入向量保存为 CSV 文件，供后续模型训练调用。
    """

    def __init__(self, data_path: str, tokenizer_path: str, model_path: str):
        """
        初始化模型类，设置数据路径，加载分词器和预训练模型。

        参数:
        -------
        data_path : str
            原始数据文件路径 (csv/tsv/xlsx/json 等格式)
        tokenizer_path : str
            分词器的加载路径（如 "hfl/chinese-roberta-wwm-ext" 或本地目录）
        model_path : str
            预训练模型的加载路径（如 "hfl/chinese-roberta-wwm-ext" 或本地目录）
        """
        self.data_path = data_path

        # 加载分词器
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            print(f"成功加载分词器: {tokenizer_path}")
        except Exception as e:
            raise ValueError(f"无法加载分词器: {tokenizer_path}. 错误: {e}")

        # 加载预训练模型
        try:
            self.model = AutoModel.from_pretrained(model_path)
            self.model.eval()  # 设置模型为评估模式
            print(f"成功加载预训练模型: {model_path}")
        except Exception as e:
            raise ValueError(f"无法加载预训练模型: {model_path}. 错误: {e}")

        # 检查是否有可用的 GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"模型已移动到设备: {self.device}")

        # 初始化存储标签和特征的属性
        self.labels = None  # 存储标签编码 (list[int])
        self.features = None  # 存储分词后文本 (list[str])

    def read_data(self):
        """
        读取数据文件。如果已存在处理后的文件（_tokenized_and_encoded.csv），
        则直接加载处理后的数据；否则，调用数据处理模块对原始数据进行处理。

        返回:
        -------
        tuple: (features, labels)
            - features: 分词后的文本特征（列表形式）
            - labels: 数字编码后的标签（列表形式）
        """
        # 生成处理后的文件名
        processed_file_path = os.path.splitext(self.data_path)[0] + '_tokenized_and_encoded.csv'

        if os.path.exists(processed_file_path):
            # 如果处理后的文件存在，则直接加载数据
            try:
                processed_data = pd.read_csv(processed_file_path)
                self.labels = processed_data['encoded_label'].tolist()
                self.features = processed_data['tokenized_text'].tolist()
                print(f"加载已处理好的数据文件: {processed_file_path}")
            except Exception as e:
                raise IOError(f"无法加载文件: {processed_file_path}. 错误: {e}")
        else:
            # 如果处理后的文件不存在，则调用数据处理模块进行处理
            try:
                data_processor = data_processing.DataProcessing(self.data_path)

                # 1) 读取原始数据
                data_processor.read_data()
                # 2) 对数据进行结构化处理
                data_processor.feature_structural_transformation()
                # 3) 保存分词后的文本和标签编码
                output_file = data_processor.save_tokenized_and_encoded_labels()
                print(f"已生成并保存分词/编码文件: {output_file}")

                # 4) 再次读取处理后的文件
                processed_data = pd.read_csv(output_file)
                self.labels = processed_data['encoded_label'].tolist()
                self.features = processed_data['tokenized_text'].tolist()
            except ImportError:
                raise ImportError("请确保 data_processing.py 存在，并包含 DataProcessing 类。")
            except Exception as e:
                raise RuntimeError(f"数据处理时发生错误: {e}")

        return self.features, self.labels

    def features_embedding(self, batch_size=32):
        """
        对 self.features (分词后的文本) 进行词级别的嵌入，存储到 CSV 文件中。
        1) 检查 _feature_embeddings.csv 文件是否已存在:
           - 若存在则直接读入，不再重新生成。
        2) 若不存在，则对每条分词文本调用模型生成 last_hidden_state，
           并将词及对应的 embedding 逐一保存到 CSV 文件。

        参数:
        -------
        batch_size : int, default 32
            每批处理的文本数量。

        返回:
        -------
        embedding_df : pd.DataFrame
            包含词级嵌入结果的 DataFrame。
        """
        processed_file_path_feature_embedding = os.path.splitext(self.data_path)[0] + '_feature_embeddings.csv'

        # 如果嵌入文件已存在，则直接读取
        if os.path.exists(processed_file_path_feature_embedding):
            print(f"词级嵌入文件已存在: {processed_file_path_feature_embedding}")
            try:
                embedding_df = pd.read_csv(processed_file_path_feature_embedding)
                print("成功加载已存在的词级嵌入文件。")
                return embedding_df
            except Exception as e:
                raise IOError(f"无法加载词级嵌入文件: {processed_file_path_feature_embedding}. 错误: {e}")

        # 若不存在，则需要计算词级 embedding
        if not self.features:
            raise ValueError("未检测到分词后的文本。请先调用 read_data() 获取 self.features。")

        all_word_embeddings = []
        total = len(self.features)

        # 使用 tqdm 进度条包装循环
        for batch_start in tqdm(range(0, total, batch_size), desc="生成词级嵌入", unit="批次"):
            batch_end = min(batch_start + batch_size, total)
            batch_features = self.features[batch_start:batch_end]

            # 准备批量输入
            split_tokens = [text.split() for text in batch_features]  # List[List[str]]

            # 调用 tokenizer
            try:
                inputs = self.tokenizer(
                    split_tokens,
                    is_split_into_words=True,   # 指示输入已是分词好的子词
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512  # 显式指定最大长度以避免警告
                )
                # 将输入移动到指定设备
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            except Exception as e:
                print(f"批次 {batch_start}-{batch_end} 的分词器调用出错: {e}")
                continue  # 跳过当前批次

            # 调用模型
            try:
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    last_hidden_state = outputs.last_hidden_state  # [batch_size, seq_len, hidden_dim]
            except Exception as e:
                print(f"批次 {batch_start}-{batch_end} 的模型调用出错: {e}")
                continue  # 跳过当前批次

            # Convert to CPU and numpy
            last_hidden_state = last_hidden_state.cpu().numpy()  # [batch_size, seq_len, hidden_dim]

            # Convert input_ids to tokens for each sequence in the batch
            input_ids = inputs["input_ids"].cpu().numpy().tolist()  # List[List[int]]
            token_sequences = [self.tokenizer.convert_ids_to_tokens(seq_ids) for seq_ids in input_ids]  # List[List[str]]

            # Iterate over each sequence in the batch
            for idx_in_batch, (tokens, embeddings) in enumerate(zip(token_sequences, last_hidden_state)):
                # Original text index
                row_idx = batch_start + idx_in_batch

                for tok, emb_vec in zip(tokens, embeddings):
                    # Filter out special tokens [CLS], [SEP], [PAD]
                    if tok in ["[CLS]", "[SEP]", "[PAD]"]:
                        continue  # 跳过特殊标记

                    # Ensure emb_vec is a 1D array
                    if isinstance(emb_vec, (np.ndarray, list)) and len(emb_vec.shape) == 1:
                        # Build row
                        row = {
                            "row_idx": row_idx,
                            "word": tok
                        }
                        # Assign embedding dimensions
                        for dim_i, val in enumerate(emb_vec):
                            row[f"dim_{dim_i + 1}"] = val
                        all_word_embeddings.append(row)
                    else:
                        print(f"批次 {batch_start}-{batch_end} 的行 {row_idx} 中词 '{tok}' 的嵌入向量形状异常: {emb_vec.shape}")
                        continue  # 跳过如果嵌入向量形状不符合预期

        # 构建 DataFrame 并写出 CSV
        try:
            embedding_df = pd.DataFrame(all_word_embeddings)
            embedding_df.to_csv(processed_file_path_feature_embedding, index=False, float_format="%.6f")
            print(f"词级嵌入向量已保存至: {processed_file_path_feature_embedding}")
        except Exception as e:
            raise IOError(f"无法保存词级嵌入文件: {processed_file_path_feature_embedding}. 错误: {e}")

        return embedding_df

# ================== 使用示例 ==================
if __name__ == "__main__":
    # 替换为实际数据文件路径
    data_path = "E:/TAR_LMMS/semester_analysis/data_SA/Nlpcc2014/WholeSentence_Nlpcc2014Train.tsv"

    # 指定分词器和模型的路径或名称 (使用 Hugging Face 上的中文 RoBERTa 模型)
    tokenizer_path = "hfl/chinese-roberta-wwm-ext"  # 可以替换为其他中文模型
    model_path = "hfl/chinese-roberta-wwm-ext"      # 可以替换为其他中文模型

    # 初始化模型类
    model = semester_analysis_model(data_path, tokenizer_path, model_path)

    # 1) 读取并处理数据
    try:
        features, labels = model.read_data()
        print("数据加载和处理完成！")
        print("部分特征示例：", features[:3])
        print("部分标签示例：", labels[:3])
    except Exception as e:
        print(f"处理数据时发生错误: {e}")

    # 2) 生成词级别的嵌入文件
    try:
        embedding_df = model.features_embedding(batch_size=32)  # 你可以调整 batch_size
        print("词级别的嵌入生成或加载完成。")
        print("部分嵌入示例：")
        print(embedding_df.head())
    except Exception as e:
        print(f"生成嵌入向量时出错: {e}")
