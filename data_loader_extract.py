import json
import torch
import pickle
import dgl
import os
import re
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from dgl.dataloading import GraphDataLoader
from build_base_graph_extract import build_graph
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def save_graph_pairs(graph_pairs, file_path):
    graphs = [graph for pair in graph_pairs for graph in pair]
    dgl.save_graphs(file_path, graphs)
    logging.info("Saved pair graph at %s", file_path)


def load_graph_pairs(file_path, num_pairs):
    graphs, _ = dgl.load_graphs(file_path)
    graph_pairs = [(graphs[i * 2], graphs[i * 2 + 1]) for i in range(num_pairs)]
    logging.info("Loaded from %s", file_path)
    return graph_pairs


# def extract_node_features(embeddings_data, idx, prefix):
#     node_features = {}
#     for item in embeddings_data[idx][prefix]:
#         node_id, embedding = item
#         node_features[node_id] = embedding
#     return node_features
def extract_node_features(embeddings_data, idx, prefix):
    node_features = []
    for item in embeddings_data[idx][prefix]:
        node_id, embedding, text = item
        node_features.append((node_id, embedding, text))
    return node_features


def load_all_embeddings(directory_path):
    embeddings_list = []

    # 获取文件名列表
    filenames = os.listdir(directory_path)

    # 定义一个函数来提取文件名中的数字
    def extract_number(filename):
        # 使用正则表达式提取文件名中的数字部分
        match = re.search(r"\d+", filename)
        return int(match.group()) if match else float("inf")

    # 根据文件名中的数字进行排序
    sorted_filenames = sorted(filenames, key=extract_number)

    for filename in sorted_filenames:
        if filename.endswith(".npz"):
            file_path = os.path.join(directory_path, filename)
            embeddings = torch.load(file_path)
            embeddings_list.extend(embeddings)
            logging.info("Loaded embeddings from %s", file_path)

    return embeddings_list


def load_embeddings_from_directory(directory_path, limit=None):
    embeddings_list = []
    count = 0
    for filename in os.listdir(directory_path):
        if filename.endswith(".npz"):
            file_path = os.path.join(directory_path, filename)
            embeddings = torch.load(file_path)
            embeddings_list.extend(embeddings)
            count += 1
            if limit and count >= limit:
                break
    return embeddings_list


class RSTDataset(Dataset):
    def __init__(
        self,
        rst_path,
        nli_data_path,  # 假设数据
        premise_emb_path,  # 假设的embedding,
        lexical_matrix_path,
        hypothesis_emb_path,
        batch_file_size=1,  # 每批处理的文件数量
        save_dir="./graph_pairs",  # 保存 graph_pairs 的目录
    ):
        self.rst_path = rst_path
        self.nli_data_path = nli_data_path
        self.premise_emb_path = premise_emb_path
        self.lexical_matrix_path = lexical_matrix_path
        self.hypothesis_emb_path = hypothesis_emb_path
        self.batch_file_size = batch_file_size
        self.save_dir = save_dir

        # 确保保存目录存在
        os.makedirs(self.save_dir, exist_ok=True)
        # 存储提前构建好的图，避免在训练时再构建
        self.graph_pairs = []

    def load_edu_labels(self, nli_data):
        # edu_lables 长度是2/3nli数据量
        edu_labels = []
        for item in nli_data:
            if "evidence_edu" not in item:
                continue
            else:
                edu_labels.append(item["evidence_edu"])
        return edu_labels  # 中立不计算

    def _create_label_tensor(self, g, positive_nodes):
        """为给定的图和正例节点创建标签张量"""
        num_nodes = g.number_of_nodes()
        labels = torch.zeros(num_nodes, dtype=torch.long)
        positive_nodes = [int(node) for node in positive_nodes]
        if positive_nodes:  # 如果有正例节点
            # 将 positive_nodes 转换为 tensor 以便进行比较
            positive_nodes_tensor = torch.tensor(positive_nodes)

            # 检查是否有节点超出范围
            if torch.any(positive_nodes_tensor >= num_nodes):
                invalid_nodes = positive_nodes_tensor[
                    positive_nodes_tensor >= num_nodes
                ]
                valid_nodes = positive_nodes_tensor[
                    positive_nodes_tensor < num_nodes  # 保留有效节点
                ]
                raise ValueError(
                    f"发现无效的节点索引: {invalid_nodes.tolist()}. "
                    f"节点索引应该在 0 到 {num_nodes-1} 之间。"
                )
                labels[valid_nodes] = 1
                return labels
            # 如果所有节点都有效，则设置标签
            labels[positive_nodes_tensor] = 1

        return labels

    def load_rst_data(self):
        """
        加载 RST 数据，只需要加载一次。
        """
        rst_results = []
        with open(self.rst_path, "r") as file:
            for line in file:
                rst_dict = json.loads(line.strip())
                rst_results.append(rst_dict)
        return rst_results

    def load_nli_data(self):
        """
        加载模型输出，只需要加载一次。
        """
        with open(self.nli_data_path, "r", encoding="utf-8") as file:
            nli_data = json.load(file)
        return nli_data

    def load_batch_files(self, batch_num):
        """
        根据批次号加载相应的词汇链和嵌入文件，并构建图。
        如果已存在保存的 graph_pairs 文件，则直接加载。
        """
        # 保存的 graph_pairs 文件名
        save_file = os.path.join(self.save_dir, f"graph_pairs_batch_{batch_num}.pkl")

        # 如果文件已存在，直接加载
        if os.path.isfile(save_file):
            with open(save_file, "rb") as f:
                self.graph_pairs = pickle.load(f)
            logging.info("Loaded graph pairs from %s", save_file)
            return

        # 否则加载文件，构建图并保存
        print("embedding dir exists")
        # 获取所有文件的路径并排序
        self.data = self.load_rst_data()
        self.nli_data = self.load_nli_data()
        self.hyp_emb = torch.load(self.hypothesis_emb_path)  # [(1,2,3),(),...]
        self.edu_labels = self.load_edu_labels(self.nli_data)  # [[[],[]],] or []

        batch_lexical_chains = []
        batch_embeddings = []

        # 加载词汇链
        with open(self.lexical_matrix_path, "rb") as f:
            batch_lexical_chains.extend(pickle.load(f))

        # 加载嵌入
        batch_embeddings.extend(torch.load(self.premise_emb_path))

        # 根据词汇链和嵌入构建图
        self.graph_pairs = self.build_graphs(
            batch_lexical_chains, batch_embeddings, self.hyp_emb, self.edu_labels
        )

        # 保存构建的 graph_pairs
        with open(save_file, "wb") as f:
            pickle.dump(self.graph_pairs, f)

    def build_graphs(self, lexical_chains, embeddings, hyp_emb, edu_labels):
        """
        根据加载的词汇链和嵌入构建图。
        """
        graph_pairs = []
        assert len(hyp_emb) * 3 == len(self.data)
        for idx in range(len(hyp_emb)):
            count = idx * 3
            rst_result = self.data[idx * 3]
            hyp_emb_three = hyp_emb[idx]

            # 构建图
            node_features_premise = extract_node_features(embeddings, count, "premise")
            rst_relations_premise = rst_result["rst_relation_premise"]
            node_types_premise = rst_result["pre_node_type"]
            g_premise = build_graph(
                node_features_premise, node_types_premise, rst_relations_premise
            )

            node_features_hypothesis = extract_node_features(
                embeddings, count, "hypothesis"
            )
            rst_relations_hypothesis = rst_result["rst_relation_hypothesis"]
            node_types_hypothesis = rst_result["hyp_node_type"]
            g_premise2 = build_graph(
                node_features_hypothesis,
                node_types_hypothesis,
                rst_relations_hypothesis,
            )
            if edu_labels == []:
                if idx == 0:
                    print("empty edu_labels")  # 单语言的没有
                edu_labels_three = {"entailment": [], "contradiction": []}
            else:
                edu_label_entailment = edu_labels[idx * 2]
                edu_label_contradiction = edu_labels[idx * 2 + 1]
                edu_labels_three = {
                    "entailment": [
                        self._create_label_tensor(g_premise, edu_label_entailment[0]),
                        self._create_label_tensor(g_premise2, edu_label_entailment[1]),
                    ],
                    "contradiction": [
                        self._create_label_tensor(
                            g_premise, edu_label_contradiction[0]
                        ),
                        self._create_label_tensor(
                            g_premise2, edu_label_contradiction[1]
                        ),
                    ],
                }

            graph_pairs.append(
                (
                    g_premise,
                    g_premise2,
                    lexical_chains[count],
                    hyp_emb_three,
                    edu_labels_three,
                )
            )

        return graph_pairs

    def __len__(self):
        return len(self.graph_pairs)

    def __getitem__(self, idx):
        (g_premise, g_premise2, lexical_chain, hyp_emb, edu_label) = self.graph_pairs[
            idx
        ]

        return (
            g_premise,
            g_premise2,
            lexical_chain,
            hyp_emb,
            edu_label,
        )


if __name__ == "__main__":
    base_data_dir = "/mnt/nlp/yuanmengying/ymy/data"
    dir = "2cd_nli_Spanish"
    language = "Spanish"
    type_list = ["train", "dev", "test"]
    batch_file_size = 1
    for type in type_list:
        print("*" * 30, type, "*" * 30)
        train_rst_path = f"{base_data_dir}/{dir}/{type}/{type}1/new_rst_result.jsonl"
        train_nli_data_path = (
            f"{base_data_dir}/nli_type_data/{language}/{type}.json"  # train_re_hyp.json
        )

        train_pre_emb_path = f"{base_data_dir}/{dir}/{type}/pre/node_embeddings.npz"

        train_hyp_emb_path = f"{base_data_dir}/{dir}/{type}/hyp/hyp_node_embeddings.npz"
        train_lexical_path = (
            f"{base_data_dir}/{dir}/graph_infos/{type}/lexical_matrixes.npz"
        )
        train_pair_graph = f"{base_data_dir}/{dir}/graph_pairs/{type}"
        train_dataset = RSTDataset(
            train_rst_path,
            train_nli_data_path,
            train_pre_emb_path,
            train_lexical_path,
            train_hyp_emb_path,
            batch_file_size,
            save_dir=train_pair_graph,
        )
        train_dataset.load_batch_files(0)
        for batch_data in train_dataset:
            print("safe")
