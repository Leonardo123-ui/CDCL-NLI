import re
import json
import datetime
import dgl
import torch
import numpy as np
import torch.nn.functional as F
import networkx as nx
import torch.nn as nn
from dgl.nn.pytorch import RelGraphConv
from dgl.dataloading import GraphDataLoader
import dgl.nn.pytorch as dglnn
from dgl.nn import GATConv
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import MT5ForConditionalGeneration, MT5Tokenizer
from hashlib import md5
import base64
import warnings
from typing import List, Dict, Tuple, Optional
import warnings

warnings.filterwarnings("ignore")


def encode_string(s):
    return int(md5(s.encode()).hexdigest(), 16) % (10**8)


def truncate_and_encode_text(text, max_input_length=300):
    """截断文本并进行Base64编码，添加警告信息"""
    if len(text) > max_input_length:
        warnings.warn(
            f"Text truncated from {len(text)} to {max_input_length} characters"
        )
        text = text[:max_input_length]
    return base64.b64encode(text.encode("utf-8"))


relation_types = [
    "Temporal",
    "TextualOrganization",
    "Joint",
    "Topic-Comment",
    "Comparison",
    "Condition",
    "Contrast",
    "Evaluation",
    "Topic-Change",
    "Summary",
    "Manner-Means",
    "Attribution",
    "Cause",
    "Background",
    "Enablement",
    "Explanation",
    "Same-Unit",
    "Elaboration",
    "span",  # 可以考虑去掉，因为关系不密切
    "lexical",  # 词汇链
]


def build_graph(node_features, node_types, rst_relations):
    """
    创建DGL图并添加节点及其特征。
    Args:
        node_features: List[Tuple] - 列表，每个元素为 (node_id, embedding, text)
        node_types: List[int] - 列表，表示节点类型（0表示父节点，1表示子节点）
        rst_relations: List[Tuple] - 列表，包含所有的RST关系，形式为 (parent_node, child_node, relation_type)
    Returns:
        DGLGraph: 构建好的图，包含节点特征和文本
    """
    num_nodes = len(node_types)

    # 将node_features转换为字典形式，分别存储embedding和text
    node_embeddings = {}
    node_texts = {}
    for node_id, embedding, text in node_features:
        node_embeddings[node_id] = embedding
        node_texts[node_id] = text

    # 创建从父节点到子节点及关系类型的映射
    parent_to_children = {}
    if rst_relations == ["NONE"]:
        rst_relations = [[0, 1, "span"]]

    for parent, child, rel_type in rst_relations:
        if parent not in parent_to_children:
            parent_to_children[parent] = []
        parent_to_children[parent].append((child, rel_type))

    # 初始化图数据结构
    graph_data = {("node", rel_type, "node"): ([], []) for rel_type in relation_types}

    for parent, children in parent_to_children.items():
        for child, rel_type in children:
            graph_data[("node", rel_type, "node")][0].append(parent)
            graph_data[("node", rel_type, "node")][1].append(child)
            # assuming undirected graph, add reverse edge
            graph_data[("node", rel_type, "node")][0].append(child)
            graph_data[("node", rel_type, "node")][1].append(parent)

    graph = dgl.heterograph(graph_data, num_nodes_dict={"node": num_nodes})

    # 初始化特征矩阵，确保所有节点都有特征
    feature_dim = len(next(iter(node_embeddings.values())))
    features = torch.zeros((num_nodes, feature_dim), dtype=torch.float32)
    for node_id, embedding in node_embeddings.items():
        features[node_id] = (
            torch.tensor(embedding, dtype=torch.float32).clone().detach()
        )

    # 初始化文本特征
    texts = [""] * num_nodes  # 默认空字符串
    for node_id, text in node_texts.items():
        texts[node_id] = text

    # 使用 networkx 进行拓扑排序，确定节点的层级顺序
    nx_graph = nx.DiGraph()
    nx_graph.add_edges_from([(parent, child) for parent, child, _ in rst_relations])
    topo_order = list(nx.topological_sort(nx_graph))

    # 从树的底部开始填充父节点特征和文本
    for node in reversed(topo_order):
        if node_types[node] == 1:  # 父节点
            if node in parent_to_children:
                # 更新embedding特征
                child_embeddings = [
                    features[child] for child, _ in parent_to_children[node]
                ]
                if child_embeddings:
                    parent_feature = sum(child_embeddings) / len(child_embeddings)
                    features[node] = parent_feature

    # 将特征添加到图中
    graph.ndata["feat"] = features.clone().detach()
    # 添加子节点和父节点标志
    node_types_tensor = torch.tensor(node_types, dtype=torch.long)
    graph.ndata["node_type"] = node_types_tensor

    target_length = 1024
    encoded_features = []
    for i, node_string in enumerate(texts):
        # 编码并填充到512长度
        encoded = truncate_and_encode_text(node_string)
        if len(encoded) > target_length:
            warnings.warn(
                f"Node {i}: Encoded length {len(encoded)} exceeds target length {target_length}"
            )
        # 使用b"\0"填充到目标长度
        padded = encoded.ljust(target_length, b"\0")
        encoded_features.append(padded)

    # 转换为张量
    padded_features = [torch.tensor(list(encoded)) for encoded in encoded_features]
    # 将所有节点的特征堆叠为二维张量
    padded_features_tensor = torch.stack(padded_features)

    # 验证最终张量的形状
    assert (
        padded_features_tensor.shape[1] == target_length
    ), f"Unexpected tensor shape: {padded_features_tensor.shape}"

    graph.ndata["text_encoded"] = padded_features_tensor  # 添加文本特征

    return graph


def merge_graphs(g_premise, g_hypothesis, lexical_chain, rel_names_short):
    num_nodes_premise = g_premise.num_nodes()
    num_nodes_hypothesis = g_hypothesis.num_nodes()

    # 获取所有可能的边类型，并过滤出关注的边类型
    all_edge_types = list(set(g_premise.etypes).union(set(g_hypothesis.etypes)))
    focused_edge_types = [etype for etype in all_edge_types if etype in rel_names_short]

    # 初始化合并图的数据结构
    combined_graph_data = {
        ("node", etype, "node"): ([], []) for etype in focused_edge_types
    }

    # 添加 g_premise 的边
    for etype in g_premise.etypes:
        if etype in rel_names_short:  # 只处理关注的边类型
            src, dst = g_premise.edges(etype=etype)
            combined_graph_data[("node", etype, "node")][0].extend(src.tolist())
            combined_graph_data[("node", etype, "node")][1].extend(dst.tolist())

    # 添加 g_hypothesis 的边，并调整索引
    for etype in g_hypothesis.etypes:
        if etype in rel_names_short:  # 只处理关注的边类型
            src, dst = g_hypothesis.edges(etype=etype)
            combined_graph_data[("node", etype, "node")][0].extend(
                (src + num_nodes_premise).tolist()
            )
            combined_graph_data[("node", etype, "node")][1].extend(
                (dst + num_nodes_premise).tolist()
            )

    # 添加 lexical_chain 的边，假设边的类型是 "lexical"
    if "lexical" in rel_names_short:  # 只处理关注的边类型
        src_nodes, dst_nodes = [], []
        for i in range(num_nodes_premise):
            for j in range(num_nodes_hypothesis):
                if lexical_chain[i][j] > 0:
                    src_nodes.append(i)
                    dst_nodes.append(
                        j + num_nodes_premise
                    )  # Offset by number of nodes in premise

        if src_nodes:
            edge_type = "lexical"
            if ("node", edge_type, "node") not in combined_graph_data:
                combined_graph_data[("node", edge_type, "node")] = ([], [])
            combined_graph_data[("node", edge_type, "node")][0].extend(src_nodes)
            combined_graph_data[("node", edge_type, "node")][1].extend(dst_nodes)
            combined_graph_data[("node", edge_type, "node")][0].extend(dst_nodes)
            combined_graph_data[("node", edge_type, "node")][1].extend(src_nodes)

    # 创建合并后的图
    num_combined_nodes = num_nodes_premise + num_nodes_hypothesis
    g_combined = dgl.heterograph(
        combined_graph_data, num_nodes_dict={"node": num_combined_nodes}
    )

    # 复制节点特征
    combined_features = torch.zeros(
        (num_combined_nodes, g_premise.ndata["feat"].shape[1]),
        dtype=torch.float32,
        device=g_premise.device,
    )
    # 获取两个图编码的维度
    d_premise = g_premise.ndata["text_encoded"].shape[1]
    d_hypothesis = g_hypothesis.ndata["text_encoded"].shape[1]

    # 使用最大的维度
    d_max = max(d_premise, d_hypothesis)  # 应该是512
    combined_texts = torch.zeros(
        (num_combined_nodes, d_max), dtype=torch.long, device=g_premise.device
    )
    combined_features[:num_nodes_premise] = g_premise.ndata["feat"].clone().detach()
    combined_features[num_nodes_premise:] = g_hypothesis.ndata["feat"].clone().detach()
    combined_texts[:num_nodes_premise, :d_premise] = (
        g_premise.ndata["text_encoded"].clone().detach()
    )  ##还不能短了
    combined_texts[num_nodes_premise:, :d_hypothesis] = (
        g_hypothesis.ndata["text_encoded"].clone().detach()
    )
    g_combined.ndata["feat"] = combined_features
    g_combined.ndata["text_encoded"] = combined_texts
    g_combined.ndata["node_type"] = torch.cat(
        [g_premise.ndata["node_type"], g_hypothesis.ndata["node_type"]]
    )

    return g_combined


def save_texts_to_json(generated_texts, golden_texts, filename):
    """
    将生成的文本和黄金标准文本保存到JSON文件中。

    :param generated_texts: 生成的文本列表
    :param golden_texts: 黄金标准文本列表
    :param filename: 要保存的文件名
    """
    try:
        # 确保生成的文本和黄金标准文本的数量相同
        if len(generated_texts) != len(golden_texts):
            print(
                f"警告：生成的文本数量 ({len(generated_texts)}) 与黄金标准文本数量 ({len(golden_texts)}) 不匹配。"
            )

        # 创建要保存的数据结构
        data = [
            {"generated": gen, "golden": gold}
            for gen, gold in zip(generated_texts, golden_texts)
        ]

        # 写入JSON文件
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"成功保存数据到文件：{filename}")

    except IOError as e:
        print(f"IO错误：无法写入文件 {filename}。错误信息：{str(e)}")
        # 尝试使用备用文件名
        backup_filename = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(backup_filename, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"已将数据保存到备用文件：{backup_filename}")
        except:
            print("无法保存到备用文件。")

    except Exception as e:
        print(f"发生错误：{str(e)}")
        print("尝试保存为纯文本格式...")
        try:
            with open(filename + ".txt", "w", encoding="utf-8") as f:
                for gen, gold in zip(generated_texts, golden_texts):
                    f.write(f"Generated: {gen}\n")
                    f.write(f"Golden: {gold}\n")
                    f.write("\n")
            print(f"已将数据保存为纯文本格式：{filename}.txt")
        except:
            print("无法保存为纯文本格式。")


class FusionMLP(nn.Module):
    def __init__(self):
        super(FusionMLP, self).__init__()
        self.fc = nn.Sequential(nn.Linear(2048, 1024), nn.ReLU())

    def forward(self, x):
        return self.fc(x)


def batch_triplet_loss_with_neutral(anchor, hyp, margin=1.0, neutral_weight=0.5):
    """
    计算批量三元组损失，支持 positive 和 negative 的维度为 [batch_size, 3, embedding_dim]。

    参数：
    - anchor: 前提的嵌入 (batch_size, embedding_dim)，形状为 [16, 1024]。
    - hyp: 假设的嵌入 (batch_size, 3, embedding_dim)，形状为 [16, 3, 1024]，包括 positive, neutral 和 negative。
    - margin: 三元组损失的间隔。
    - neutral_weight: 中立假设损失的权重。

    返回：
    - total_loss: 批量的总损失。
    """
    # 分别提取 positive（蕴含），neutral（中立），negative（矛盾）的嵌入
    positive = hyp[:, 0, :]  # [batch_size, 1024] 蕴含
    neutral = hyp[:, 1, :]  # [16, 1024] 中立
    negative = hyp[:, 2, :]  # [16, 1024] 矛盾

    # 计算 anchor 和 positive（蕴含）的距离
    dist_pos = F.pairwise_distance(anchor, positive, p=2)  # [batch_size]

    # 计算 anchor 和 negative（矛盾）的距离
    dist_neg = F.pairwise_distance(anchor, negative, p=2)  # [batch_size]

    # 计算 anchor 和 neutral（中立）的距离
    dist_neutral = F.pairwise_distance(anchor, neutral, p=2)  # [batch_size]

    # 三元组损失：正样本与 Anchor 应更接近
    loss_triplet = torch.clamp(dist_pos - dist_neg + margin, min=0).mean()

    # 中立损失：Neutral 应介于 Positive 和 Negative 之间
    loss_neutral = (
        torch.clamp(dist_pos - dist_neutral, min=0)
        + torch.clamp(dist_neutral - dist_neg, min=0)
    ).mean()

    # 总损失：三元组损失 + 中立损失
    total_loss = loss_triplet + neutral_weight * loss_neutral

    return total_loss


class RGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, rel_names):
        super().__init__()
        self.rel_names = rel_names
        # 关系权重
        self.relation_weights = nn.Parameter(torch.ones(len(rel_names)))
        self.softmax = nn.Softmax(dim=0)

        # 第一层异构图卷积
        self.conv1 = dglnn.HeteroGraphConv(
            {
                rel: dglnn.GATConv(
                    in_feats=in_dim,
                    out_feats=hidden_dim,
                    num_heads=4,
                    feat_drop=0.1,
                    attn_drop=0.1,
                    residual=True,
                    allow_zero_in_degree=True,
                )
                for rel in rel_names
            },
            aggregate="mean",
        )

        # 第二层异构图卷积
        self.conv2 = dglnn.HeteroGraphConv(
            {
                rel: dglnn.GATConv(
                    in_feats=hidden_dim * 4,
                    out_feats=out_dim,
                    num_heads=1,
                    feat_drop=0.1,
                    attn_drop=0.1,
                    residual=True,
                    allow_zero_in_degree=True,
                )
                for rel in rel_names
            },
            aggregate="mean",
        )

        self.dropout = nn.Dropout(0.1)

    def forward(self, g, inputs, return_attention=False):
        """
        Args:
            g: DGLHeteroGraph
            inputs: 节点特征字典 {"node_type": features}
            return_attention: 是否返回注意力权重
        """
        attention_weights = {} if return_attention else None

        # 第一层卷积
        h_dict = {}
        for ntype, features in inputs.items():
            h_dict[ntype] = features

        # 分别对每种关系进行处理以获取注意力权重
        if return_attention:
            for rel in g.canonical_etypes:
                _, etype, _ = rel
                # 使用子图来获取特定关系的注意力权重
                subgraph = g[rel]
                src_type, _, dst_type = rel

                # 获取第一层的注意力权重
                _, a1 = self.conv1.mods[etype](
                    subgraph, (h_dict[src_type], h_dict[dst_type]), get_attention=True
                )
                attention_weights[etype] = a1.mean(1).squeeze()  # 平均多头注意力

        # 正常的前向传播
        h = self.conv1(g, h_dict)
        h = {k: self.dropout(F.elu(v.flatten(1))) for k, v in h.items()}

        # 第二层卷积
        h = self.conv2(g, h)
        out = {k: v.squeeze(1) for k, v in h.items()}

        if return_attention:
            # 计算节点重要性
            return out, attention_weights
        return out


def decode_text_from_tensor(encoded_tensor):
    # 去除填充的空字节并转换为字节数据
    byte_data = bytes(encoded_tensor.tolist()).rstrip(b"\0")
    # 使用 Base64 解码还原原始文本
    decoded_text = base64.b64decode(byte_data).decode("utf-8")
    return decoded_text


def clean_text(text):
    if isinstance(text, list):
        return [clean_text(t) for t in text]
    return str(text).strip().replace("\x00", "").replace("\ufeff", "")


class ExplainableHeteroClassifier(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        n_classes,
        rel_names,
        device,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.label_smoothing = 0.1
        # 任务特定编码器
        self.rgat_classification = RGAT(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            rel_names=rel_names,
        )
        self.classifier = nn.Linear(hidden_dim * 2, n_classes)

        self.rgat_generation = RGAT(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            rel_names=relation_types,  # 使用生成任务特定的关系类型
        )
        self.node_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2),
        )
        self.pooling = dglnn.AvgPooling()
        # 假设嵌入的投影层
        # 添加投影层来调整维度
        self.query_proj = nn.Linear(in_dim, hidden_dim)
        self.key_proj = nn.Linear(in_dim, hidden_dim)
        self.value_proj = nn.Linear(in_dim, hidden_dim)

        # 关系类型权重
        self.relation_weights = nn.Parameter(torch.ones(len(rel_names)))
        self.softmax = nn.Softmax(dim=0)
        self.device = device
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True,  # 使用 batch_first=True 更直观
        )

    def freeze_task_components(self, task):
        """冻结特定任务的组件"""
        if task == "classification":
            for param in self.rgat_generation.parameters():
                param.requires_grad = False
            for param in self.node_classifier.parameters():
                param.requires_grad = False
        elif task == "generation":
            for param in self.rgat_classification.parameters():
                param.requires_grad = False
            for param in self.classifier.parameters():
                param.requires_grad = False

    def graph_info(self, graph):
        """
        获取图的编码信息
        Returns:
            node_feats: 节点特征
            attention_weights: 注意力权重
            graph_repr: 图表示
        """
        node_feats, attention_weights = self.rgat_classification(
            graph, {"node": graph.ndata["feat"]}, return_attention=True
        )
        graph_repr = self.pooling(graph, node_feats["node"])

        if torch.isnan(graph_repr).any():
            print("Warning: NaN detected in RGAT output.")

        return node_feats, attention_weights, graph_repr

    def classify(self, graph_repr, hyp_emb):
        """编码输入图"""
        # 1. 获取RGAT的节点表示和注意力权重
        combined_features = torch.cat([graph_repr, hyp_emb], dim=1)
        logits = self.classifier(combined_features)
        # outputs.update({"cli_logits": logits})  # {"logits": logits}
        return logits

    def extract_explanation(self, graph, hyp_emb):
        # 获取 RGAT 的节点特征和注意力权重
        node_feats, attention_weights = self.rgat_generation(
            graph, {"node": graph.ndata["feat"]}, return_attention=True
        )
        batch_size = hyp_emb.size(0)
        graph_nodes = graph.batch_num_nodes()  # 每个图的节点数量 type:Tensor
        total_nodes = graph.num_nodes()  # 所有图的总节点数

        # 使用返回的 attention_weights 直接计算节点的重要性
        node_importance = []
        for etype, weights in attention_weights.items():
            # 计算每个关系类型上的节点加权重要性
            edge_importance = torch.zeros(total_nodes, device=weights.device)
            for edge_idx, edge_weight in enumerate(weights):
                src, dst = graph.edges(etype=etype)
                edge_importance[dst[edge_idx]] += edge_weight.mean().item()
            node_importance.append(edge_importance)

        # 合并不同关系上的节点重要性
        node_importance = sum(node_importance)  # size [total_nodes] type Tensor
        # 按批次拆分节点重要性
        node_importance_split = torch.split(
            node_importance, graph_nodes.tolist()
        )  # 按图拆分
        node_importance_padded = torch.nn.utils.rnn.pad_sequence(
            node_importance_split, batch_first=True
        )  # [batch_size, max_nodes]
        node_feats_split = torch.split(node_feats["node"], graph_nodes.tolist())
        node_feats_padded = torch.nn.utils.rnn.pad_sequence(
            node_feats_split, batch_first=True
        )  # [batch_size, max_nodes, hidden_dim]
        weighted_node_feats_padded = (
            node_feats_padded * node_importance_padded.unsqueeze(-1)
        )

        # 创建注意力掩码
        max_nodes = node_importance_padded.size(1)
        attention_mask = torch.zeros(
            batch_size, max_nodes, dtype=torch.bool, device=graph.device
        )
        for i, length in enumerate(graph_nodes):
            attention_mask[i, length:] = True
        # 使用 hypothesis 对节点重要性进行加权
        hyp_expanded = hyp_emb.unsqueeze(1).expand(
            -1, max_nodes, -1
        )  # [batch_size, max_nodes, hidden_dim]

        # 使用注意力计算节点和 hypothesis 的交互特征
        attn_output, attn_weights = self.attention(
            query=hyp_expanded,
            key=node_feats_padded,
            value=node_feats_padded,
            key_padding_mask=attention_mask,
        )  # [batch_size, max_nodes, hidden_dim]

        # 合并重要性特征与交互特征
        # combined_features = torch.cat([node_feats_padded, attn_output], dim=-1)
        combined_features = torch.cat([weighted_node_feats_padded, attn_output], dim=-1)
        combined_features_split = [
            combined_features[i, : graph_nodes[i]] for i in range(batch_size)
        ]
        combined_features = torch.cat(
            combined_features_split, dim=0
        )  # [total_nodes, hidden_dim * 2]

        # 生成节点分类 logits
        node_logits = self.node_classifier(combined_features)

        return node_logits, attention_weights


class ExplainableHeteroClassifier_without_lexical_chain(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        n_classes,
        rel_names,
        device,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.label_smoothing = 0.1
        # 任务特定编码器
        self.rgat_classification = RGAT(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            rel_names=rel_names,
        )
        self.classifier = nn.Linear(hidden_dim * 3, n_classes)

        self.rgat_generation = RGAT(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            rel_names=relation_types,  # 使用生成任务特定的关系类型
        )
        self.node_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2),
        )
        self.pooling = dglnn.AvgPooling()
        # 假设嵌入的投影层
        # 添加投影层来调整维度
        self.query_proj = nn.Linear(in_dim, hidden_dim)
        self.key_proj = nn.Linear(in_dim, hidden_dim)
        self.value_proj = nn.Linear(in_dim, hidden_dim)

        # 关系类型权重
        self.relation_weights = nn.Parameter(torch.ones(len(rel_names)))
        self.softmax = nn.Softmax(dim=0)
        self.device = device
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True,  # 使用 batch_first=True 更直观
        )

    def freeze_task_components(self, task):
        """冻结特定任务的组件"""
        if task == "classification":
            for param in self.rgat_generation.parameters():
                param.requires_grad = False
            for param in self.node_classifier.parameters():
                param.requires_grad = False
        elif task == "generation":
            for param in self.rgat_classification.parameters():
                param.requires_grad = False
            for param in self.classifier.parameters():
                param.requires_grad = False

    def graph_info(self, graph):
        """
        获取图的编码信息
        Returns:
            node_feats: 节点特征
            attention_weights: 注意力权重
            graph_repr: 图表示
        """
        node_feats, attention_weights = self.rgat_classification(
            graph, {"node": graph.ndata["feat"]}, return_attention=True
        )
        graph_repr = self.pooling(graph, node_feats["node"])

        if torch.isnan(graph_repr).any():
            print("Warning: NaN detected in RGAT output.")

        return node_feats, attention_weights, graph_repr

    def classify(self, graph_repr, hyp_emb):
        """编码输入图"""
        # 1. 获取RGAT的节点表示和注意力权重
        combined_features = torch.cat([graph_repr, hyp_emb], dim=1)
        logits = self.classifier(combined_features)
        # outputs.update({"cli_logits": logits})  # {"logits": logits}
        return logits

    def extract_explanation(self, graph, hyp_emb):
        # 获取 RGAT 的节点特征和注意力权重
        node_feats, attention_weights = self.rgat_generation(
            graph, {"node": graph.ndata["feat"]}, return_attention=True
        )
        batch_size = hyp_emb.size(0)
        graph_nodes = graph.batch_num_nodes()  # 每个图的节点数量 type:Tensor
        total_nodes = graph.num_nodes()  # 所有图的总节点数

        # 使用返回的 attention_weights 直接计算节点的重要性
        node_importance = []
        for etype, weights in attention_weights.items():
            # 计算每个关系类型上的节点加权重要性
            edge_importance = torch.zeros(total_nodes, device=weights.device)
            for edge_idx, edge_weight in enumerate(weights):
                src, dst = graph.edges(etype=etype)
                edge_importance[dst[edge_idx]] += edge_weight.mean().item()
            node_importance.append(edge_importance)

        # 合并不同关系上的节点重要性
        node_importance = sum(node_importance)  # size [total_nodes] type Tensor
        # 按批次拆分节点重要性
        node_importance_split = torch.split(
            node_importance, graph_nodes.tolist()
        )  # 按图拆分
        node_importance_padded = torch.nn.utils.rnn.pad_sequence(
            node_importance_split, batch_first=True
        )  # [batch_size, max_nodes]
        node_feats_split = torch.split(node_feats["node"], graph_nodes.tolist())
        node_feats_padded = torch.nn.utils.rnn.pad_sequence(
            node_feats_split, batch_first=True
        )  # [batch_size, max_nodes, hidden_dim]

        weighted_node_feats_padded = (
            node_feats_padded * node_importance_padded.unsqueeze(-1)
        )
        # 创建注意力掩码
        max_nodes = node_importance_padded.size(1)
        attention_mask = torch.zeros(
            batch_size, max_nodes, dtype=torch.bool, device=graph.device
        )
        for i, length in enumerate(graph_nodes):
            attention_mask[i, length:] = True
        # 使用 hypothesis 对节点重要性进行加权
        hyp_expanded = hyp_emb.unsqueeze(1).expand(
            -1, max_nodes, -1
        )  # [batch_size, max_nodes, hidden_dim]

        # 使用注意力计算节点和 hypothesis 的交互特征
        attn_output, attn_weights = self.attention(
            query=hyp_expanded,
            key=node_feats_padded,
            value=node_feats_padded,
            key_padding_mask=attention_mask,
        )  # [batch_size, max_nodes, hidden_dim]

        # 合并重要性特征与交互特征
        # combined_features = torch.cat([node_feats_padded, attn_output], dim=-1)
        combined_features = torch.cat([weighted_node_feats_padded, attn_output], dim=-1)
        combined_features_split = [
            combined_features[i, : graph_nodes[i]] for i in range(batch_size)
        ]
        combined_features = torch.cat(
            combined_features_split, dim=0
        )  # [total_nodes, hidden_dim * 2]

        # 生成节点分类 logits
        node_logits = self.node_classifier(combined_features)

        return node_logits, attention_weights


class ExplainableHeteroClassifier_Single(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        n_classes,
        rel_names,
        device,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.label_smoothing = 0.1
        # 任务特定编码器
        self.rgat_classification = RGAT(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            rel_names=relation_types,
        )
        self.classifier = nn.Linear(hidden_dim * 2, n_classes)

        self.rgat_generation = RGAT(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            rel_names=relation_types,  # 使用生成任务特定的关系类型
        )
        self.node_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2),
        )
        self.pooling = dglnn.AvgPooling()
        # 假设嵌入的投影层
        # 添加投影层来调整维度
        self.query_proj = nn.Linear(in_dim, hidden_dim)
        self.key_proj = nn.Linear(in_dim, hidden_dim)
        self.value_proj = nn.Linear(in_dim, hidden_dim)

        # 关系类型权重
        self.relation_weights = nn.Parameter(torch.ones(len(rel_names)))
        self.softmax = nn.Softmax(dim=0)
        self.device = device
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True,  # 使用 batch_first=True 更直观
        )

    def freeze_task_components(self, task):
        """冻结特定任务的组件"""
        if task == "classification":
            for param in self.rgat_generation.parameters():
                param.requires_grad = False
            for param in self.node_classifier.parameters():
                param.requires_grad = False
        elif task == "generation":
            for param in self.rgat_classification.parameters():
                param.requires_grad = False
            for param in self.classifier.parameters():
                param.requires_grad = False

    def graph_info(self, graph):
        """
        获取图的编码信息
        Returns:
            node_feats: 节点特征
            attention_weights: 注意力权重
            graph_repr: 图表示
        """
        node_feats, attention_weights = self.rgat_classification(
            graph, {"node": graph.ndata["feat"]}, return_attention=True
        )
        graph_repr = self.pooling(graph, node_feats["node"])

        if torch.isnan(graph_repr).any():
            print("Warning: NaN detected in RGAT output.")

        return node_feats, attention_weights, graph_repr

    def classify(self, graph_repr, hyp_emb):
        """编码输入图"""
        # 1. 获取RGAT的节点表示和注意力权重
        combined_features = torch.cat([graph_repr, hyp_emb], dim=1)
        logits = self.classifier(combined_features)
        # outputs.update({"cli_logits": logits})  # {"logits": logits}
        return logits

    def extract_explanation(self, graph, hyp_emb):
        # 获取 RGAT 的节点特征和注意力权重
        node_feats, attention_weights = self.rgat_generation(
            graph, {"node": graph.ndata["feat"]}, return_attention=True
        )
        batch_size = hyp_emb.size(0)
        graph_nodes = graph.batch_num_nodes()  # 每个图的节点数量 type:Tensor
        total_nodes = graph.num_nodes()  # 所有图的总节点数

        # 使用返回的 attention_weights 直接计算节点的重要性
        node_importance = []
        for etype, weights in attention_weights.items():
            # 计算每个关系类型上的节点加权重要性
            edge_importance = torch.zeros(total_nodes, device=weights.device)
            for edge_idx, edge_weight in enumerate(weights):
                src, dst = graph.edges(etype=etype)
                edge_importance[dst[edge_idx]] += edge_weight.mean().item()
            node_importance.append(edge_importance)

        # 合并不同关系上的节点重要性
        node_importance = sum(node_importance)  # size [total_nodes] type Tensor
        # 按批次拆分节点重要性
        node_importance_split = torch.split(
            node_importance, graph_nodes.tolist()
        )  # 按图拆分
        node_importance_padded = torch.nn.utils.rnn.pad_sequence(
            node_importance_split, batch_first=True
        )  # [batch_size, max_nodes]
        node_feats_split = torch.split(node_feats["node"], graph_nodes.tolist())
        node_feats_padded = torch.nn.utils.rnn.pad_sequence(
            node_feats_split, batch_first=True
        )  # [batch_size, max_nodes, hidden_dim]
        weighted_node_feats_padded = (
            node_feats_padded * node_importance_padded.unsqueeze(-1)
        )

        # 创建注意力掩码
        max_nodes = node_importance_padded.size(1)
        attention_mask = torch.zeros(
            batch_size, max_nodes, dtype=torch.bool, device=graph.device
        )
        for i, length in enumerate(graph_nodes):
            attention_mask[i, length:] = True
        # 使用 hypothesis 对节点重要性进行加权
        hyp_expanded = hyp_emb.unsqueeze(1).expand(
            -1, max_nodes, -1
        )  # [batch_size, max_nodes, hidden_dim]

        # 使用注意力计算节点和 hypothesis 的交互特征
        attn_output, attn_weights = self.attention(
            query=hyp_expanded,
            key=node_feats_padded,
            value=node_feats_padded,
            key_padding_mask=attention_mask,
        )  # [batch_size, max_nodes, hidden_dim]

        # 合并重要性特征与交互特征
        # combined_features = torch.cat([node_feats_padded, attn_output], dim=-1)
        combined_features = torch.cat([weighted_node_feats_padded, attn_output], dim=-1)
        combined_features_split = [
            combined_features[i, : graph_nodes[i]] for i in range(batch_size)
        ]
        combined_features = torch.cat(
            combined_features_split, dim=0
        )  # [total_nodes, hidden_dim * 2]

        # 生成节点分类 logits
        node_logits = self.node_classifier(combined_features)

        return node_logits, attention_weights
