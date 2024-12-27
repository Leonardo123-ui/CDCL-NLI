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
    """Truncate text and perform Base64 encoding, adding a warning message if necessary."""
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
    "span",  # Can consider removing as the relationship is not close
    "lexical",  # Lexical chain
]


def build_graph(node_features, node_types, rst_relations):
    """
    Create a DGL graph and add nodes and their features.
    Args:
        node_features: List[Tuple] - List where each element is (node_id, embedding, text)
        node_types: List[int] - List indicating node types (0 for parent nodes, 1 for child nodes)
        rst_relations: List[Tuple] - List containing all RST relations in the format (parent_node, child_node, relation_type)
    Returns:
        DGLGraph: Constructed graph containing node features and text
    """
    num_nodes = len(node_types)

    # Convert node_features to a dictionary format, storing embeddings and text separately
    node_embeddings = {}
    node_texts = {}
    for node_id, embedding, text in node_features:
        node_embeddings[node_id] = embedding
        node_texts[node_id] = text

    # Create a mapping from parent nodes to child nodes and relation types
    parent_to_children = {}
    if rst_relations == ["NONE"]:
        rst_relations = [[0, 1, "span"]]

    for parent, child, rel_type in rst_relations:
        if parent not in parent_to_children:
            parent_to_children[parent] = []
        parent_to_children[parent].append((child, rel_type))

    # Initialize graph data structure
    graph_data = {("node", rel_type, "node"): ([], []) for rel_type in relation_types}

    for parent, children in parent_to_children.items():
        for child, rel_type in children:
            graph_data[("node", rel_type, "node")][0].append(parent)
            graph_data[("node", rel_type, "node")][1].append(child)
            # Assuming undirected graph, add reverse edge
            graph_data[("node", rel_type, "node")][0].append(child)
            graph_data[("node", rel_type, "node")][1].append(parent)

    graph = dgl.heterograph(graph_data, num_nodes_dict={"node": num_nodes})

    # Initialize feature matrix, ensuring all nodes have features
    feature_dim = len(next(iter(node_embeddings.values())))
    features = torch.zeros((num_nodes, feature_dim), dtype=torch.float32)
    for node_id, embedding in node_embeddings.items():
        features[node_id] = (
            torch.tensor(embedding, dtype=torch.float32).clone().detach()
        )

    # Initialize text features
    texts = [""] * num_nodes  # Default to empty string
    for node_id, text in node_texts.items():
        texts[node_id] = text

    # Perform topological sorting using networkx to determine node hierarchy
    nx_graph = nx.DiGraph()
    nx_graph.add_edges_from([(parent, child) for parent, child, _ in rst_relations])
    topo_order = list(nx.topological_sort(nx_graph))

    # Fill parent node features and text starting from the bottom of the tree
    for node in reversed(topo_order):
        if node_types[node] == 1:  # Parent node
            if node in parent_to_children:
                # Update embedding features
                child_embeddings = [
                    features[child] for child, _ in parent_to_children[node]
                ]
                if child_embeddings:
                    parent_feature = sum(child_embeddings) / len(child_embeddings)
                    features[node] = parent_feature

    # Add features to the graph
    graph.ndata["feat"] = features.clone().detach()
    # Add child and parent node markers
    node_types_tensor = torch.tensor(node_types, dtype=torch.long)
    graph.ndata["node_type"] = node_types_tensor

    target_length = 1024
    encoded_features = []
    for i, node_string in enumerate(texts):
        # Encode and pad to length 512
        encoded = truncate_and_encode_text(node_string)
        if len(encoded) > target_length:
            warnings.warn(
                f"Node {i}: Encoded length {len(encoded)} exceeds target length {target_length}"
            )
        # Pad to target length using b"\0"
        padded = encoded.ljust(target_length, b"\0")
        encoded_features.append(padded)

    # Convert to tensor
    padded_features = [torch.tensor(list(encoded)) for encoded in encoded_features]
    # Stack all node features into a 2D tensor
    padded_features_tensor = torch.stack(padded_features)

    # Verify the final tensor shape
    assert (
        padded_features_tensor.shape[1] == target_length
    ), f"Unexpected tensor shape: {padded_features_tensor.shape}"

    graph.ndata["text_encoded"] = padded_features_tensor  # Add text features

    return graph


def merge_graphs(g_premise, g_hypothesis, lexical_chain, rel_names_short):
    num_nodes_premise = g_premise.num_nodes()
    num_nodes_hypothesis = g_hypothesis.num_nodes()

    # Get all possible edge types and filter the ones of interest
    all_edge_types = list(set(g_premise.etypes).union(set(g_hypothesis.etypes)))
    focused_edge_types = [etype for etype in all_edge_types if etype in rel_names_short]

    # Initialize data structure for the combined graph
    combined_graph_data = {
        ("node", etype, "node"): ([], []) for etype in focused_edge_types
    }

    # Add edges from g_premise
    for etype in g_premise.etypes:
        if etype in rel_names_short:  # Only process relevant edge types
            src, dst = g_premise.edges(etype=etype)
            combined_graph_data[("node", etype, "node")][0].extend(src.tolist())
            combined_graph_data[("node", etype, "node")][1].extend(dst.tolist())

    # Add edges from g_hypothesis and adjust indices
    for etype in g_hypothesis.etypes:
        if etype in rel_names_short:  # Only process relevant edge types
            src, dst = g_hypothesis.edges(etype=etype)
            combined_graph_data[("node", etype, "node")][0].extend(
                (src + num_nodes_premise).tolist()
            )
            combined_graph_data[("node", etype, "node")][1].extend(
                (dst + num_nodes_premise).tolist()
            )

    # Add edges from lexical_chain, assuming edge type is "lexical"
    if "lexical" in rel_names_short:  # Only process relevant edge types
        src_nodes, dst_nodes = [], []
        for i in range(num_nodes_premise):
            for j in range(num_nodes_hypothesis):
                if lexical_chain[i][j] > 0:
                    src_nodes.append(i)
                    dst_nodes.append(
                        j + num_nodes_premise
                    )  # Offset by the number of nodes in premise

        if src_nodes:
            edge_type = "lexical"
            if ("node", edge_type, "node") not in combined_graph_data:
                combined_graph_data[("node", edge_type, "node")] = ([], [])
            combined_graph_data[("node", edge_type, "node")][0].extend(src_nodes)
            combined_graph_data[("node", edge_type, "node")][1].extend(dst_nodes)
            combined_graph_data[("node", edge_type, "node")][0].extend(dst_nodes)
            combined_graph_data[("node", edge_type, "node")][1].extend(src_nodes)

    # Create the combined graph
    num_combined_nodes = num_nodes_premise + num_nodes_hypothesis
    g_combined = dgl.heterograph(
        combined_graph_data, num_nodes_dict={"node": num_combined_nodes}
    )

    # Copy node features
    combined_features = torch.zeros(
        (num_combined_nodes, g_premise.ndata["feat"].shape[1]),
        dtype=torch.float32,
        device=g_premise.device,
    )
    # Get the encoded dimensions of both graphs
    d_premise = g_premise.ndata["text_encoded"].shape[1]
    d_hypothesis = g_hypothesis.ndata["text_encoded"].shape[1]

    # Use the maximum dimension
    d_max = max(d_premise, d_hypothesis)  # Should be 512
    combined_texts = torch.zeros(
        (num_combined_nodes, d_max), dtype=torch.long, device=g_premise.device
    )
    combined_features[:num_nodes_premise] = g_premise.ndata["feat"].clone().detach()
    combined_features[num_nodes_premise:] = g_hypothesis.ndata["feat"].clone().detach()
    combined_texts[:num_nodes_premise, :d_premise] = (
        g_premise.ndata["text_encoded"].clone().detach()
    )  ## Cannot be shorter
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
    Save generated texts and golden standard texts to a JSON file.

    :param generated_texts: List of generated texts
    :param golden_texts: List of golden standard texts
    :param filename: File name to save
    """
    try:
        # Ensure the number of generated texts matches the number of golden texts
        if len(generated_texts) != len(golden_texts):
            print(
                f"Warning: Number of generated texts ({len(generated_texts)}) does not match number of golden texts ({len(golden_texts)})."
            )

        # Create the data structure to save
        data = [
            {"generated": gen, "golden": gold}
            for gen, gold in zip(generated_texts, golden_texts)
        ]

        # Write to JSON file
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"Successfully saved data to file: {filename}")

    except IOError as e:
        print(f"IOError: Unable to write to file {filename}. Error: {str(e)}")
        # Try using a backup file name
        backup_filename = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(backup_filename, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"Data saved to backup file: {backup_filename}")
        except:
            print("Unable to save to backup file.")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print("Attempting to save as plain text...")
        try:
            with open(filename + ".txt", "w", encoding="utf-8") as f:
                for gen, gold in zip(generated_texts, golden_texts):
                    f.write(f"Generated: {gen}\n")
                    f.write(f"Golden: {gold}\n")
                    f.write("\n")
            print(f"Data saved as plain text: {filename}.txt")
        except:
            print("Unable to save as plain text.")


class FusionMLP(nn.Module):
    def __init__(self):
        super(FusionMLP, self).__init__()
        self.fc = nn.Sequential(nn.Linear(2048, 1024), nn.ReLU())

    def forward(self, x):
        return self.fc(x)


def batch_triplet_loss_with_neutral(anchor, hyp, margin=1.0, neutral_weight=0.5):
    """
    Compute batch triplet loss, supporting dimensions of positive and negative as [batch_size, 3, embedding_dim].

    Args:
    - anchor: Embedding of the premise (batch_size, embedding_dim), shape [16, 1024].
    - hyp: Embedding of the hypothesis (batch_size, 3, embedding_dim), shape [16, 3, 1024], including positive, neutral, and negative.
    - margin: Margin for triplet loss.
    - neutral_weight: Weight for neutral hypothesis loss.

    Returns:
    - total_loss: Total batch loss.
    """
    # Extract embeddings for positive (entailment), neutral, and negative (contradiction)
    positive = hyp[:, 0, :]  # [batch_size, 1024] entailment
    neutral = hyp[:, 1, :]  # [16, 1024] neutral
    negative = hyp[:, 2, :]  # [16, 1024] contradiction

    # Compute distance between anchor and positive (entailment)
    dist_pos = F.pairwise_distance(anchor, positive, p=2)  # [batch_size]

    # Compute distance between anchor and negative (contradiction)
    dist_neg = F.pairwise_distance(anchor, negative, p=2)  # [batch_size]

    # Compute distance between anchor and neutral
    dist_neutral = F.pairwise_distance(anchor, neutral, p=2)  # [batch_size]

    # Triplet loss: positive samples should be closer to the anchor
    loss_triplet = torch.clamp(dist_pos - dist_neg + margin, min=0).mean()

    # Neutral loss: Neutral should lie between Positive and Negative
    loss_neutral = (
        torch.clamp(dist_pos - dist_neutral, min=0)
        + torch.clamp(dist_neutral - dist_neg, min=0)
    ).mean()

    # Total loss: Triplet loss + Neutral loss
    total_loss = loss_triplet + neutral_weight * loss_neutral

    return total_loss


class RGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, rel_names):
        super().__init__()
        self.rel_names = rel_names
        # Relation weights
        self.relation_weights = nn.Parameter(torch.ones(len(rel_names)))
        self.softmax = nn.Softmax(dim=0)

        # First layer of heterogeneous graph convolution
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

        # Second layer of heterogeneous graph convolution
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
            inputs: Node feature dictionary {"node_type": features}
            return_attention: Whether to return attention weights
        """
        attention_weights = {} if return_attention else None

        # First convolution layer
        h_dict = {}
        for ntype, features in inputs.items():
            h_dict[ntype] = features

        # Process each relation to get attention weights
        if return_attention:
            for rel in g.canonical_etypes:
                _, etype, _ = rel
                # Use subgraph to get attention weights for specific relations
                subgraph = g[rel]
                src_type, _, dst_type = rel

                # Get attention weights from the first layer
                _, a1 = self.conv1.mods[etype](
                    subgraph, (h_dict[src_type], h_dict[dst_type]), get_attention=True
                )
                attention_weights[etype] = a1.mean(
                    1
                ).squeeze()  # Average multi-head attention

        # Normal forward propagation
        h = self.conv1(g, h_dict)
        h = {k: self.dropout(F.elu(v.flatten(1))) for k, v in h.items()}

        # Second convolution layer
        h = self.conv2(g, h)
        out = {k: v.squeeze(1) for k, v in h.items()}

        if return_attention:
            # Compute node importance
            return out, attention_weights
        return out


def decode_text_from_tensor(encoded_tensor):
    # Remove padded null bytes and convert to byte data
    byte_data = bytes(encoded_tensor.tolist()).rstrip(b"\0")
    # Use Base64 decoding to restore the original text
    decoded_text = base64.b64decode(byte_data).decode("utf-8")
    return decoded_text


def clean_text(text):
    if isinstance(text, list):
        return [clean_text(t) for t in text]
    return str(text).strip().replace("\x00", "").replace("\ufeff", "")


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
        # Task-specific encoder
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
            rel_names=relation_types,  # Use relation types specific to the generation task
        )
        self.node_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2),
        )
        self.pooling = dglnn.AvgPooling()
        # Projection layer for hypothesis embeddings
        # Add projection layers to adjust dimensions
        self.query_proj = nn.Linear(in_dim, hidden_dim)
        self.key_proj = nn.Linear(in_dim, hidden_dim)
        self.value_proj = nn.Linear(in_dim, hidden_dim)

        # Relation type weights
        self.relation_weights = nn.Parameter(torch.ones(len(rel_names)))
        self.softmax = nn.Softmax(dim=0)
        self.device = device
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True,  # Using batch_first=True for better readability
        )

    def freeze_task_components(self, task):
        """Freeze components of a specific task"""
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
        Retrieve graph encoding information
        Returns:
            node_feats: Node features
            attention_weights: Attention weights
            graph_repr: Graph representation
        """
        node_feats, attention_weights = self.rgat_classification(
            graph, {"node": graph.ndata["feat"]}, return_attention=True
        )
        graph_repr = self.pooling(graph, node_feats["node"])

        if torch.isnan(graph_repr).any():
            print("Warning: NaN detected in RGAT output.")

        return node_feats, attention_weights, graph_repr

    def classify(self, graph_repr, hyp_emb):
        """Encode input graph"""
        # 1. Retrieve RGAT node representations and attention weights
        combined_features = torch.cat([graph_repr, hyp_emb], dim=1)
        logits = self.classifier(combined_features)
        return logits

    def extract_explanation(self, graph, hyp_emb):
        # Retrieve RGAT node features and attention weights
        node_feats, attention_weights = self.rgat_generation(
            graph, {"node": graph.ndata["feat"]}, return_attention=True
        )
        batch_size = hyp_emb.size(0)
        graph_nodes = (
            graph.batch_num_nodes()
        )  # Number of nodes per graph (type: Tensor)
        total_nodes = graph.num_nodes()  # Total number of nodes across all graphs

        # Directly compute node importance using returned attention_weights
        node_importance = []
        for etype, weights in attention_weights.items():
            # Compute weighted importance for each relation type
            edge_importance = torch.zeros(total_nodes, device=weights.device)
            for edge_idx, edge_weight in enumerate(weights):
                src, dst = graph.edges(etype=etype)
                edge_importance[dst[edge_idx]] += edge_weight.mean().item()
            node_importance.append(edge_importance)

        # Combine node importance across different relations
        node_importance = sum(node_importance)  # size [total_nodes] (type: Tensor)
        # Split node importance by batch
        node_importance_split = torch.split(
            node_importance, graph_nodes.tolist()
        )  # Split by graph
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
        # Create attention mask
        max_nodes = node_importance_padded.size(1)
        attention_mask = torch.zeros(
            batch_size, max_nodes, dtype=torch.bool, device=graph.device
        )
        for i, length in enumerate(graph_nodes):
            attention_mask[i, length:] = True
        # Use hypothesis to weight node importance
        hyp_expanded = hyp_emb.unsqueeze(1).expand(
            -1, max_nodes, -1
        )  # [batch_size, max_nodes, hidden_dim]

        # Use attention to compute interaction features between nodes and hypothesis
        attn_output, attn_weights = self.attention(
            query=hyp_expanded,
            key=node_feats_padded,
            value=node_feats_padded,
            key_padding_mask=attention_mask,
        )  # [batch_size, max_nodes, hidden_dim]

        # Combine importance features with interaction features
        combined_features = torch.cat([weighted_node_feats_padded, attn_output], dim=-1)
        combined_features_split = [
            combined_features[i, : graph_nodes[i]] for i in range(batch_size)
        ]
        combined_features = torch.cat(
            combined_features_split, dim=0
        )  # [total_nodes, hidden_dim * 2]

        # Generate node classification logits
        node_logits = self.node_classifier(combined_features)

        return node_logits, attention_weights
