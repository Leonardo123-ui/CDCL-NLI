import numpy as np
from collections import defaultdict
from transformers import XLMRobertaTokenizer
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

tokenizer = XLMRobertaTokenizer.from_pretrained(
    "/mnt/nlp/yuanmengying/models/xlm_roberta_large"
)
smoothing_function = SmoothingFunction().method1


def calculate_graph_bleu(exp_text, generated_explanations):
    """
    计算BLEU分数
    Args:
        generated_explanations: 预测的解释文本列表
        exp_text: 参考的解释文本列表
    Returns:
        float: BLEU分数
    """
    # 将列表中的字符串连接起来
    gen_exp = " ".join(generated_explanations)
    ref_exp = " ".join(exp_text)

    # 使用BART tokenizer进行分词
    gen_tokens = tokenizer.tokenize(gen_exp)
    ref_tokens = tokenizer.tokenize(ref_exp)

    # 计算BLEU分数
    bleu_score = sentence_bleu(
        [ref_tokens], gen_tokens, smoothing_function=smoothing_function
    )

    return bleu_score


def accuracy_score(y_true, y_pred):
    """
    计算准确率

    参数:
    y_true: 真实标签列表或数组
    y_pred: 预测标签列表或数组

    返回:
    float: 准确率（正确预测的样本比例）
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError("y_true 和 y_pred 必须具有相同的长度。")

    return np.mean(y_true == y_pred)


def precision_score(y_true, y_pred, average="macro"):
    """
    计算精确率

    参数:
    y_true: 真实标签列表或数组
    y_pred: 预测标签列表或数组
    average: 'macro' 用于计算宏平均精确率

    返回:
    float: 精确率
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError("y_true 和 y_pred 必须具有相同的长度。")

    labels = np.unique(np.concatenate([y_true, y_pred]))
    precisions = []

    for label in labels:
        true_positives = np.sum((y_true == label) & (y_pred == label))
        predicted_positives = np.sum(y_pred == label)

        if predicted_positives == 0:
            precision = 0.0
        else:
            precision = true_positives / predicted_positives

        precisions.append(precision)

    if average == "macro":
        return np.mean(precisions)
    else:
        raise ValueError("目前只支持 'macro' 平均方法")


def recall_score(y_true, y_pred, average="macro"):
    """
    计算召回率

    参数:
    y_true: 真实标签列表或数组
    y_pred: 预测标签列表或数组
    average: 'macro' 用于计算宏平均召回率

    返回:
    float: 召回率
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError("y_true 和 y_pred 必须具有相同的长度。")

    labels = np.unique(np.concatenate([y_true, y_pred]))
    recalls = []

    for label in labels:
        true_positives = np.sum((y_true == label) & (y_pred == label))
        actual_positives = np.sum(y_true == label)

        if actual_positives == 0:
            recall = 0.0
        else:
            recall = true_positives / actual_positives

        recalls.append(recall)

    if average == "macro":
        return np.mean(recalls)
    else:
        raise ValueError("目前只支持 'macro' 平均方法")


def f1_score(y_true, y_pred, average="macro"):
    """
    计算 F1 分数

    参数:
    y_true: 真实标签列表或数组
    y_pred: 预测标签列表或数组
    average: 'macro' 用于计算宏平均 F1 分数

    返回:
    float: F1 分数
    """
    precision = precision_score(y_true, y_pred, average=average)
    recall = recall_score(y_true, y_pred, average=average)

    if precision + recall == 0:
        return 0.0
    else:
        return 2 * (precision * recall) / (precision + recall)


def is_best_model(current_results, best_results, stage):
    """
    判断当前模型是否为最佳模型

    Args:
        current_metrics: 当前评估指标字典
        best_metric: 历史最佳指标值
        stage: 训练阶段
    """
    if stage == "classification":
        current = current_results["classification_metrics"]["f1_macro_cli"]
    elif stage == "generation":
        current = current_results["generation_metrics"]["f1_macro_gen"]
    else:  # joint
        # 对于联合训练，可以使用加权组合

        current = (
            current_results["classification_metrics"]["f1_macro_cli"] * 0.8
            + current_results["generation_metrics"]["f1_macro_gen"] * 0.2
        )
        # current = current_results["classification_metrics"]["f1_macro_cli"]

    return current > best_results
