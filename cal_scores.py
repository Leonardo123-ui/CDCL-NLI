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
    Calculate BLEU score
    Args:
        generated_explanations: List of predicted explanation texts
        exp_text: List of reference explanation texts
    Returns:
        float: BLEU score
    """
    # Join strings in the list
    gen_exp = " ".join(generated_explanations)
    ref_exp = " ".join(exp_text)

    # Tokenize using the BART tokenizer
    gen_tokens = tokenizer.tokenize(gen_exp)
    ref_tokens = tokenizer.tokenize(ref_exp)

    # Calculate BLEU score
    bleu_score = sentence_bleu(
        [ref_tokens], gen_tokens, smoothing_function=smoothing_function
    )

    return bleu_score


def accuracy_score(y_true, y_pred):
    """
    Calculate accuracy

    Args:
    y_true: List or array of true labels
    y_pred: List or array of predicted labels

    Returns:
    float: Accuracy (proportion of correctly predicted samples)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    return np.mean(y_true == y_pred)


def precision_score(y_true, y_pred, average="macro"):
    """
    Calculate precision

    Args:
    y_true: List or array of true labels
    y_pred: List or array of predicted labels
    average: 'macro' for macro-average precision

    Returns:
    float: Precision
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

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
        raise ValueError("Only 'macro' averaging method is currently supported")


def recall_score(y_true, y_pred, average="macro"):
    """
    Calculate recall

    Args:
    y_true: List or array of true labels
    y_pred: List or array of predicted labels
    average: 'macro' for macro-average recall

    Returns:
    float: Recall
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

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
        raise ValueError("Only 'macro' averaging method is currently supported")


def f1_score(y_true, y_pred, average="macro"):
    """
    Calculate F1 score

    Args:
    y_true: List or array of true labels
    y_pred: List or array of predicted labels
    average: 'macro' for macro-average F1 score

    Returns:
    float: F1 score
    """
    precision = precision_score(y_true, y_pred, average=average)
    recall = recall_score(y_true, y_pred, average=average)

    if precision + recall == 0:
        return 0.0
    else:
        return 2 * (precision * recall) / (precision + recall)


def is_best_model(current_results, best_results, stage):
    """
    Determine whether the current model is the best model

    Args:
        current_metrics: Dictionary of current evaluation metrics
        best_metric: Historical best metric value
        stage: Training stage
    """
    if stage == "classification":
        current = current_results["classification_metrics"]["f1_macro_cli"]
    elif stage == "generation":
        current = current_results["generation_metrics"]["f1_macro_gen"]
    else:  # joint
        # For joint training, weighted combination can be used
        current = (
            current_results["classification_metrics"]["f1_macro_cli"] * 0.8
            + current_results["generation_metrics"]["f1_macro_gen"] * 0.2
        )

    return current > best_results
