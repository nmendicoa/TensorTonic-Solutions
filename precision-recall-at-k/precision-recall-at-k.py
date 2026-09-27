def precision_recall_at_k(recommended: list, relevant: list, k: int) -> list[float]:
    """
    Returns [precision, recall] as a list of two floats.
    """
    top_k = set(recommended[: k])
    relevant = set(relevant)

    precision_k = len(set(top_k).intersection(relevant)) / k
    recall_k = len(set(top_k).intersection(relevant)) / len(relevant)

    return [precision_k, recall_k]
    
    