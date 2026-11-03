def remove_stopwords(tokens: list, stopwords: list) -> list:
    """
    Returns a list of tokens.
    """
    # Write code here
    blocked = set(stopwords)
    return [t for t in tokens if t not in blocked] 