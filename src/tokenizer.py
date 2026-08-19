import tokenize as python_tokenize
from io import StringIO

from gensim import corpora
from gensim.utils import simple_preprocess


SPECIAL_TOKENS = {
    "[PAD]": 0,
    "[UNK]": 1,
}


def tokenize_code(code):
    """
    Tokenize Python source code.
    """

    tokens = []

    for token in python_tokenize.generate_tokens(StringIO(code).readline):
        token_type = token.type
        token_string = token.string

        if token_type in (
            python_tokenize.ENCODING,
            python_tokenize.ENDMARKER,
            python_tokenize.NEWLINE,
            python_tokenize.NL,
            python_tokenize.INDENT,
            python_tokenize.DEDENT,
        ):
            continue

        tokens.append(token_string)

    return tokens


def tokenize_summary(summary):
    """
    Tokenize a natural language summary.
    """
    return simple_preprocess(summary)


def build_vocabulary(tokenized_texts):
    """
    Build a vocabulary from tokenized texts.
    """
    dictionary = corpora.Dictionary(tokenized_texts)
    dictionary.patch_with_special_tokens(SPECIAL_TOKENS)

    return dictionary


def tokens_to_ids(tokens, vocabulary):
    """
    Convert tokens into vocabulary IDs.
    """
    return vocabulary.doc2idx(tokens)


def build_code_vocabulary(dataset):
    """
    Build the vocabulary for Python source code.
    """
    tokenized_codes = [
        tokenize_code(example["code"])
        for example in dataset
    ]

    return build_vocabulary(tokenized_codes)


def build_summary_vocabulary(dataset):
    """
    Build the vocabulary for natural language summaries.
    """
    tokenized_summaries = [
        tokenize_summary(example["docstring"])
        for example in dataset
    ]

    return build_vocabulary(tokenized_summaries)