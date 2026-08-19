import tokenize as python_tokenize
from io import StringIO
from collections import Counter
from nltk.tokenize import word_tokenize

SPECIAL_TOKENS = {
    "<PAD>": 0,
    "<UNK>": 1,
    "<SOS>": 2,
    "<EOS>": 3
}

def tokenize_code(code):
    """
    Tokenize Python source code using the built-in tokenize module.
    """
    tokens = []
    try:
        for token in python_tokenize.generate_tokens(StringIO(code).readline):
            if token.type in (
                python_tokenize.ENCODING,
                python_tokenize.ENDMARKER,
                python_tokenize.NEWLINE,
                python_tokenize.NL,
                python_tokenize.INDENT,
                python_tokenize.DEDENT,
            ):
                continue
            tokens.append(token.string)
    except Exception:
        # Fallback in case of malformed code
        tokens = code.split()
    return tokens

def tokenize_summary(summary):
    """
    Tokenize natural language summary using NLTK.
    """
    if not summary:
        return []
    return word_tokenize(summary.lower())


class Vocabulary:
    def __init__(self, specials=SPECIAL_TOKENS):
        self.stoi = specials.copy()
        self.itos = {idx: tok for tok, idx in specials.items()}

    def __len__(self):
        return len(self.stoi)

    def add_token(self, token):
        if token not in self.stoi:
            idx = len(self.stoi)
            self.stoi[token] = idx
            self.itos[idx] = token

    def build_vocab(self, tokenized_texts, max_size=50000, min_freq=2):
        counter = Counter()
        for text in tokenized_texts:
            counter.update(text)

        sorted_by_freq = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        for token, freq in sorted_by_freq:
            if freq >= min_freq and len(self.stoi) < max_size:
                self.add_token(token)

    def numericalize(self, tokens):
        return [self.stoi.get(token, self.stoi["<UNK>"]) for token in tokens]