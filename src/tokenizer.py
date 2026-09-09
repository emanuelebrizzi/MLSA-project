from collections import Counter

class SimpleTokenizer:
    """
    A basic tokenizer that builds a vocabulary from text and converts
    strings to sequences of integer IDs and vice versa.
    """
    def __init__(self, max_vocab_size=10000):
        self.max_vocab_size = max_vocab_size
        self.PAD_IDX = 0
        self.UNK_IDX = 1
        self.SOS_IDX = 2
        self.EOS_IDX = 3
        
        # Core dictionaries
        self.word2idx = {
            '<PAD>': self.PAD_IDX, 
            '<UNK>': self.UNK_IDX, 
            '<SOS>': self.SOS_IDX, 
            '<EOS>': self.EOS_IDX
        }
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        
    def build_vocab(self, texts):
        """
        Builds the vocabulary based on word frequencies in the provided texts.
        """
        counter = Counter()
        for text in texts:
            # Simple whitespace splitting
            counter.update(text.split())
            
        # Keep only the most common words, leaving space for special tokens
        most_common = counter.most_common(self.max_vocab_size - len(self.word2idx))
        
        for word, _ in most_common:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def encode(self, text, max_len=None, add_special_tokens=True):
        """
        Converts a text string into a list of integer IDs.
        """
        tokens = [self.word2idx.get(word, self.UNK_IDX) for word in text.split()]
        
        if add_special_tokens:
            tokens = [self.SOS_IDX] + tokens + [self.EOS_IDX]
            
        if max_len is not None:
            # Truncate if too long
            tokens = tokens[:max_len]
            # Pad if too short
            tokens += [self.PAD_IDX] * (max_len - len(tokens))
            
        return tokens

    def decode(self, token_ids, skip_special_tokens=True):
        """
        Converts a list of integer IDs back into a text string.
        """
        words = []
        for idx in token_ids:
            # Handle PyTorch tensors if passed by mistake
            if hasattr(idx, 'item'):
                idx = idx.item()
                
            if skip_special_tokens and idx in [self.PAD_IDX, self.SOS_IDX, self.EOS_IDX]:
                continue
                
            words.append(self.idx2word.get(idx, '<UNK>'))
            
        return ' '.join(words)
        
    def __len__(self):
        return len(self.word2idx)