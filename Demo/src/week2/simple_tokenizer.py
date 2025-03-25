class SimpleTokenizer:
    def __init__(self):
        self.vocab = {}
        self.add_word('<PAD>') #padding token
        self.add_word('<UNK>') #unknown word token

    def add_word(self, word):
        if word not in self.vocab:
            self.vocab[word] = len(self.vocab)

    def tokenize(self, text):
        return [word if word in self.vocab else '<UNK' for word in text.split()]

    def convert_tokens_to_ids(self, tokens):
        return [self.vocab[token] for token in tokens]

    def convert_ids_to_tokens(self, ids):
        reverse_vocab = {id: word for word, id in self.vocab.items()}
        return [reverse_vocab[id] for id in ids]

# example usage
tokenize = SimpleTokenizer()

#building vocabulary
sentences = [
    'I love Vietnam',
    'Vietnamese people are pretty friendly',
    'My mon loves cooking',
    'I am Vietnamese'
]

for sentence in sentences:
    for word in sentence.split():
        tokenize.add_word(word)

print("Vocabulary: ", tokenize.vocab)

# tokenizing a sentence
sentence = "I love Vietnam"
tokens = tokenize.tokenize(sentence)
print("Tokens: ", tokens)

# converting tokens to ids
ids = tokenize.convert_tokens_to_ids(tokens)
print("ids: ", ids)

# converting back
tokens_back = tokenize.convert_ids_to_tokens(ids)
print("tokens from ids: ", tokens_back)