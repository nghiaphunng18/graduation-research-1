from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

tokenize = get_tokenizer('basic_english')

sentences = [
    'I love Vietnam',
    'Vietnamese people are pretty friendly',
    'My mom loves cooking',
    'I am Vietnamese'
]

# token không có
unk_token = '<unk>'
# token để padding
pad_token = '<pad>'
# token bắt đầu câu
bos_token = '<bos>'
# token kết thúc câu
eos_token = '<eos>'

vocab = build_vocab_from_iterator(map(tokenize, sentences), specials=[unk_token, pad_token, bos_token, eos_token])
vocab.set_default_index(vocab[unk_token])

# building vocabulary
print(vocab.get_stoi())
print(vocab.get_itos())

# select vocabulary
print(vocab['love'])
print(vocab.lookup_indices(['i', 'love', 'vietnamese']))