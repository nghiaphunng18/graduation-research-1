from tensorflow.keras.prepprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = [
             'I I I I I I love Vietnam Vietnam Vietnam Vietnam',
             'Vietnamese people are pretty friendly',
             'My My My My My My My My mom loves cooking',
             'I am Vietnamese'
]

tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")

tokenizer.fit_on_texts(sentences)

word_index = tokenizer.word_index
print(word_index)

new_sentences = [
    'I love dog',
    'I live in Hanoi'
]
new_sequences = tokenizer.texts_to_sequences(new_sentences)
print(new_sequences)

padding_sequences = pad_sequences(new_sequences)
print(padding_sequences)

