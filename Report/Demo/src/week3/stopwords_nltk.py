import nltk

from src.week1.demo_textpreprocess import stop_words

nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

#sample text
text = "This is an example sentence. However, it contains stop words!"
stop_words = set(stopwords.words('english'))

word_tokens = word_tokenize(text)
print("word_tokens: ", word_tokens)

filtered_text = [word for word in word_tokens if not word in stop_words]

print("filtered text: ", filtered_text)