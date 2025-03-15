import nltk # type: ignore
import spacy # type: ignore
from nltk.tokenize import word_tokenize # type: ignore
from nltk.corpus import stopwords # type: ignore
from nltk.stem import WordNetLemmatizer, SnowballStemmer # type: ignore
from nltk.tag import pos_tag # type: ignore
from nltk.chunk import ne_chunk # type: ignore
import string 

text = "Natural language processing helps computers understand and process human languages. Barack Obama was born in Hawaii."

# download required resources
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger")
nltk.download("wordnet")
nltk.download("words")
nltk.download("maxent_ne_chunker")

# 1. Tokenization
tokens = word_tokenize(text)
print("Tokens: ", tokens)

# 2. Lowercasing
lowercase_tokens = []
for token in tokens:
    lowercase_tokens.append(token.lower())
print("Lowercase tokens: ", lowercase_tokens)

# 3. Stopword removal
stop_words = set(stopwords.words('english')) # danh sách từ dừng tiếng Anh
filtered_tokens = []
for token in lowercase_tokens:
    if token not in stop_words:
        filtered_tokens.append(token)
print("Filtered tokens: ", filtered_tokens)

# 4. Stemming and lemmatization
stemmer = SnowballStemmer('english') # loại bỏ hậu tố của từ để lấy gốc từ 
lemmatizer = WordNetLemmatizer() # giảm từ về dạng gốc dựa trên từ loại của chúng
stemmed_tokens = []
lemmatized_tokens = []
for token in filtered_tokens:
    stemmed_tokens.append(stemmer.stem(token))
    lemmatized_tokens.append(lemmatizer.lemmatize(token))
print("Stemmed tokens: ", stemmed_tokens)
print("Lemmatized tokens: ", lemmatized_tokens)

# 5. Removing digit and punctuation (chọn lemmatized tokens để xử lý tiếp)
cleaned_tokens = []
for token in lemmatized_tokens:
    if not token.isdigit() and token not in string.punctuation:
        cleaned_tokens.append(token)
print("Cleaned tokens: ", cleaned_tokens)

# 6. POS tagging: quá trình xác định từ loại mỗi từ
pos_tags = pos_tag(cleaned_tokens)
print("POS tags: ", pos_tags)

# 7. Named entity recognition - NER
named_entities = ne_chunk(pos_tags)
print("Named entities: ", named_entities)

# Print result
print("\nOriginal text:", text) 
print("Preprocessed tokens:", cleaned_tokens) 
print("POS tags:", pos_tags) 
print("Named entities:", named_entities)

# Using spaCy for NER
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)
for ent in doc.ents:
    print(ent.text, ent.label_)
