import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

s = '''Good muffins cost $3.88\\nin New York. Please buuy me two of them.\n\nThanks.'''
print(sent_tokenize(s))
