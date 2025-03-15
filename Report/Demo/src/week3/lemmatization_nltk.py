import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

# create wordnetlematizer object
wnl = WordNetLemmatizer()

# single word lemmatization examples
list1 = ['kites', 'babies', 'dogs', 'flying', 'smilling', 'driving', 'died', 'tried', 'feet']

for word in list1:
    print(word + " ---> " + wnl.lemmatize(word))